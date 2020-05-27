import argparse
import numpy as np
import sys
import os
import os.path as osp
import time

import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
import torch.nn.functional as F

import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
import torchvision.transforms as standard_transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

from models.resnet_deeplab import resnet101_ibn_a_deeplab, get_seg_optimizer, poly_lr_scheduler
from models.gen_networks import define_G
from models.deeplabv3 import CriterionDSN, get_deeplabV3
from dataset.cityscapes_dataset import CityscapesDataSetLMDB
from dataset.gta5_dataset import GTA5DataSetLMDB
from utils.tools import AverageMeter, colorize_mask
from utils.evaluate import test_miou


def main(args):
    writer = SummaryWriter(log_dir=args.tensorboard_log_dir)
    w, h = map(int, args.input_size.split(','))
    w_target, h_target = map(int, args.input_size_target.split(','))

    joint_transform = joint_transforms.Compose([
        joint_transforms.FreeScale((h, w)),
        joint_transforms.RandomHorizontallyFlip(),
    ])
    normalize = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    tgt_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*normalize),
    ])
    target_transform = extended_transforms.MaskToTensor()
    restore_transform = standard_transforms.ToPILImage()

    if args.seg_net == 'fcn':
        mean_std = ([103.939, 116.779, 123.68], [1.0, 1.0, 1.0])
        val_input_transform = standard_transforms.Compose([
            extended_transforms.FreeScale((h, w)),
            extended_transforms.FlipChannels(),
            standard_transforms.ToTensor(),
            standard_transforms.Lambda(lambda x: x.mul_(255)),
            standard_transforms.Normalize(*mean_std),
        ])
    else:
        normalize = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        val_input_transform = standard_transforms.Compose([
            extended_transforms.FreeScale((h, w)),
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*normalize),
        ])

    tgt_dataset = CityscapesDataSetLMDB(
        args.data_dir_target, args.data_list_target,
        joint_transform=joint_transform,
        transform=tgt_input_transform,
        target_transform=target_transform,
    )
    tgt_loader = data.DataLoader(
        tgt_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_dataset = CityscapesDataSetLMDB(
        args.data_dir_val, args.data_list_val,
        transform=val_input_transform,
        target_transform=target_transform,
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    upsample = nn.Upsample(size=(h_target, w_target),
                           mode='bilinear', align_corners=True)

    if args.bn_sync:
        print('Using Sync BN')
        deeplabv3.BatchNorm2d = partial(InPlaceABNSync, activation='none')
    net = get_deeplabV3(args.num_classes, args.model_path_prefix)
    net_static = get_deeplabV3(args.num_classes, args.model_path_prefix)
    if not args.bn_sync:
        net.freeze_bn()
    net = torch.nn.DataParallel(net)

    file_name = os.path.join()
    net.load_state_dict(torch.load(file_name))
    net_static.load_state_dict(torch.load(file_name))
    for param in net_static.parameters():
        param.requires_grad = False

    optimizer = torch.optim.SGD(
        net.parameters(),
        args.learning_rate, args.momentum
    )
    net = torch.nn.DataParallel(net.cuda())
    net_static = torch.nn.DataParallel(net_static.cuda())
    # criterion = torch.nn.MSELoss()
    # criterion = torch.nn.SmoothL1Loss()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    gen_model = define_G()
    gen_model.load_state_dict(torch.load(
        os.path.join(args.resume, args.gen_name)
    ))
    gen_model.eval()
    for param in gen_model.parameters():
        param.requires_grad = False
    gen_model = torch.nn.DataParallel(gen_model.cuda())

    # for seg net
    def normalize(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        if args.seg_net == 'fcn':
            mean = [103.939, 116.779, 123.68]
            flip_x = torch.cat(
                [x[:, 2-i, :, :].unsqueeze(1) for i in range(3)],
                dim=1,
            )
            new_x = []
            for tem_x in flip_x:
                tem_new_x = []
                for c, m in zip(tem_x, mean):
                    tem_new_x.append(c.mul(255.0).sub(m).unsqueeze(0))
                new_x.append(torch.cat(tem_new_x, dim=0).unsqueeze(0))
            new_x = torch.cat(new_x, dim=0)
            return new_x
        else:
            for tem_x in x:
                for c, m, s in zip(tem_x, mean, std):
                    c = c.sub(m).div(s)
            return x

    def de_normalize(x, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        new_x = []
        for tem_x in x:
            tem_new_x = []
            for c, m, s in zip(tem_x, mean, std):
                tem_new_x.append(c.mul(s).add(s).unsqueeze(0))
            new_x.append(torch.cat(tem_new_x, dim=0).unsqueeze(0))
        new_x = torch.cat(new_x, dim=0)
        return new_x

    # ###################################################
    # direct test with gen
    # ###################################################
    print('Direct Test')
    mean_iu = test_miou(net, val_loader, upsample, './dataset/info.json')
    direct_input_transform = standard_transforms.Compose([
        extended_transforms.FreeScale((h, w)),
        standard_transforms.ToTensor(),
        standard_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    val_dataset_direct = CityscapesDataSetLMDB(
        args.data_dir_val, args.data_list_val,
        transform=direct_input_transform,
        target_transform=target_transform,
    )
    val_loader_direct = data.DataLoader(
        val_dataset_direct, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )

    class NewModel(object):
        def __init__(self, gen_net, val_net):
            self.gen_net = gen_net
            self.val_net = val_net

        def __call__(self, x):
            x = de_normalize(self.gen_net(x))
            new_x = normalize(x)
            out = self.val_net(new_x)
            return out

        def eval(self):
            self.gen_net.eval()
            self.val_net.eval()

    new_model = NewModel(gen_model, net)
    print('Test with Gen')
    mean_iu = test_miou(new_model, val_loader_direct,
                        upsample, './dataset/info.json')
    # return

    num_batches = len(tgt_loader)
    highest = 0

    for epoch in range(args.num_epoch):

        loss_rec = AverageMeter()
        data_time_rec = AverageMeter()
        batch_time_rec = AverageMeter()

        tem_time = time.time()
        for batch_index, batch_data in enumerate(tgt_loader):
            iteration = batch_index+1+epoch*num_batches

            net.train()
            net_static.eval()  # fine-tune use eval

            img, _, name = batch_data
            img = img.cuda()
            data_time_rec.update(time.time()-tem_time)

            with torch.no_grad():
                gen_output = gen_model(img)
                gen_seg_output_logits = net_static(
                    normalize(de_normalize(gen_output)))
            ori_seg_output_logits = net(normalize(de_normalize(img)))

            prob = torch.nn.Softmax(dim=1)
            max_value, label = torch.max(prob(gen_seg_output_logits), dim=1)
            label_mask = torch.zeros(label.shape, dtype=torch.uint8).cuda()
            for tem_label in range(19):
                tem_mask = label == tem_label
                if torch.sum(tem_mask) < 5:
                    continue
                value_vec = max_value[tem_mask]
                large_value = torch.topk(value_vec, int(
                    args.percent*value_vec.shape[0]))[0][0]
                large_mask = max_value > large_value
                label_mask = label_mask | (tem_mask & large_mask)
            label[label_mask] = 255

            # loss = criterion(ori_seg_output_logits, gen_seg_output_logits)
            loss = criterion(ori_seg_output_logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_rec.update(loss.item())
            writer.add_scalar('A_seg_loss', loss.item(), iteration)
            batch_time_rec.update(time.time()-tem_time)
            tem_time = time.time()

            if (batch_index+1) % args.print_freq == 0:
                print(
                    f'Epoch [{epoch+1:d}/{args.num_epoch:d}][{batch_index+1:d}/{num_batches:d}]\t'
                    f'Time: {batch_time_rec.avg:.2f}   '
                    f'Data: {data_time_rec.avg:.2f}   '
                    f'Loss: {loss_rec.avg:.2f}'
                )
            if iteration % args.checkpoint_freq == 0:
                mean_iu = test_miou(net, val_loader, upsample,
                                    './dataset/info.json', print_results=False)
                if mean_iu > highest:
                    torch.save(
                        net.module.state_dict(),
                        os.path.join(args.save_path_prefix,
                                     'cityscapes_best_fcn.pth')
                    )
                    highest = mean_iu
                    print(f'save fcn model with {mean_iu:.2%}')

    print(('-'*100+'\n')*3)
    print('>'*50+'Final Model')
    net.module.load_state_dict(
        torch.load(os.path.join(args.save_path_prefix,
                                'cityscapes_best_fcn.pth'))
    )
    mean_iu = test_miou(net, val_loader,
                        upsample, './dataset/info.json')

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # train data params
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--seg_net", type=str, default='fcn',)
    parser.add_argument("--num_workers", type=int, default=1,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--input_size", type=str,
                        # default='1536,768',
                        default='1024,512',
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--input_size_target", type=str, default='2048,1024',)
    parser.add_argument("--n_classes", type=int, default=19)
    parser.add_argument("--ignore_index", type=int, default=255)
    # train path params
    parser.add_argument("--data_dir_target", type=str,
                        # default='/mnt/data-1/data/liangchen.song/seg/lmdb/cityscapes_train',
                        default='/mnt/data-1/data/liangchen.song/seg/lmdb/cityscapes_val',
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data_list_target", type=str,
                        # default='/mnt/data-1/data/liangchen.song/seg/cityscapes_train.txt',
                        default='/mnt/data-1/data/liangchen.song/seg/cityscapes_val.txt',
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--data_dir_val", type=str,
                        default='/mnt/data-1/data/liangchen.song/seg/lmdb/cityscapes_val',
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data_list_val", type=str,
                        # default='/data-sdc/data/yonghao.xu/filelist/cityscapes_labellist_val.txt',
                        default='/mnt/data-1/data/liangchen.song/seg/cityscapes_val.txt',
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--model_path_prefix", type=str,
                        default='/home/users/liangchen.song/data/models')
    # optimize params
    parser.add_argument("--learning_rate", type=float,
                        default=2e-6,
                        # default=2.5e-4,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--lr_decay", type=int, default=10)
    parser.add_argument("--poly_power", type=int, default=0.9)
    parser.add_argument("--num_epoch", type=int, default=5)
    parser.add_argument("--percent", type=float, default=0.5)
    # log params
    parser.add_argument("--print_freq", type=int, default=1)
    parser.add_argument("--show_img_freq", type=int, default=5)
    parser.add_argument("--checkpoint_freq", type=int, default=1000)
    parser.add_argument("--save_path_prefix", type=str, default='./data/out')
    parser.add_argument(
        "--resume", type=str, default='/mnt/data-1/data/liangchen.song/trained_models')
    parser.add_argument("--fcn_name", type=str, default='fcn.pth')
    parser.add_argument("--gen_name", type=str, default='gen.pth')
    parser.add_argument("--tensorboard_log_dir", type=str, default='./logs')

    args = parser.parse_args()
    main(args)
