import argparse
import numpy as np
import sys
import os
import os.path as osp
import time
from PIL import Image

import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
import torch.nn.functional as F

import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
import torchvision.transforms as standard_transforms

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

from models.resnet_deeplab import resnet101_ibn_a_deeplab, get_seg_optimizer, poly_lr_scheduler
from dataset.cityscapes_dataset import CityscapesDataSetLMDB
from dataset.gta5_dataset import GTA5DataSetLMDB
from utils.tools import AverageMeter, colorize_mask
from utils.evaluate import test_miou


import encoding.utils as utils
from encoding.nn import SegmentationLosses, BatchNorm2d
from encoding.nn import SegmentationMultiLosses
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import get_segmentation_dataset
from encoding.models import PSP, MultiEvalModule


def main(args):
    writer = SummaryWriter(log_dir=args.tensorboard_log_dir)
    w, h = map(int, args.input_size.split(','))
    w_target, h_target = map(int, args.input_size_target.split(','))

    joint_transform = joint_transforms.Compose([
        joint_transforms.FreeScale((h, w)),
        joint_transforms.RandomHorizontallyFlip(),
    ])
    normalize = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*normalize),
    ])
    target_transform = extended_transforms.MaskToTensor()
    restore_transform = standard_transforms.ToPILImage()

    if '5' in args.data_dir:
        dataset = GTA5DataSetLMDB(
            args.data_dir, args.data_list,
            joint_transform=joint_transform,
            transform=input_transform, target_transform=target_transform,
        )
    else:
        dataset = CityscapesDataSetLMDB(
            args.data_dir, args.data_list,
            joint_transform=joint_transform,
            transform=input_transform, target_transform=target_transform,
        )
    loader = data.DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_dataset = CityscapesDataSetLMDB(
        args.data_dir_target, args.data_list_target,
        # joint_transform=joint_transform,
        transform=input_transform, target_transform=target_transform
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )


    upsample = nn.Upsample(size=(h_target, w_target),
                           mode='bilinear', align_corners=True)

    net = PSP(
        nclass = args.n_classes, backbone='resnet101', 
        root=args.model_path_prefix, norm_layer=BatchNorm2d,
    )

    params_list = [
        {'params': net.pretrained.parameters(), 'lr': args.learning_rate},
        {'params': net.head.parameters(), 'lr': args.learning_rate*10},
        {'params': net.auxlayer.parameters(), 'lr': args.learning_rate*10},
    ]
    optimizer = torch.optim.SGD(params_list,
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    criterion = SegmentationLosses(nclass=args.n_classes, aux=True, ignore_index=255)
    # criterion = SegmentationMultiLosses(nclass=args.n_classes, ignore_index=255)

    net = DataParallelModel(net).cuda()
    criterion = DataParallelCriterion(criterion).cuda()

    logger = utils.create_logger(args.tensorboard_log_dir, 'PSP_train')
    scheduler = utils.LR_Scheduler(args.lr_scheduler, args.learning_rate,
                                   args.num_epoch, len(loader), logger=logger,
                                   lr_step=args.lr_step)

    net_eval = Eval(net)

    num_batches = len(loader)
    best_pred = 0.0
    for epoch in range(args.num_epoch):

        loss_rec = AverageMeter()
        data_time_rec = AverageMeter()
        batch_time_rec = AverageMeter()

        tem_time = time.time()
        for batch_index, batch_data in enumerate(loader):
            scheduler(optimizer, batch_index, epoch, best_pred)
            show_fig = (batch_index+1) % args.show_img_freq == 0
            iteration = batch_index+1+epoch*num_batches

            net.train()
            img, label, name = batch_data
            img = img.cuda()
            label_cuda = label.cuda()
            data_time_rec.update(time.time()-tem_time)

            output = net(img)
            loss = criterion(output, label_cuda)
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
            # if show_fig:
            #     # base_lr = optimizer.param_groups[0]["lr"]
            #     output = torch.argmax(output[0][0], dim=1).detach()[0, ...].cpu()
            #     # fig, axes = plt.subplots(2, 1, figsize=(12, 14))
            #     # axes = axes.flat
            #     # axes[0].imshow(colorize_mask(output.numpy()))
            #     # axes[0].set_title(name[0])
            #     # axes[1].imshow(colorize_mask(label[0, ...].numpy()))
            #     # axes[1].set_title(f'seg_true_{base_lr:.6f}')
            #     # writer.add_figure('A_seg', fig, iteration)
            #     output_mask = np.asarray(colorize_mask(output.numpy()))
            #     label = np.asarray(colorize_mask(label[0,...].numpy()))
            #     image_out = np.concatenate([output_mask, label])
            #     writer.add_image('A_seg', image_out, iteration)

        mean_iu = test_miou(net_eval, val_loader, upsample,
                            './style_seg/dataset/info.json')
        torch.save(
            net.module.state_dict(),
            os.path.join(args.save_path_prefix, f'{epoch:d}_{mean_iu*100:.0f}.pth')
        )

    writer.close()


class Eval(object):
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        out = []
        with torch.no_grad():
            output_list = self.model(x)
            for output in output_list:
                out.append(output[0])
        return torch.cat(out)

    def eval(self):
        self.model.eval()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # train data params
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Number of images sent to the network in one step.")
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
    parser.add_argument("--data_dir", type=str,
                        default='/home/users/liangchen.song/data/seg/lmdb/gta5_mini',
                        # default='/home/users/liangchen.song/data/seg/lmdb/gta5_trans_valid',
                        # default='/home/users/liangchen.song/data/seg/lmdb/cityscapes_train',
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data_list", type=str,
                        default='/home/users/liangchen.song/data/seg/image_list/gta5_mini.txt',
                        # default='/mnt/data-1/data/liangchen.song/seg/ori_gta_trans/valid_imagelist.txt',
                        # default='/mnt/data-1/data/liangchen.song/seg/cityscapes_train.txt',
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--data_dir_target", type=str, 
                        default='/home/users/liangchen.song/data/seg/lmdb/cityscapes_val',
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data_list_target", type=str, 
                        default='/home/users/liangchen.song/data/seg/image_list/cityscapes_val.txt',
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--model_path_prefix", type=str,
                        default='/home/users/liangchen.song/data/models')
    # optimize params
    parser.add_argument("--learning_rate", type=float, 
                        default=3e-3,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--lr_scheduler", type=str, default='poly')
    parser.add_argument("--lr_step", type=int, default=None)
    parser.add_argument("--poly_power", type=int, default=0.9)
    parser.add_argument("--num_epoch", type=int, default=5)
    # log params
    parser.add_argument("--print_freq", type=int, default=1)
    parser.add_argument("--show_img_freq", type=int, default=5)
    parser.add_argument("--save_path_prefix", type=str, default='./data/out')
    parser.add_argument("--tensorboard_log_dir", type=str, default='./logs')

    args = parser.parse_args()
    main(args)
