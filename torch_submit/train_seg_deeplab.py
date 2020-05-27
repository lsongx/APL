import argparse
import numpy as np
import sys
import os
import os.path as osp
import time
from functools import partial

import torch
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as standard_transforms

sys.path.append(os.path.join(os.getcwd(), 'models', 'deeplabv3_lib'))
import models.deeplabv3 as deeplabv3
from models.deeplabv3 import CriterionDSN, get_deeplabV3
from models.deeplabv3_lib import InPlaceABN, InPlaceABNSync
from utils.tools import *
from utils.evaluate import _evaluate, test_miou
import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
from dataset.gta5_dataset import GTA5DataSetLMDB
from dataset.cityscapes_dataset import CityscapesDataSetLMDB


np.random.seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data_dir", type=str, 
                        default='/home/users/liangchen.song/data/seg/lmdb/gta5_mini',
                        # default='/home/users/liangchen.song/data/seg/lmdb/gta5_trans_valid',
                        # default='/data-sdc/data/yonghao.xu/GTA-5',
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--tmp_path", type=str, default='/home/users/liangchen.song/data/tmp',
                        help="Path to tmp files.")
    parser.add_argument("--data_list", type=str, 
                        default='/home/users/liangchen.song/data/seg/image_list/gta5_mini.txt',
                        # default='/home/users/liangchen.song/data/seg/image_list/gta5_trans_valid.txt',
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore_label", type=int, default=255,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input_size", type=str,
                        default='1024,512',
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data_dir_target", type=str, 
                        default='/home/users/liangchen.song/data/seg/lmdb/cityscapes_val',
                        # default='/data-sdc/data/yonghao.xu/cityscapes/',
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data_list_target", type=str, 
                        default='/home/users/liangchen.song/data/seg/image_list/cityscapes_val.txt',
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input_size_target", type=str, default='2048,1024',
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--learning_rate", type=float, default=2.5e-4,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--power", type=float, default=0.9,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=0.99,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num_classes", type=int, default=19,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num_epoch", type=int, default=5,
                        help="Number of training epochs.")
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--model_path_prefix", type=str, default='/home/users/liangchen.song/data/models',
                        help="Where restore model parameters from.")
    parser.add_argument("--snapshot_dir", type=str, default='./data/out',
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight_decay", type=float, default=0.0005,
                        help="Regularization parameter for L2-loss.")
    parser.add_argument("--print_freq", type=int, default=1)
    parser.add_argument("--eval_freq", type=int, default=20)
    parser.add_argument("--tensorboard_log_dir", type=str, default='./logs')
    parser.add_argument('--bn_sync', type=int, default=0)
    return parser.parse_args()


def main():
    """Create the model and start the training."""
    args = get_arguments()

    w, h = map(int, args.input_size.split(','))

    w_target, h_target = map(int, args.input_size_target.split(','))

    # Create network
    if args.bn_sync:
        print('Using Sync BN')
        deeplabv3.BatchNorm2d = partial(InPlaceABNSync, activation='none')
    net = get_deeplabV3(args.num_classes, args.model_path_prefix)
    if not args.bn_sync:
        net.freeze_bn()
    net = torch.nn.DataParallel(net)

    net = net.cuda()

    mean_std = ([104.00698793, 116.66876762, 122.67891434], [1.0, 1.0, 1.0])

    train_joint_transform = joint_transforms.Compose([
        joint_transforms.FreeScale((h, w)),
    ])
    input_transform = standard_transforms.Compose([
        extended_transforms.FlipChannels(),
        standard_transforms.ToTensor(),
        standard_transforms.Lambda(lambda x: x.mul_(255)),
        standard_transforms.Normalize(*mean_std),
    ])
    val_input_transform = standard_transforms.Compose([
        extended_transforms.FreeScale((h, w)),
        extended_transforms.FlipChannels(),
        standard_transforms.ToTensor(),
        standard_transforms.Lambda(lambda x: x.mul_(255)),
        standard_transforms.Normalize(*mean_std),
    ])
    target_transform = extended_transforms.MaskToTensor()
    # show img
    restore_transform = standard_transforms.Compose([
        extended_transforms.DeNormalize(*mean_std),
        standard_transforms.Lambda(lambda x: x.div_(255)),
        standard_transforms.ToPILImage(),
        extended_transforms.FlipChannels(),
    ])
    visualize = standard_transforms.ToTensor()

    if '5' in args.data_dir:
        src_dataset = GTA5DataSetLMDB(
            args.data_dir, args.data_list, 
            joint_transform=train_joint_transform,
            transform=input_transform, target_transform=target_transform,
        )
    else:
        src_dataset = CityscapesDataSetLMDB(
            args.data_dir, args.data_list,
            joint_transform=train_joint_transform,
            transform=input_transform, target_transform=target_transform,
        )
    src_loader = data.DataLoader(
        src_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True
    )

    tgt_val_dataset = CityscapesDataSetLMDB(
        args.data_dir_target, args.data_list_target,
        # no val resize
        # joint_transform=val_joint_transform,
        transform=val_input_transform, target_transform=target_transform,
    )
    tgt_val_loader = data.DataLoader(
        tgt_val_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True,
    )

    # freeze bn
    for module in net.module.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            for param in module.parameters():
                param.requires_grad=False
    optimizer = optim.SGD(
        [
            {'params': filter(lambda p: p.requires_grad, net.module.parameters()), 
             'lr': args.learning_rate}
        ],
        lr=args.learning_rate,
        momentum=args.momentum, weight_decay=args.weight_decay
    )
    # optimizer = optim.Adam(
    #     net.parameters(), lr=args.learning_rate,
    #     weight_decay=args.weight_decay
    # )

    # interp = partial(
    #     nn.functional.interpolate,
    #     size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True
    # )
    # interp_tgt = partial(
    #     nn.functional.interpolate,
    #     size=(h_target, w_target), mode='bilinear', align_corners=True
    # )
    upsample = nn.Upsample(size=(h_target, w_target), mode='bilinear')

    n_class = args.num_classes

    # criterion = torch.nn.CrossEntropyLoss(
    #     ignore_index=255, reduction='sum')
    # criterion = torch.nn.CrossEntropyLoss(
    #     ignore_index=255, size_average=True
    # )
    criterion = CriterionDSN(
        ignore_index=255,
        # size_average=False
    )

    num_batches = len(src_loader)
    max_iter = args.iterations
    i_iter = 0
    highest_miu = 0

    while True:

        cls_loss_rec = AverageMeter()
        aug_loss_rec = AverageMeter()
        mask_rec = AverageMeter()
        confidence_rec = AverageMeter()
        miu_rec = AverageMeter()
        data_time_rec = AverageMeter()
        batch_time_rec = AverageMeter()
        # load_time_rec = AverageMeter()
        # trans_time_rec = AverageMeter()

        tem_time = time.time()
        for batch_index, src_data in enumerate(src_loader):
            i_iter += 1
            lr = adjust_learning_rate(args, optimizer, i_iter, max_iter)
            net.train()
            optimizer.zero_grad()

            # train with source

            # src_images, src_label, src_img_name, (load_time, trans_time) = src_data
            src_images, src_label, src_img_name = src_data
            src_images = src_images.cuda()
            src_label = src_label.cuda()
            data_time_rec.update(time.time()-tem_time)

            src_output = net(src_images)
            # src_output = interp(src_output)

            # Segmentation Loss
            cls_loss_value = criterion(src_output, src_label)

            total_loss = cls_loss_value
            total_loss.backward()
            optimizer.step()

            src_output = torch.nn.functional.upsample(
                input=src_output[0], size=(h, w), mode='bilinear', align_corners=True
            )

            _, predict_labels = torch.max(src_output, 1)
            lbl_pred = predict_labels.detach().cpu().numpy()
            lbl_true = src_label.detach().cpu().numpy()
            _, _, _, mean_iu, _ = _evaluate(lbl_pred, lbl_true, 19)

            cls_loss_rec.update(cls_loss_value.detach_().item())
            miu_rec.update(mean_iu)
            # load_time_rec.update(torch.mean(load_time).item())
            # trans_time_rec.update(torch.mean(trans_time).item())

            batch_time_rec.update(time.time()-tem_time)
            tem_time = time.time()

            if i_iter % args.print_freq == 0:
                print(
                    # f'Epoch [{epoch+1:d}/{args.num_epoch:d}][{batch_index+1:d}/{num_batches:d}]\t'
                    f'Iter: [{i_iter}/{max_iter}]\t'
                    f'Time: {batch_time_rec.avg:.2f}   '
                    f'Data: {data_time_rec.avg:.2f}   '
                    # f'Load: {load_time_rec.avg:.2f}   '
                    # f'Trans: {trans_time_rec.avg:.2f}   '
                    f'Mean iu: {miu_rec.avg*100:.1f}   '
                    f'CLS: {cls_loss_rec.avg:.2f}'
                )
            if i_iter % args.eval_freq == 0:
                miu = test_miou(net, tgt_val_loader,
                                upsample, './dataset/info.json')
                if miu > highest_miu:
                    torch.save(net.module.state_dict(), osp.join(
                        args.snapshot_dir, f'{i_iter:d}_{miu*1000:.0f}.pth'))
                    highest_miu = miu
                print(f'>>>>>>>>>Learning Rate {lr}<<<<<<<<<')
            if i_iter == max_iter:
                return


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def adjust_learning_rate(args, optimizer, i_iter, max_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.learning_rate, i_iter, max_iter, args.power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


if __name__ == '__main__':
    main()
