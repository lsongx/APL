import argparse
import numpy as np
import sys
import os
import os.path as osp
import time
import random

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

from dataset.cityscapes_dataset import CityscapesDataSetLMDB
from dataset.gta5_dataset import GTA5DataSetLMDB
from adv_model import StyleTrans


def main(args):
    writer = SummaryWriter(log_dir=args.tensorboard_log_dir)
    w, h = map(int, args.input_size.split(','))

    joint_transform = joint_transforms.Compose([
        joint_transforms.FreeScale((h, w)),
        joint_transforms.RandomHorizontallyFlip(),
    ])
    normalize = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    src_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
    ])
    tgt_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*normalize),
    ])
    val_input_transform = standard_transforms.Compose([
        extended_transforms.FreeScale((h, w)),
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*normalize),
    ])
    target_transform = extended_transforms.MaskToTensor()
    restore_transform = standard_transforms.ToPILImage()

    src_dataset = GTA5DataSetLMDB(
        args.data_dir, args.data_list,
        joint_transform=joint_transform,
        transform=src_input_transform, 
        target_transform=target_transform,
    )
    src_loader = data.DataLoader(
        src_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
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
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )

    style_trans = StyleTrans(args)
    style_trans.train(src_loader, tgt_loader, val_loader, writer)

    writer.close()


def get_options():
    parser = argparse.ArgumentParser()
    # train data params
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--input_size", type=str,
                        default='1024,512',
                        help="Comma-separated string with height and width of source images.")
    # train path params
    parser.add_argument("--data_dir", type=str,
                        default='/mnt/data-1/data/liangchen.song/seg/lmdb/gta5_trans_valid',
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data_list", type=str,
                        default='/home/users/liangchen.song/data/seg/image_list/gta5_trans_valid.txt',
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--data_dir_target", type=str, 
                        default='/mnt/data-1/data/liangchen.song/seg/lmdb/cityscapes_val',
                        # default='/mnt/data-1/data/liangchen.song/seg/lmdb/cityscapes_train',
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data_list_target", type=str, 
                        default='/home/users/liangchen.song/data/seg/image_list/cityscapes_val.txt',
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--data_dir_val", type=str,
                        default='/mnt/data-1/data/liangchen.song/seg/lmdb/cityscapes_val',
                        # default='/mnt/data-1/data/liangchen.song/seg/lmdb/cityscapes_train',
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data_list_val", type=str,
                        default='/home/users/liangchen.song/data/seg/image_list/cityscapes_val.txt',
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--model_path_prefix", type=str,
                        default='/home/users/liangchen.song/data/models')
    parser.add_argument("--resume", type=str,
                        default='/home/users/liangchen.song/data/trained_models/')
    # loss params
    parser.add_argument("--lambda_values", type=str, default='1,0,0,1e-2')
    # network params
    parser.add_argument("--seg_net", type=str, default='fcn')
    parser.add_argument("--n_blocks", type=int, default=9)
    parser.add_argument("--n_classes", type=int, default=19)
    # optimize params
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--D_learning_rate", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--warm_up_epoch", type=int, default=3)
    parser.add_argument("--num_epoch", type=int, default=10)
    # log params
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--show_img_freq", type=int, default=100)
    parser.add_argument("--checkpoint_freq", type=int, default=200)
    parser.add_argument("--save_path_prefix", type=str, default='./data/out')
    parser.add_argument("--fcn_name", type=str, default='fcn.pth')
    parser.add_argument("--tensorboard_log_dir", type=str, default='./logs')
    parser.add_argument("-f", type=str, default=None)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_options()
    main(args)
