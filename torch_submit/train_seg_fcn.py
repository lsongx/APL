import argparse
import numpy as np
import sys
import os
import os.path as osp
import time
import random
from functools import partial

import torch
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as standard_transforms

# from model.deeplab_vgg import DeeplabVGG
from models.fcn8s import FCN8s
from utils.tools import *
from utils.evaluate import _evaluate, test_miou
import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
from dataset.gta5_dataset import GTA5DataSetLMDB
from dataset.cityscapes_dataset import CityscapesDataSetLMDB


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
                        # default='/home/users/liangchen.song/data/seg/lmdb/gta5_mini',
                        default='/home/users/liangchen.song/data/seg/lmdb/gta5_trans_valid',
                        # default='/data-sdc/data/yonghao.xu/GTA-5',
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--tmp_path", type=str, default='/home/users/liangchen.song/data/tmp',
                        help="Path to tmp files.")
    parser.add_argument("--data_list", type=str, 
                        # default='/home/users/liangchen.song/data/seg/image_list/gta5_mini.txt',
                        default='/home/users/liangchen.song/data/seg/image_list/gta5_trans_valid.txt',
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
    parser.add_argument("--learning_rate", type=float, default=1e-9,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=0.99,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--teacher_alpha", type=float, default=0.99,
                        help="Teacher alpha in EMA.")
    parser.add_argument("--confidence_threshold", type=float, default=0,
                        help="Confidence threshold.")
    parser.add_argument("--st_weight", type=float, default=10,
                        help="Self ensembling weight.")
    parser.add_argument("--num_classes", type=int, default=19,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num_epoch", type=int, default=5,
                        help="Number of training epochs.")
    parser.add_argument("--model_path_prefix", type=str, default='/home/users/liangchen.song/data/models',
                        help="Where restore model parameters from.")
    parser.add_argument("--snapshot_dir", type=str, default='./data/out',
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight_decay", type=float, default=0.0005,
                        help="Regularization parameter for L2-loss.")
    parser.add_argument("--print_freq", type=int, default=1)
    parser.add_argument("--tensorboard_log_dir", type=str, default='./logs')
    return parser.parse_args()


def main():
    """Create the model and start the training."""
    args = get_arguments()

    w, h = map(int, args.input_size.split(','))

    w_target, h_target = map(int, args.input_size_target.split(','))

    # Create network
    student_net = FCN8s(args.num_classes, args.model_path_prefix)
    student_net = torch.nn.DataParallel(student_net)

    student_net = student_net.cuda()

    mean_std = ([103.939, 116.779, 123.68], [1.0, 1.0, 1.0])

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

    optimizer = optim.SGD(
        student_net.parameters(), lr=args.learning_rate,
        momentum=args.momentum, weight_decay=args.weight_decay
    )
    # optimizer = optim.Adam(
    #     student_net.parameters(), lr=args.learning_rate,
    #     weight_decay=args.weight_decay
    # )

    student_params = list(student_net.parameters())

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

    # src_criterion = torch.nn.CrossEntropyLoss(
    #     ignore_index=255, reduction='sum')
    src_criterion = torch.nn.CrossEntropyLoss(
        ignore_index=255, size_average=False
    )

    num_batches = len(src_loader)
    highest = 0

    for epoch in range(args.num_epoch):

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
            student_net.train()
            optimizer.zero_grad()

            # train with source

            # src_images, src_label, src_img_name, (load_time, trans_time) = src_data
            src_images, src_label, src_img_name = src_data
            src_images = src_images.cuda()
            src_label = src_label.cuda()
            data_time_rec.update(time.time()-tem_time)

            src_output = student_net(src_images)
            # src_output = interp(src_output)

            # Segmentation Loss
            cls_loss_value = src_criterion(src_output, src_label)
            cls_loss_value /= src_images.shape[0]

            total_loss = cls_loss_value
            total_loss.backward()
            optimizer.step()

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

            if (batch_index+1) % args.print_freq == 0:
                print(
                    f'Epoch [{epoch+1:d}/{args.num_epoch:d}][{batch_index+1:d}/{num_batches:d}]\t'
                    f'Time: {batch_time_rec.avg:.2f}   '
                    f'Data: {data_time_rec.avg:.2f}   '
                    # f'Load: {load_time_rec.avg:.2f}   '
                    # f'Trans: {trans_time_rec.avg:.2f}   '
                    f'Mean iu: {miu_rec.avg*100:.1f}   '
                    f'CLS: {cls_loss_rec.avg:.2f}'
                )

        miu = test_miou(student_net, tgt_val_loader,
                        upsample, './dataset/info.json')
        if miu > highest:
            torch.save(student_net.module.state_dict(), osp.join(
                args.snapshot_dir, f'final_fcn.pth'))
            highest = miu
            print('>'*50+f'save highest with {miu:.2%}')
    # torch.save(student_net.module.state_dict(), osp.join(
    #     args.snapshot_dir, f'final_fcn.pth'))


if __name__ == '__main__':
    main()
