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
        shuffle=True, num_workers=args.num_workers, pin_memory=True
    )

    upsample = nn.Upsample(size=(h_target, w_target),
                           mode='bilinear', align_corners=True)

    net = resnet101_ibn_a_deeplab(args.model_path_prefix, n_classes=args.n_classes)
    # optimizer = get_seg_optimizer(net, args)
    optimizer = torch.optim.SGD(
        net.parameters(), args.learning_rate, args.momentum
    )
    net = torch.nn.DataParallel(net)
    criterion = torch.nn.CrossEntropyLoss(
        size_average=False,
        ignore_index=args.ignore_index
    )

    num_batches = len(loader)
    for epoch in range(args.num_epoch):

        loss_rec = AverageMeter()
        data_time_rec = AverageMeter()
        batch_time_rec = AverageMeter()

        tem_time = time.time()
        for batch_index, batch_data in enumerate(loader):
            show_fig = (batch_index+1) % args.show_img_freq == 0
            iteration = batch_index+1+epoch*num_batches

            # poly_lr_scheduler(
            #     optimizer=optimizer,
            #     init_lr=args.learning_rate,
            #     iter=iteration - 1,
            #     lr_decay_iter=args.lr_decay,
            #     max_iter=args.num_epoch*num_batches,
            #     power=args.poly_power,
            # )

            net.train()
            # net.module.freeze_bn()
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
            if show_fig:
                base_lr = optimizer.param_groups[0]["lr"]
                output = torch.argmax(output, dim=1).detach()[0, ...].cpu()
                fig, axes = plt.subplots(2, 1, figsize=(12, 14))
                axes = axes.flat
                axes[0].imshow(colorize_mask(output.numpy()))
                axes[0].set_title(name[0])
                axes[1].imshow(colorize_mask(label[0, ...].numpy()))
                axes[1].set_title(f'seg_true_{base_lr:.6f}')
                writer.add_figure('A_seg', fig, iteration)

        mean_iu = test_miou(net, val_loader, upsample,'./ae_seg/dataset/info.json')
        torch.save(
            net.module.state_dict(),
            os.path.join(args.save_path_prefix, f'{epoch:d}_{mean_iu*100:.0f}.pth')
        )

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # train data params
    parser.add_argument("--batch_size", type=int, default=1,
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
                        default='/mnt/data-1/data/liangchen.song/seg/lmdb_data/gta5_mini',
                        # default='/mnt/data-1/data/liangchen.song/seg/lmdb_data/gta5_trans_valid',
                        # default='/mnt/data-1/data/liangchen.song/seg/lmdb_data/cityscapes_train',
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data_list", type=str,
                        default='/data-sdc/data/yonghao.xu/filelist/GTA5_imagelist_train.txt',
                        # default='/mnt/data-1/data/liangchen.song/seg/ori_gta_trans/valid_imagelist.txt',
                        # default='/mnt/data-1/data/liangchen.song/seg/cityscapes_train.txt',
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--data_dir_target", type=str, 
                        default='/mnt/data-1/data/liangchen.song/seg/lmdb_data/cityscapes_val',
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data_list_target", type=str, default='/data-sdc/data/yonghao.xu/filelist/cityscapes_labellist_val.txt',
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--model_path_prefix", type=str,
                        default='/home/users/liangchen.song/data/models')
    # optimize params
    parser.add_argument("--learning_rate", type=float, 
                        default=1e-10,
                        # default=2.5e-4,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--lr_decay", type=int, default=10)
    parser.add_argument("--poly_power", type=int, default=0.9)
    parser.add_argument("--num_epoch", type=int, default=5)
    # log params
    parser.add_argument("--print_freq", type=int, default=1)
    parser.add_argument("--show_img_freq", type=int, default=5)
    parser.add_argument("--save_path_prefix", type=str, default='./data/out')
    parser.add_argument("--tensorboard_log_dir", type=str, default='./logs')

    args = parser.parse_args()
    main(args)
