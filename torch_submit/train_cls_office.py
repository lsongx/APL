import os
import argparse
import time
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import matplotlib
matplotlib.use('Agg')
from tensorboardX import SummaryWriter

from models.resnet import resnet50
from utils.evaluate import cls_evaluate
from adv_model_cls import StyleTrans




class Office31(torch.utils.data.Dataset):

    def __init__(
        self, img_root, batch_size=32, transform=None, normalize=None,
        num_workers=4, pin_memory=True, drop_last=False,
    ):
        if normalize is None:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        test_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            normalize,
        ])

        w_root = os.path.join(img_root, 'webcam', 'images')
        d_root = os.path.join(img_root, 'dslr', 'images')
        a_root = os.path.join(img_root, 'amazon', 'images')

        self.w_dataset = datasets.ImageFolder(root=w_root, transform=transform)
        self.d_dataset = datasets.ImageFolder(root=d_root, transform=transform)
        self.a_dataset = datasets.ImageFolder(root=a_root, transform=transform)
        self.w_dataset_test = datasets.ImageFolder(
            root=w_root, transform=transform)
        self.d_dataset_test = datasets.ImageFolder(
            root=d_root, transform=transform)
        self.a_dataset_test = datasets.ImageFolder(
            root=a_root, transform=transform)

        self.loader = {}
        self.loader['w'] = DataLoader(
            self.w_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last,
        )
        self.loader['d'] = DataLoader(
            self.d_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last,
        )
        self.loader['a'] = DataLoader(
            self.a_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last,
        )
        self.loader_test = {}
        self.loader_test['w'] = DataLoader(
            self.w_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        self.loader_test['d'] = DataLoader(
            self.d_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        self.loader_test['a'] = DataLoader(
            self.a_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.classes = self.w_dataset.classes
        self.class_to_idx = self.w_dataset.class_to_idx


class ClsNet(nn.Module):
    def __init__(
        self, model_path_prefix, num_classes,
        embed=1024, dropout=0,
    ):
        super(ClsNet, self).__init__()
        self.num_classes = num_classes
        self.embed = embed

        self.base = resnet50(
            pretrained=True,
            model_path_prefix=model_path_prefix,
        )

        self.class_classifier = nn.Sequential()
        if self.embed == 0:
            self.class_classifier.add_module('c_final', nn.Linear(2048, self.num_classes))
        else:
            self.class_classifier.add_module('c_fc1', nn.Linear(2048, self.embed))
            self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(self.embed))
            self.class_classifier.add_module('c_relu1', nn.ReLU(inplace=True))
            self.class_classifier.add_module('c_drop1', nn.Dropout(dropout))

    def forward(self, x):
        x = self.base(x)
        feature = x.view(x.shape[0], -1)
        class_output = self.class_classifier(feature)
        return class_output


def main(args):
    print(f'{args.src_idx} -----> {args.tgt_idx}')
    #-------------------------------------------------------------
    # STEP 0. Prepare Data
    #-------------------------------------------------------------
    data = Office31(
        args.data_path, args.batch_size,
        num_workers=args.num_workers, drop_last=True
    )
    num_classes = len(data.classes)

    src_loader = data.loader[args.src_idx]
    tgt_loader = data.loader[args.tgt_idx]
    tgt_loader_test = data.loader_test[args.tgt_idx]

    # to [-1, 1], for generating
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
    )
    data_gen = Office31(
        args.data_path, args.batch_size//2, normalize=normalize,
        num_workers=args.num_workers, drop_last=True
    )
    src_gen_loader = data_gen.loader[args.src_idx]
    tgt_gen_loader = data_gen.loader[args.tgt_idx]
    tgt_gen_loader_test = data_gen.loader_test[args.tgt_idx]

    #-------------------------------------------------------------
    # STEP 1. Train Model on Source
    #-------------------------------------------------------------
    cls_net = ClsNet(
        args.model_path_prefix, num_classes,
        args.embed, args.dropout
    )
    cls_net = torch.nn.DataParallel(cls_net).cuda()

    cls_optimizer = torch.optim.SGD(
        [
            {'params': cls_net.module.base.parameters(), 'lr': args.lr/10},
            {'params': cls_net.module.class_classifier.parameters()},
        ],
        lr=args.lr, momentum=args.momentum, weight_decay=args.l2_decay
    )

    ce_criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    for epoch in range(args.src_train_epochs):
        cls_net.train()
        for batch_idx, batch in enumerate(src_loader):
            data, label = batch[0].cuda(), batch[1].cuda()
            output = cls_net(data)
            loss = ce_criterion(output, label)

            cls_optimizer.zero_grad()
            loss.backward()
            cls_optimizer.step()

            output_label = torch.argmax(output, dim=1)
            acc = torch.sum(output_label == label).float() / label.shape[0]
            if (batch_idx+1) % args.print_freq == 0:
                print(
                    f'[{epoch+1}/{args.src_train_epochs}]'
                    f'[{batch_idx+1}/{len(src_loader)}]\t'
                    f'Loss {loss.item():.2f}\t'
                    f'Acc {acc.item():.1%}'
                )
        acc = cls_evaluate(cls_net, tgt_loader_test)
        print(f'Direct Transfer: {acc:.2%}')
        if epoch == args.src_train_epochs//2:
            cls_optimizer.param_groups[0]['lr'] *= 1e-1
            cls_optimizer.param_groups[1]['lr'] *= 1e-1

    # cls_net.load_state_dict(torch.load(
    #     os.path.join(args.save_path_prefix, 'office.pth')
    # ))

    #-------------------------------------------------------------
    # STEP 2. Task Guided Adv Gen
    #-------------------------------------------------------------
    for param in cls_net.module.parameters():
        param.requires_grad = False
    writer = SummaryWriter(log_dir=args.tensorboard_log_dir)
    style_trans = StyleTrans(args, cls_net)
    style_trans.train(src_gen_loader, tgt_gen_loader, tgt_gen_loader_test, writer)
    writer.close()

    #-------------------------------------------------------------
    # STEP 3.  Fine Tune with Gen
    #-------------------------------------------------------------
    cls_net_out = ClsNet(
        args.model_path_prefix, num_classes,
        args.embed, args.dropout
    )
    cls_net_out = torch.nn.DataParallel(cls_net_out).cuda()
    cls_net_out.load_state_dict(cls_net.state_dict().copy())

    cls_out_optimizer = torch.optim.SGD(
        [
            {'params': cls_net_out.module.base.parameters(), 'lr': args.lr/10},
            {'params': cls_net_out.module.class_classifier.parameters()},
        ],
        lr=args.fine_tune_lr, momentum=args.momentum, weight_decay=args.l2_decay
    )

    for epoch in range(args.fine_tune_epochs):
        for batch_idx, batch_data in enumerate(tgt_gen_loader):
            data = batch_data[0].cuda()
            cls_net.eval()
            cls_net_out.train()

            with torch.no_grad():
                pseudo_logits = style_trans.combine_model(data)
            real_logits = cls_net_out(
                style_trans.normalize(style_trans.de_normalize(data))
            )

            prob = torch.nn.Softmax(dim=1)
            max_value, label = torch.max(prob(pseudo_logits), dim=1)
            label_mask = torch.zeros(label.shape, dtype=torch.uint8).cuda()
            for tem_label in range(num_classes):
                tem_mask = label == tem_label
                if torch.sum(tem_mask) < 5:
                    continue
                value_vec = max_value[tem_mask]
                large_value = torch.topk(value_vec, int(
                    args.percent*value_vec.shape[0]))[0][0]
                large_mask = max_value > large_value
                label_mask = label_mask | (tem_mask & large_mask)
            label[label_mask] = 255

            loss = ce_criterion(real_logits, label)
            cls_out_optimizer.zero_grad()
            loss.backward()
            cls_out_optimizer.step()

            if (batch_idx+1) % args.print_freq == 0:
                print(
                    f'[{epoch+1}/{args.src_train_epochs}]'
                    f'[{batch_idx+1}/{len(src_loader)}]\t'
                    f'Loss {loss.item():.2f}'
                )
        acc = cls_evaluate(cls_net_out, tgt_loader_test)
        print(f'After Fine Tune: {acc:.2%}')

    torch.save(
        cls_net_out.state_dict(), 
        os.path.join(args.save_path_prefix, args.model_name)
    )




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data path
    parser.add_argument('--data_path', type=str,
                        default='/home/users/liangchen.song/data/da/office31')
    parser.add_argument('--tensorboard_log_dir', type=str, default='./logs')
    parser.add_argument('--src_idx', type=str, default='w')
    parser.add_argument('--tgt_idx', type=str, default='a')
    # model path
    parser.add_argument('--model_path_prefix', type=str,
                        default='/home/users/liangchen.song/data/models/')
    parser.add_argument('--save_path_prefix', type=str,
                        default='./data/out')
    parser.add_argument('--model_name', type=str,
                        default=time.strftime('%m%d_%H%M', time.localtime(time.time()))+'.pth')
    # train
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--gen_lr', type=float, default=1e-4)
    parser.add_argument('--fine_tune_lr', type=float, default=1e-3)
    parser.add_argument('--l2_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    # epochs
    parser.add_argument('--src_train_epochs', type=int, default=6)
    parser.add_argument('--warm_up_epoch', type=int, default=5)
    parser.add_argument('--gen_epochs', type=int, default=20)
    parser.add_argument('--fine_tune_epochs', type=int, default=5)
    # log freq
    parser.add_argument('--print_freq', type=int, default=5)
    parser.add_argument('--show_img_freq', type=int, default=5)
    parser.add_argument('--checkpoint_freq', type=int, default=90)
    # loss hyperparameters
    parser.add_argument('--lambda_values', type=str, default='1,1e-10')
    parser.add_argument('--percent', type=float, default=0.5)
    # network structure
    parser.add_argument('--embed', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0)

    main(parser.parse_args())
