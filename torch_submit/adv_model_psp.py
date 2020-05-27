import functools
import os
import time
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
import matplotlib.pyplot as plt

import encoding

from models.gen_networks import define_G, FCDiscriminator
from models.resnet_deeplab import resnet101_ibn_a_deeplab
from models.fcn8s import FCN8s
from utils.tools import AverageMeter, colorize_mask
from utils.evaluate import _evaluate

###############################################################################
# FULL MODEL
###############################################################################

class StyleTrans(object):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.G = define_G()

        self.mse_criterion = torch.nn.MSELoss()
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()

        # loss lambda values
        # [content_loss, style_loss]
        self.lambda_values = list(map(float, opt.lambda_values.split(',')))
        self._init_seg()
        self._init_vgg()
        self._init_discrim()
        self._init_optimizer()
        self.cuda()

    # --------------------------------------------------
    # self initializing functions
    # --------------------------------------------------
    def _init_vgg(self):
        vgg = models.vgg16()
        del vgg.classifier
        state_dict = torch.load(
            os.path.join(self.opt.model_path_prefix, 'vgg16-397923af.pth')
        )
        state_dict_list = list(state_dict.keys())
        for key in state_dict_list:
            if 'classifier' in key:
                del state_dict[key]
        vgg.load_state_dict(state_dict)
        self.vgg = vgg

    def _init_seg(self):
        model = encoding.models.PSP(
            nclass=args.n_classes, backbone='resnet101',
            root=args.model_path_prefix,
        )
        file_name = os.path.join(args.resume, 'psp.pth')
        model.load_state_dict(torch.load(file_name))
        self.seg = model
        self.seg.eval()

    def _init_discrim(self):
        self.D_0 = FCDiscriminator(self.opt.n_classes)
        self.D_1 = FCDiscriminator(self.opt.n_classes)

    def _init_optimizer(self):
        optimizer_name = self.opt.optimizer.lower()
        self.D_0_optimizer = torch.optim.SGD(
            self.D.parameters(),
            self.opt.D_learning_rate,
            self.opt.momentum
        )
        self.D_1_optimizer = torch.optim.SGD(
            self.D.parameters(),
            self.opt.D_learning_rate,
            self.opt.momentum
        )
        if self.opt.optimizer == 'sgd':
            self.G_optimizer = torch.optim.SGD(
                self.G.parameters(),
                self.opt.learning_rate,
                self.opt.momentum,
            )
        elif self.opt.optimizer == 'adam':
            self.G_optimizer = torch.optim.Adam(
                self.G.parameters(),
                self.opt.learning_rate,
            )
        else:
            raise RuntimeError(f'Optimizer: {self.opt.optimizer}')

    def cuda(self):
        self.G = nn.DataParallel(self.G.cuda())
        self.vgg = nn.DataParallel(self.vgg.cuda())
        self.seg = nn.DataParallel(self.seg.cuda())
        self.D_0 = nn.DataParallel(self.D_0.cuda())
        self.D_1 = nn.DataParallel(self.D_1.cuda())
        self.seg.eval()
        self.vgg.eval()

        for param in self.vgg.parameters():
            param.requires_grad = False
        features = list(self.vgg.module.features.children())
        self.content_feat = nn.Sequential(*features[: 16])

        for param in self.seg.parameters():
            param.requires_grad = False
        self.style_feat = functools.partial(self.seg, return_style=True)
        return self
    # --------------------------------------------------
    # loss computing functions
    # --------------------------------------------------
    @staticmethod
    def normalize(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        for tem_x in x:
            for c, m, s in zip(tem_x, mean, std):
                c = c.sub(m).div(s)
        return x

    @staticmethod
    def de_normalize(x, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        for tem_x in x:
            for c, m, s in zip(tem_x, mean, std):
                c = c.mul(s).add(s)
        return x

    @staticmethod
    def to_image(x):
        if x.ndimension() > 3:
            raise RuntimeError('Only support batch size 1')
        mean = [0.5]*3
        std = [0.5]*3
        for c, m, s in zip(x, mean, std):
            c.mul_(s).add_(m)
        x_new = torchvision.transforms.functional.to_pil_image(x)
        return x_new

    def compute_content_feat(self, x):
        new_x = self.normalize(x)
        content_feat = self.content_feat(new_x)
        return content_feat

    def compute_content_loss(self, x, ori_x):
        ori_feat = self.compute_content_feat(ori_x)
        rec_feat = self.compute_content_feat(x)
        content_loss = self.mse_criterion(rec_feat, ori_feat)
        return content_loss

    def compute_seg_map(self, x):
        with torch.no_grad():
            new_x = self.normalize(x)
            seg_result = self.seg(new_x)[1]
            seg_map = torch.argmax(seg_result, dim=1)
        return seg_map

    def compute_discrim_loss(self, x, domain_label):
        new_x = self.normalize(x)
        seg_out_0, seg_out_1 = self.seg(new_x)
        domain_output_0 = self.D_0(seg_out_0)
        domain_output_1 = self.D_1(seg_out_1)
        discrim_loss_0 = self.bce_criterion(
            domain_output_0,
            torch.FloatTensor(domain_output_0.shape).fill_(domain_label).cuda(),
        )
        discrim_loss_1 = self.bce_criterion(
            domain_output_1,
            torch.FloatTensor(domain_output_1.shape).fill_(domain_label).cuda(),
        )        
        return discrim_loss_0+discrim_loss_1

    # --------------------------------------------------
    # trainer
    # --------------------------------------------------

    def train(self, src_loader, tgt_loader, writer):
        tgt_domain_label = 1
        src_domain_label = 0
        num_batches = min(len(src_loader), len(tgt_loader))

        for epoch in range(self.opt.warm_up_epoch):
            for batch_index, batch_data in enumerate(zip(src_loader, tgt_loader)):
                self.G.train()
                src_batch, tgt_batch = batch_data
                src_img, _, src_name = src_batch
                tgt_img, _, tgt_name = tgt_batch
                src_img_cuda = src_img.cuda()
                tgt_img_cuda = tgt_img.cuda()

                rec_tgt = self.G(tgt_img_cuda)  # output [-1,1]
                rec_loss = self.mse_criterion(rec_tgt, tgt_img_cuda)
                self.G_optimizer.zero_grad()
                rec_loss.backward()
                self.G_optimizer.step()

                tgt_img_cuda = self.de_normalize(tgt_img_cuda).detach()
                tgt_D_loss = self.compute_discrim_loss(
                    tgt_img_cuda, tgt_domain_label
                )
                rec_D_loss = self.compute_discrim_loss(
                    src_img_cuda, src_domain_label
                )
                D_loss = tgt_D_loss + rec_D_loss
                self.D_0_optimizer.zero_grad()
                self.D_1_optimizer.zero_grad()
                D_loss.backward()
                self.D_0_optimizer.step()
                self.D_1_optimizer.step()

                if (batch_index+1) % self.opt.print_freq == 0:
                    print(
                        f'Warm Up Epoch [{epoch+1:d}/{self.opt.warm_up_epoch:d}]'
                        f'[{batch_index+1:d}/{num_batches:d}]\t'
                        f'G Loss: {rec_loss.item():.2f}   '
                        f'D Loss: {D_loss.item():.2f}'
                    )

        for epoch in range(self.opt.num_epoch):

            content_loss_rec = AverageMeter()
            style_loss1_rec = AverageMeter()
            style_loss2_rec = AverageMeter()
            data_time_rec = AverageMeter()
            batch_time_rec = AverageMeter()

            tem_time = time.time()
            for batch_index, batch_data in enumerate(zip(src_loader, tgt_loader)):
                iteration = batch_index+1+epoch*num_batches

                self.G.train()
                src_batch, tgt_batch = batch_data
                src_img, _, src_name = src_batch
                tgt_img, tgt_label, tgt_name = tgt_batch
                src_img_cuda = src_img.cuda()
                tgt_img_cuda = tgt_img.cuda()
                data_time_rec.update(time.time()-tem_time)

                rec_tgt = self.G(tgt_img_cuda) # output [-1,1]
                if (batch_index+1) % self.opt.show_img_freq == 0:
                    rec_results = rec_tgt.detach().clone().cpu()
                # return to [0,1], for VGG takes input [0,1]
                rec_tgt = self.de_normalize(rec_tgt) 

                content_loss = self.compute_content_loss(rec_tgt, tgt_img_cuda)
                style_loss1, style_loss2 =\
                    self.compute_style_loss(rec_tgt, src_img_cuda)
                loss_style = content_loss * self.lambda_values[0] +\
                             style_loss1 * self.lambda_values[1] +\
                             style_loss2 * self.lambda_values[2]

                # adv train G
                for param in self.D_0.parameters():
                    param.requires_grad = False
                for param in self.D_1.parameters():
                    param.requires_grad = False

                adv_tgt_rec_discrim_loss = self.compute_discrim_loss(
                    rec_tgt, src_domain_label
                )
                G_loss = loss_style +\
                         adv_tgt_rec_discrim_loss * self.lambda_values[3]

                self.G_optimizer.zero_grad()
                G_loss.backward()
                self.G_optimizer.step()

                # train D
                for param in self.D_0.parameters():
                    param.requires_grad = True
                for param in self.D_1.parameters():
                    param.requires_grad = True
                rec_tgt = rec_tgt.detach()

                tgt_rec_discrim_loss = self.compute_discrim_loss(
                    rec_tgt, tgt_domain_label
                )
                # tgt_rec_discrim_loss = 0
                tgt_img_cuda = self.de_normalize(tgt_img_cuda)
                tgt_discrim_loss = self.compute_discrim_loss(
                    tgt_img_cuda, tgt_domain_label
                )
                src_discrim_loss = self.compute_discrim_loss(
                    src_img_cuda, src_domain_label
                )
                D_loss = 0.5 * (tgt_rec_discrim_loss + tgt_discrim_loss) +\
                         src_discrim_loss

                self.D_0_optimizer.zero_grad()
                self.D_1_optimizer.zero_grad()
                D_loss.backward()
                self.D_0_optimizer.step()
                self.D_1_optimizer.step()

                content_loss_rec.update(content_loss.item())
                style_loss1_rec.update(style_loss1.item())
                style_loss2_rec.update(style_loss2.item())
                writer.add_scalar(
                    'AA_content_loss', content_loss.item(), iteration
                )
                writer.add_scalar(
                    'AA_style_loss_1', style_loss1.item(), iteration
                )
                writer.add_scalar(
                    'AA_style_loss_2', style_loss2.item(), iteration
                )
                writer.add_scalar(
                    'AA_G_loss', G_loss.item(), iteration
                )
                writer.add_scalar(
                    'AA_D_loss', D_loss.item(), iteration
                )
                batch_time_rec.update(time.time()-tem_time)
                tem_time = time.time()

                if (batch_index+1) % self.opt.print_freq == 0:
                    print(
                        f'Epoch [{epoch+1:d}/{self.opt.num_epoch:d}]'
                        f'[{batch_index+1:d}/{num_batches:d}]\t'
                        f'Time: {batch_time_rec.avg:.2f}   '
                        f'Data: {data_time_rec.avg:.2f}   '
                        f'Loss: {content_loss_rec.avg:.2f}   '
                        f'Style1: {style_loss1_rec.avg:.2f}   '
                        f'Style2: {style_loss2_rec.avg:.2f}'
                    )
                if (batch_index+1) % self.opt.show_img_freq == 0:
                    fig, axes = plt.subplots(5, 1, figsize=(12, 30))
                    axes = axes.flat
                    axes[0].imshow(self.to_image(rec_results[0, ...]))
                    axes[0].set_title(f'rec')
                    axes[1].imshow(self.to_image(tgt_img[0, ...]))
                    axes[1].set_title(tgt_name[0])

                    rec_seg = self.compute_seg_map(rec_results).cpu().numpy()
                    tgt_img_cuda = self.de_normalize(tgt_img_cuda)
                    ori_seg = self.compute_seg_map(tgt_img_cuda).cpu().numpy()
                    tgt_label = tgt_label.numpy()
                    rec_miu = _evaluate(rec_seg, tgt_label, self.opt.n_classes)[3]
                    ori_miu = _evaluate(ori_seg, tgt_label, self.opt.n_classes)[3]

                    axes[2].imshow(colorize_mask(rec_seg[0, ...]))
                    axes[2].set_title(f'rec_label_{rec_miu*100:.2f}')

                    axes[3].imshow(colorize_mask(ori_seg[0, ...]))
                    axes[3].set_title(f'ori_label_{ori_miu*100:.2f}')

                    gt_label = tgt_label[0, ...]
                    axes[4].imshow(colorize_mask(gt_label))
                    axes[4].set_title(f'gt_label')

                    writer.add_figure('A_rec', fig, iteration)
                    miu_inc = 0 if ori_miu > rec_miu else rec_miu-ori_miu 
                    writer.add_scalar('AA_val_acc', miu_inc, iteration)
                if (batch_index+1) % self.opt.checkpoint_freq == 0:
                    model_name = time.strftime(
                        '%m%d_%H%M_', time.localtime(time.time())
                    ) + str(iteration) + '.pth'
                    torch.save(
                        self.G.module.state_dict(), 
                        os.path.join(self.opt.save_path_prefix, model_name)
                    )


