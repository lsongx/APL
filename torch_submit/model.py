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

from models.gen_networks import define_G
from models.resnet_deeplab import resnet101_ibn_a_deeplab
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
        self.sl1_criterion = torch.nn.SmoothL1Loss()

        # loss lambda values
        # [content_loss, style_loss]
        self.lambda_values = list(map(float, opt.lambda_values.split(',')))
        self._init_seg()
        self._init_vgg()
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
        deeplab = resnet101_ibn_a_deeplab()
        deeplab.load_state_dict(torch.load(self.opt.deeplab_resume))
        self.seg = deeplab


    def _init_optimizer(self):
        optimizer_name = self.opt.optimizer.lower()
        if self.opt.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.G.parameters(),
                self.opt.learning_rate,
                self.opt.momentum,
            )
        elif self.opt.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(
                self.G.parameters(),
                self.opt.learning_rate,
            )
        else:
            raise RuntimeError(f'Optimizer: {self.opt.optimizer}')

    def cuda(self):
        self.G = nn.DataParallel(self.G.cuda())
        self.vgg = nn.DataParallel(self.vgg.cuda())
        self.seg = nn.DataParallel(self.seg.cuda())

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

    @staticmethod
    def compute_gram_mat(feat1, feat2):
        b, feat1_c, _, _ = feat1.shape
        b, feat2_c, _, _ = feat2.shape
        feat1 = feat1.view(b, feat1_c, -1).clone()
        feat2 = feat2.view(b, feat2_c, -1).clone()
        gram_mat = torch.bmm(feat1, feat2.transpose(1,2))
        return gram_mat

    def compute_style_feat(self, x):
        new_x = self.normalize(x)
        style_feat_list = self.style_feat(new_x)
        return style_feat_list

    def compute_style_loss(self, x, ref):
        s_feat_list = self.compute_style_feat(x)
        ref_s_feat_list = self.compute_style_feat(ref)
        style_loss_list = []
        for s_feat, ref_s_feat in zip(s_feat_list, ref_s_feat_list):
            gram_mat = self.compute_gram_mat(s_feat, s_feat)
            ref_gram_mat = self.compute_gram_mat(ref_s_feat, ref_s_feat)
            # style_loss_list.append(self.mse_criterion(gram_mat, ref_gram_mat))
            style_loss_list.append(self.sl1_criterion(gram_mat, ref_gram_mat))
        return style_loss_list

    def compute_seg_map(self, x):
        with torch.no_grad():
            new_x = self.normalize(x)
            seg_result = self.seg(new_x)
            seg_map = torch.argmax(seg_result, dim=1)
        return seg_map

    # --------------------------------------------------
    # trainer
    # --------------------------------------------------

    def train(self, src_loader, tgt_loader, writer):
        num_batches = min(len(src_loader), len(tgt_loader))
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
                rec_tgt = self.de_normalize(rec_tgt) # return to [0,1]

                content_loss = self.compute_content_loss(rec_tgt, tgt_img_cuda)
                style_loss1, style_loss2 =\
                    self.compute_style_loss(rec_tgt, src_img_cuda)
                loss = content_loss * self.lambda_values[0] +\
                       style_loss1 * self.lambda_values[1] +\
                       style_loss2 * self.lambda_values[2]
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

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
                    rec_miu = _evaluate(rec_seg, tgt_label, 19)[3]
                    ori_miu = _evaluate(ori_seg, tgt_label, 19)[3]

                    axes[2].imshow(colorize_mask(rec_seg[0, ...]))
                    axes[2].set_title(f'rec_label_{rec_miu*100:.2f}')

                    axes[3].imshow(colorize_mask(ori_seg[0, ...]))
                    axes[3].set_title(f'ori_label_{ori_miu*100:.2f}')

                    gt_label = tgt_label[0, ...]
                    axes[4].imshow(colorize_mask(gt_label))
                    axes[4].set_title(f'gt_label')

                    writer.add_figure('A_rec', fig, iteration)
                if (batch_index+1) % self.opt.checkpoint_freq == 0:
                    model_name = time.strftime(
                        '%m%d_%H%M_', time.localtime(time.time())
                    ) + str(iteration) + '.pth'
                    torch.save(
                        self.G.module.state_dict(), 
                        os.path.join(self.opt.save_path_prefix, model_name)
                    )


