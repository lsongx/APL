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

from models.gen_networks import define_G, FCDiscriminator, PoolingDiscriminator
from models.resnet_deeplab import resnet101_ibn_a_deeplab
from models.deeplabv3 import get_deeplabV3
from utils.tools import AverageMeter, colorize_mask
from utils.evaluate import _evaluate, test_miou

###############################################################################
# FULL MODEL
###############################################################################

class StyleTrans(object):
    def __init__(self, opt, info_json='./dataset/info.json'):
        super().__init__()
        self.opt = opt
        self.info_json = info_json

        self.G = define_G()

        self.mse_criterion = torch.nn.MSELoss()
        self.ce_criterion = torch.nn.CrossEntropyLoss()
        self.sl1_criterion = torch.nn.SmoothL1Loss()
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
        model = get_deeplabV3(self.opt.n_classes, pretrained=False)
        file_name = os.path.join(self.opt.resume, self.opt.deeplabv3_name)
        model.load_state_dict(torch.load(file_name))
        self.seg = model
        self.seg.eval()

    def _init_discrim(self):
        self.D = FCDiscriminator(self.opt.n_classes)
        self.D_dsn = FCDiscriminator(self.opt.n_classes)
        self.D_pool = PoolingDiscriminator(in_channel=512)

    def _init_optimizer(self):
        optimizer_name = self.opt.optimizer.lower()
        self.D_optimizer = torch.optim.SGD(
            [
                {'params': self.D_pool.parameters()},
                {'params': self.D_dsn.parameters()},
                {'params': self.D.parameters()}
            ],
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
        self.D = nn.DataParallel(self.D.cuda())
        self.D_dsn = nn.DataParallel(self.D.cuda())
        self.D_pool = nn.DataParallel(self.D_pool.cuda())
        self.seg.eval()
        self.vgg.eval()

        for param in self.vgg.parameters():
            param.requires_grad = False
        features = list(self.vgg.module.features.children())
        self.content_feat = nn.Sequential(*features[: 16])

        for param in self.seg.parameters():
            param.requires_grad = False
        return self
    # --------------------------------------------------
    # loss computing functions
    # --------------------------------------------------
    def fcn_normalize(self, x):
        mean = [103.939, 116.779, 123.68]
        flip_x = torch.cat(
            [x[:,2-i,:,:].unsqueeze(1) for i in range(3)],
            dim = 1,
        )
        new_x = []
        for tem_x in flip_x:
            tem_new_x = []
            for c, m in zip(tem_x, mean):
                tem_new_x.append(c.mul(255.0).sub(m).unsqueeze(0))
            new_x.append(torch.cat(tem_new_x, dim=0).unsqueeze(0))
        new_x = torch.cat(new_x, dim=0)
        return new_x

    def normalize(self, x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        # new_x = []
        # for tem_x in flip_x:
        #     tem_new_x = []
        #     for c, m in zip(tem_x, mean):
        #         tem_new_x.append(c.mul(255.0).sub(m).unsqueeze(0))
        #     new_x.append(torch.cat(tem_new_x, dim=0).unsqueeze(0))
        # new_x = torch.cat(new_x, dim=0)
        # return new_x
        """
        Do NOT normalize.
        The perceptual loss is too accurate.
        """
        return x

    @staticmethod
    def de_normalize(x, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        new_x = []
        for tem_x in x:
            tem_new_x = []
            for c, m, s in zip(tem_x, mean, std):
                tem_new_x.append(c.mul(s).add(s).unsqueeze(0))
            new_x.append(torch.cat(tem_new_x, dim=0).unsqueeze(0))
        new_x = torch.cat(new_x, dim=0)
        return new_x

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
        # x: [0,1]
        new_x = self.normalize(x)
        content_feat = self.content_feat(new_x)
        return content_feat

    def compute_content_loss(self, x, ori_x):
        # x: [0,1]
        # ori_x: [0,1]
        ori_feat = self.compute_content_feat(ori_x)
        rec_feat = self.compute_content_feat(x)
        content_loss = self.mse_criterion(rec_feat, ori_feat)
        # content_loss = self.sl1_criterion(rec_feat, ori_feat)
        return content_loss

    @staticmethod
    def compute_gram_mat(feat1, feat2):
        b, feat1_c, _, _ = feat1.shape
        b, feat2_c, _, _ = feat2.shape
        feat1 = feat1.view(b, feat1_c, -1).clone()
        feat2 = feat2.view(b, feat2_c, -1).clone()
        gram_mat = torch.bmm(feat1, feat2.transpose(1,2))
        return gram_mat

    def compute_seg_map(self, x):
        # x: [0,1]
        with torch.no_grad():
            new_x = self.fcn_normalize(x)
            seg_result = self.seg(new_x)[0]
            seg_map = torch.argmax(seg_result, dim=1)
        return seg_map

    def compute_discrim_loss(self, x, domain_label):
        # input: [0,1]
        inter_feature = {}
        def get_output(m, input, output):
            inter_feature['saved'] = output
        # handle = self.seg.module.features5.register_forward_hook(get_output)
        handle = self.seg.module.layer2.register_forward_hook(get_output)

        new_x = self.fcn_normalize(x)
        seg_out, seg_out_dsn = self.seg(new_x)
        domain_output = self.D(seg_out)
        discrim_loss = self.bce_criterion(
            domain_output,
            torch.FloatTensor(domain_output.shape).fill_(domain_label).cuda(),
        )
        domain_output_dsn = self.D_dsn(seg_out_dsn)
        discrim_loss_dsn = self.bce_criterion(
            domain_output_dsn,
            torch.FloatTensor(domain_output_dsn.shape).fill_(domain_label).cuda(),
        )

        pool_domain_output = self.D_pool(inter_feature['saved'])
        pool_discrim_loss = self.ce_criterion(
            pool_domain_output,
            torch.ones(pool_domain_output.shape[0]).long().cuda() * domain_label
        )
        return discrim_loss + pool_discrim_loss + discrim_loss_dsn*0.4
        # return discrim_loss
        # return pool_discrim_loss

    # --------------------------------------------------
    # trainer
    # --------------------------------------------------

    def train(self, src_loader, tgt_loader, val_loader, writer):
        highest_miu = 0
        tgt_domain_label = 1
        src_domain_label = 0
        num_batches = min(len(src_loader), len(tgt_loader))

        # self.G_optimizer.param_groups[0]['lr'] *= 10
        # self.D_optimizer.param_groups[0]['lr'] *= 10
        # self.D_optimizer.param_groups[1]['lr'] *= 10
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
                self.D_optimizer.zero_grad()
                D_loss.backward()
                self.D_optimizer.step()

                if (batch_index+1) % self.opt.print_freq == 0:
                    print(
                        f'Warm Up Epoch [{epoch+1:d}/{self.opt.warm_up_epoch:d}]'
                        f'[{batch_index+1:d}/{num_batches:d}]\t'
                        f'G Loss: {rec_loss.item():.2f}   '
                        f'D Loss: {D_loss.item():.2f}'
                    )

        # self.G_optimizer.param_groups[0]['lr'] /= 10
        # self.D_optimizer.param_groups[0]['lr'] /= 10
        # self.D_optimizer.param_groups[1]['lr'] /= 10
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
                rec_tgt_de_norm = self.de_normalize(rec_tgt) 

                '''
                --------------------------------------------------
                NOTICE: 
                DO NOT ADD DE-NORM HERE 
                WITHOUT NORM WE CAN ADD MORE NOISE TO CONTENT LOSS 
                --------------------------------------------------
                '''
                content_loss = self.compute_content_loss(rec_tgt_de_norm, tgt_img_cuda)

                # style_loss1, style_loss2 =\
                #     self.compute_style_loss(rec_tgt, src_img_cuda)
                style_loss1 = torch.zeros(1).cuda()
                style_loss2 = torch.zeros(1).cuda()
                loss_style = content_loss * self.lambda_values[0] +\
                             style_loss1 * self.lambda_values[1] +\
                             style_loss2 * self.lambda_values[2]

                # adv train G
                for param in self.D.parameters():
                    param.requires_grad = False
                

                adv_tgt_rec_discrim_loss = self.compute_discrim_loss(
                    rec_tgt_de_norm, src_domain_label
                )
                G_loss = loss_style +\
                         adv_tgt_rec_discrim_loss * self.lambda_values[3]

                self.G_optimizer.zero_grad()
                G_loss.backward()
                self.G_optimizer.step()

                # train D
                for param in self.D.parameters():
                    param.requires_grad = True

                # add de norm here, since D do not need noise for training
                tgt_img_cuda_de_norm = self.de_normalize(tgt_img_cuda)

                rec_tgt_de_norm = rec_tgt_de_norm.detach()

                tgt_rec_discrim_loss = self.compute_discrim_loss(
                    rec_tgt_de_norm, tgt_domain_label
                )
                # tgt_rec_discrim_loss = 0
                tgt_discrim_loss = self.compute_discrim_loss(
                    tgt_img_cuda_de_norm, tgt_domain_label
                )
                src_discrim_loss = self.compute_discrim_loss(
                    src_img_cuda, src_domain_label
                )
                D_loss = 0.5 * (tgt_rec_discrim_loss + tgt_discrim_loss) +\
                         src_discrim_loss

                self.D_optimizer.zero_grad()
                D_loss.backward()
                self.D_optimizer.step()

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
                    fig, axes = plt.subplots(5, 1, figsize=(6, 20), dpi=120)
                    axes = axes.flat
                    axes[0].imshow(self.to_image(rec_results[0, ...]))
                    axes[0].set_title(f'rec')
                    axes[1].imshow(self.to_image(tgt_img[0, ...]))
                    axes[1].set_title(tgt_name[0])

                    rec_seg = self.compute_seg_map(rec_results).cpu().numpy() # already normed in to_image method
                    # tgt_img_cuda = self.de_normalize(tgt_img_cuda)
                    ori_seg = self.compute_seg_map(tgt_img_cuda_de_norm).cpu().numpy()

                    axes[2].imshow(colorize_mask(rec_seg[0, ...]))
                    axes[2].set_title(f'rec_label')

                    axes[3].imshow(colorize_mask(ori_seg[0, ...]))
                    axes[3].set_title(f'ori_label')

                    tgt_label = tgt_label.numpy()
                    gt_label = tgt_label[0, ...]
                    axes[4].imshow(colorize_mask(gt_label))
                    axes[4].set_title(f'gt_label')

                    writer.add_figure('A_rec', fig, iteration)
                if iteration % self.opt.checkpoint_freq == 0:

                    combine_model = _CombineModel(
                        self.G, self.seg, 
                        self.de_normalize, self.fcn_normalize,
                    )
                    mean_iu = test_miou(
                        combine_model, val_loader,
                        combine_model.upsample, self.info_json,
                        print_results=False
                    )

                    if mean_iu > highest_miu:
                        torch.save(
                            self.G.module.state_dict(), 
                            os.path.join(self.opt.save_path_prefix, 'gen.pth')
                        )
                        highest_miu = mean_iu
                        print('>'*50+f'saving highest {mean_iu:.2%}')



class _CombineModel(object):
    def __init__(self, gen_net, val_net, de_normalize, normalize):
        self.gen_net = gen_net
        self.val_net = val_net
        self.de_normalize = de_normalize
        self.normalize = normalize
        self.upsample = torch.nn.Upsample(
            size=(1024, 2048),
            mode='bilinear', align_corners=True
        )

    def __call__(self, x):
        x = self.de_normalize(self.gen_net(x))
        new_x = self.normalize(x)
        out = self.val_net(new_x)
        return out

    def eval(self):
        self.gen_net.eval()
        self.val_net.eval()
