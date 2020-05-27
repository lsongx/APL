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

from models.gen_networks import define_G, PoolingDiscriminator
# from models.resnet import ResNet
from utils.tools import AverageMeter
from utils.evaluate import cls_evaluate


class CombineModel(object):
    def __init__(self, gen_net, val_net, de_normalize, normalize):
        self.gen_net = gen_net
        self.val_net = val_net
        self.de_normalize = de_normalize
        self.normalize = normalize

    def __call__(self, x):
        x = self.de_normalize(self.gen_net(x))
        new_x = self.normalize(x)
        out = self.val_net(new_x)
        return out

    def eval(self):
        self.gen_net.eval()
        self.val_net.eval()

###############################################################################
# FULL MODEL
###############################################################################

class StyleTrans(object):
    def __init__(self, opt, cls_net=None):
        super().__init__()
        self.opt = opt

        self.G = define_G(which_model_netG='resnet_6blocks')

        self.mse_criterion = torch.nn.MSELoss()
        self.ce_criterion = torch.nn.CrossEntropyLoss()
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()

        self.lambda_values = list(map(float, opt.lambda_values.split(',')))
        self.cls = cls_net
        self._init_vgg()
        self._init_discrim()
        self._init_optimizer()
        self.cuda()

        self.combine_model = CombineModel(
            self.G, self.cls,
            self.de_normalize, self.normalize,
        )

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

    def _init_discrim(self):
        self.D = PoolingDiscriminator(in_channel=512)

    def _init_optimizer(self):
        self.D_optimizer = torch.optim.SGD(
            [
                {'params': self.D.parameters()}
            ],
            self.opt.gen_lr,
            self.opt.momentum
        )
        self.G_optimizer = torch.optim.SGD(
            self.G.parameters(),
            self.opt.gen_lr,
            self.opt.momentum,
        )

    def cuda(self):
        self.G = nn.DataParallel(self.G.cuda())
        self.vgg = nn.DataParallel(self.vgg.cuda())
        # self.cls = nn.DataParallel(self.cls.cuda())
        self.D = nn.DataParallel(self.D.cuda())
        self.cls.eval()
        self.vgg.eval()

        for param in self.vgg.parameters():
            param.requires_grad = False
        features = list(self.vgg.module.features.children())
        self.content_feat = nn.Sequential(*features[: 16])

        for param in self.cls.parameters():
            param.requires_grad = False
        self.style_feat = functools.partial(self.cls, return_style=True)
        return self

    # --------------------------------------------------
    # loss computing functions
    # --------------------------------------------------
    @staticmethod
    def normalize(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        new_x = []
        for tem_x in x:
            tem_new_x = []
            for c, m, s in zip(tem_x, mean, std):
                tem_new_x.append(c.sub(m).div(s).unsqueeze(0))
            new_x.append(torch.cat(tem_new_x, dim=0).unsqueeze(0))
        new_x = torch.cat(new_x, dim=0)
        return new_x

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
        new_x = self.normalize(x)
        content_feat = self.content_feat(new_x)
        return content_feat

    def compute_content_loss(self, x, ori_x):
        ori_feat = self.compute_content_feat(ori_x)
        rec_feat = self.compute_content_feat(x)
        content_loss = self.mse_criterion(rec_feat, ori_feat)
        return content_loss

    def compute_cls_label(self, x):
        with torch.no_grad():
            new_x = self.normalize(x)
            cls_result = self.cls(new_x)
            cls_label = torch.argmax(cls_result, dim=1)
        return cls_label

    def compute_discrim_loss(self, x, domain_label):
        inter_feature = {}
        def get_output(m, input, output):
            inter_feature['saved'] = output
        handle = self.cls.module.base.layer2.register_forward_hook(get_output)

        forward_result = self.cls(x)
        domain_output = self.D(inter_feature['saved'])
        discrim_loss = self.ce_criterion(
            domain_output,
            torch.ones(domain_output.shape[0]).long().cuda() * domain_label
        )
        return discrim_loss

    # --------------------------------------------------
    # trainer
    # --------------------------------------------------

    def train(self, src_loader, tgt_loader, val_loader, writer):
        tgt_domain_label = 1
        src_domain_label = 0
        num_batches = min(len(src_loader), len(tgt_loader))

        self.G_optimizer.param_groups[0]['lr'] *= 100
        self.D_optimizer.param_groups[0]['lr'] *= 100
        for epoch in range(self.opt.warm_up_epoch):
            for batch_index, batch_data in enumerate(zip(src_loader, tgt_loader)):
                self.G.train()
                src_batch, tgt_batch = batch_data
                src_img, _ = src_batch
                tgt_img, _ = tgt_batch
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

        self.G_optimizer.param_groups[0]['lr'] /= 100
        self.D_optimizer.param_groups[0]['lr'] /= 100
        for epoch in range(self.opt.gen_epochs):

            content_loss_rec = AverageMeter()
            data_time_rec = AverageMeter()
            batch_time_rec = AverageMeter()

            tem_time = time.time()
            for batch_index, batch_data in enumerate(zip(src_loader, tgt_loader)):
                iteration = batch_index+1+epoch*num_batches

                self.G.train()
                src_batch, tgt_batch = batch_data
                src_img, _ = src_batch
                tgt_img, _ = tgt_batch
                src_img_cuda = src_img.cuda()
                tgt_img_cuda = tgt_img.cuda()
                data_time_rec.update(time.time()-tem_time)

                rec_tgt = self.G(tgt_img_cuda) # output [-1,1]
                if (batch_index+1) % self.opt.show_img_freq == 0:
                    rec_results = rec_tgt.detach().clone().cpu()
                # return to [0,1], for VGG takes input [0,1]
                rec_tgt = self.de_normalize(rec_tgt) 
                tgt_img_cuda = self.de_normalize(tgt_img_cuda)

                content_loss = self.compute_content_loss(rec_tgt, tgt_img_cuda)
                loss_style = content_loss * self.lambda_values[0]

                # adv train G
                for param in self.D.parameters():
                    param.requires_grad = False

                adv_tgt_rec_discrim_loss = self.compute_discrim_loss(
                    rec_tgt, src_domain_label
                )
                G_loss = loss_style +\
                         adv_tgt_rec_discrim_loss * self.lambda_values[1]

                self.G_optimizer.zero_grad()
                G_loss.backward()
                self.G_optimizer.step()

                # train D
                for param in self.D.parameters():
                    param.requires_grad = True
                rec_tgt = rec_tgt.detach()

                tgt_rec_discrim_loss = self.compute_discrim_loss(
                    rec_tgt, tgt_domain_label
                )
                tgt_discrim_loss = self.compute_discrim_loss(
                    tgt_img_cuda, tgt_domain_label
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
                writer.add_scalar(
                    'content_loss', content_loss.item(), iteration
                )
                writer.add_scalar(
                    'G_loss', G_loss.item(), iteration
                )
                writer.add_scalar(
                    'D_loss', D_loss.item(), iteration
                )
                batch_time_rec.update(time.time()-tem_time)
                tem_time = time.time()

                if (batch_index+1) % self.opt.print_freq == 0:
                    print(
                        f'Epoch [{epoch+1:d}/{self.opt.gen_epochs:d}]'
                        f'[{batch_index+1:d}/{num_batches:d}]\t'
                        f'Time: {batch_time_rec.avg:.2f}   '
                        f'Data: {data_time_rec.avg:.2f}   '
                        f'Loss: {content_loss_rec.avg:.2f}'
                    )
                if (batch_index+1) % self.opt.show_img_freq == 0:
                    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
                    axes = axes.flat
                    axes[0].imshow(self.to_image(rec_results[0, ...]))
                    label_new = self.compute_cls_label(rec_results)[0]
                    axes[0].set_title(f'rec_label_{label_new}')
                    axes[1].imshow(self.to_image(tgt_img[0, ...]))
                    label_ori = self.compute_cls_label(tgt_img)[0]
                    axes[1].set_title(f'ori_label_{label_ori}')
                    writer.add_figure('Gen', fig, iteration)

                if iteration % self.opt.checkpoint_freq == 0:

                    acc = cls_evaluate(self.combine_model, val_loader)

                    model_name = time.strftime(
                        '%m%d_%H%M_', time.localtime(time.time())
                    ) + str(iteration) + f'_{acc*1000:.0f}.pth'
                    torch.save(
                        self.G.module.state_dict(), 
                        os.path.join(self.opt.save_path_prefix, model_name)
                    )
                    print(f'Model saved as {model_name}')

