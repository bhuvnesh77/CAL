"""
WS-DAN models
Hu et al.,
"See Better Before Looking Closer: Weakly Supervised Data Augmentation Network for Fine-Grained Visual Classification",
arXiv:1901.09891
"""
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from torch.autograd import Variable
from models.ConvNext import convnext_small
from torchvision import transforms
import torchattacks
from torchvision.utils import save_image
import models.resnet as resnet
from models.inception import inception_v3, BasicConv2d
import random

__all__ = ['WSDAN_CAL']
EPSILON = 1e-6


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

# Bilinear Attention Pooling
class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions, fake_att):
        
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        feature_matrix_raw = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix_raw, dim=-1)

        # if self.training:
        #     #fake_att = adversarial(features, )
        #     fake_att = torch.zeros_like(attentions).uniform_(0, 2)
        # else:
        #     fake_att = torch.ones_like(attentions)
        counterfactual_feature = (torch.einsum('imjk,injk->imn', (fake_att, features)) / float(H * W)).view(B, -1)

        counterfactual_feature = torch.sign(counterfactual_feature) * torch.sqrt(torch.abs(counterfactual_feature) + EPSILON)

        counterfactual_feature = F.normalize(counterfactual_feature, dim=-1)

            
        return feature_matrix, counterfactual_feature

def batch_augment(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
    batches, _, imgH, imgW = images.size()

    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            crop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) >= theta_c
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

            crop_images.append(
                F.upsample_bilinear(images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                                    size=(imgH, imgW)))
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    elif mode == 'drop':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()

            drop_masks.append(F.upsample_bilinear(atten_map, size=(imgH, imgW)) < theta_d)
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()
        return drop_images

    else:
        raise ValueError('Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)

class WSDAN_CAL(nn.Module):
    def __init__(self, num_classes, M=32, net='inception_mixed_6e', pretrained=False):
        super(WSDAN_CAL, self).__init__()
        self.num_classes = num_classes
        self.M = M
        self.net = net

    
        # Network Initialization
        if 'inception' in net:
            if net == 'inception_mixed_6e':
                self.features = inception_v3(pretrained=pretrained).get_features_mixed_6e()
                self.num_features = 768
            elif net == 'inception_mixed_7c':
                self.features = inception_v3(pretrained=pretrained).get_features_mixed_7c()
                self.num_features = 2048
            else:
                raise ValueError('Unsupported net: %s' % net)
        elif 'resnet' in net:
            self.features = getattr(resnet, net)(pretrained=pretrained).get_features()
            self.num_features = 512 * self.features[-1][-1].expansion
        elif 'att' in net:
            print('==> Using MANet with resnet101 backbone')
            self.features = MANet()
            self.num_features = 2048
        elif 'convnext' in net:
            self.features = convnext_small(pretrained=True).get_features()
            self.num_features = 768
        else:
            raise ValueError('Unsupported net: %s' % net)

        # Attention Maps
        self.attentions = BasicConv2d(self.num_features, self.M, kernel_size=1)

        # Bilinear Attention Pooling
        self.bap = BAP(pool='GAP')

        # Classification Layer
        self.fc = nn.Linear(self.M * self.num_features, self.num_classes, bias=False)

        logging.info('WSDAN: using {} as feature extractor, num_classes: {}, num_attentions: {}'.format(net, self.num_classes, self.M))
        # self.load_chkpt() #loading parameters from path

    def load_chkpt(self, path = '/home/bhuvnesh.kumar/Downloads/Projects/CAL/fgvc/FGVC/bird/Random_Reverse/wsdan-resnet101-cal/model_bestacc.pth'):
        chpt = torch.load(path)
        state_dict = chpt['state_dict']
        self.load_state_dict(state_dict)

    def visualize(self, x, x_hat= None):
        batch_size = x.size(0)

        # Feature Maps, Attention Maps and Feature Matrix
        feature_maps = self.features(x)
        # adv_fm = self.features(x_hat)
        if self.net != 'inception_mixed_7c':
            attention_maps = self.attentions(feature_maps)
            # adv_am = self.attentions(adv_fm)
        else:
            attention_maps = feature_maps[:, :self.M, ...]
        # fake_att = adv_am
        # torch.ones_like(attention_maps)
        fake_att = torch.flip(feature_maps, [0,1])
        feature_matrix, _ = self.bap(feature_maps, attention_maps, fake_att)
        p = self.fc(feature_matrix * 100.)

        return p, attention_maps, fake_att

    def forward(self, x, x_hat=None, adv=True):

        batch_size = x.size(0)


        
        #pixattack = torchattacks.Pixle(p, x_dimensions=(2,10), y_dimensions=(2,10), restarts=10, max_iterations=50)
        #adv_images = pixattack(x, y)  
        
        # Feature Maps, Attention Maps and Feature Matrix

        if x_hat!=None:
            adv_feature_maps = self.features(x_hat)

        feature_maps = self.features(x)
        
        # feature_maps_pertubed = avg.repeat(1,feature_maps.shape[1],1,1) + feature_maps[torch.randperm(feature_maps.size()[0])] # uniform + shuffle
        # feature_maps_pertubed = torch.zeros_like(feature_maps).uniform_(0, 2) + torch.flip(feature_maps, [0,1])  # random + reverse
        
        if self.net != 'inception_mixed_7c':
            if x_hat!=None:
                attention_maps_pertubed = self.attentions(adv_feature_maps)
            
            
            attention_maps = self.attentions(feature_maps)
            
            
        else:
            if x_hat!=None:
                attention_maps_pertubed = self.attentions(adv_feature_maps)
            
            attention_maps = feature_maps[:, :self.M, ...]
        
        if self.training:

            if x_hat!=None:
                fake_att = attention_maps_pertubed
            else:
                fake_att = torch.zeros_like(attention_maps).uniform_(0, 2)
                # if perturb == masked:
                # avg = torch.mean(attention_maps.type(torch.float32), dim=1, keepdim=True)
                # mean_tensor = avg.repeat(1,attention_maps.shape[1],1,1)
                # mask_tensor = attention_maps > mean_tensor
                # new_tensor = (~mask_tensor).int()
                # fake_att = mean_tensor * new_tensor

                #fake_att = torch.zeros_like(attention_maps).uniform_(0, 2) + torch.flip(attention_maps, [0,1]) + attention_maps[torch.randperm(attention_maps.size()[0])] # Random + Reverse + Shuffle
                # fake_att = torch.zeros_like(attention_maps).uniform_(0, 2) + torch.flip(attention_maps, [0,1])  # Random + Reverse  
            
        else:
            fake_att = torch.ones_like(attention_maps)
        
        feature_matrix, feature_matrix_hat = self.bap(feature_maps, attention_maps, fake_att)





        # Classification
        p = self.fc(feature_matrix * 100.)

        # Generate Attention Map
        if self.training:
            # Randomly choose one of attention maps Ak
            attention_map = []
            for i in range(batch_size):
                attention_weights = torch.sqrt(attention_maps[i].sum(dim=(1, 2)).detach() + EPSILON)
                attention_weights = F.normalize(attention_weights, p=1, dim=0)
                k_index = np.random.choice(self.M, 2, p=attention_weights.cpu().numpy())
                attention_map.append(attention_maps[i, k_index, ...])
            attention_map = torch.stack(attention_map)  # (B, 2, H, W) - one for cropping, the other for dropping
        else:
            attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)
        
        if adv:
            return p
        else:
            return p, p - self.fc(feature_matrix_hat * 100.), feature_matrix, attention_map

    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and model_dict[k].size() == v.size()}

        if len(pretrained_dict) == len(state_dict):
            print('%s: All params loaded' % type(self).__name__)
        else:
            print('%s: Some params were not loaded:' % type(self).__name__)
            not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
            print(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))

        model_dict.update(pretrained_dict)
        super(WSDAN_CAL, self).load_state_dict(model_dict)

def distort(x, num_pixels=200, value=1.0):
    for batch_idx in range(x.size(0)):
        for _ in range(num_pixels):

            x[batch_idx,:, int(random.random()*400), int(random.random()*400)] = value
        return x
    

