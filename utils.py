import os
import time
import shutil
import math

import torch
import torch.nn as nn
import numpy as np
from torch.optim import SGD
from tensorboardX import SummaryWriter
from Adam import Adam
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torchvision import transforms
from torchvision.transforms import InterpolationMode
import random
import math
from pytorch_msssim import ssim
import lpips

import torchvision.models #as models


def show_feature_map(feature_map,layer,name='rgb',rgb=False):
    feature_map = feature_map.squeeze(0)
    #if rgb: feature_map = feature_map.permute(1,2,0)*0.5+0.5
    feature_map = feature_map.cpu().numpy()
    feature_map_num = feature_map.shape[0]
    row_num = math.ceil(np.sqrt(feature_map_num))
    if rgb:
        #plt.figure()
        #plt.imshow(feature_map)
        #plt.axis('off')
        feature_map = cv2.cvtColor(feature_map,cv2.COLOR_BGR2RGB)
        cv2.imwrite('data/'+layer+'/'+name+".png",feature_map*255)
        #plt.show()
    else:
        plt.figure()
        for index in range(1, feature_map_num+1):
            t = (feature_map[index-1]*255).astype(np.uint8)
            t = cv2.applyColorMap(t, cv2.COLORMAP_TWILIGHT)
            plt.subplot(row_num, row_num, index)
            plt.imshow(t, cmap='gray')
            plt.axis('off')
            #ensure_path('data/'+layer)
            cv2.imwrite('data/'+layer+'/'+str(name)+'_'+str(index)+".png",t)
        #plt.show()
        plt.savefig('data/'+layer+'/'+str(name)+".png")

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img)))

class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.3f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.3f}m'.format(t / 60)
    else:
        return '{:.3f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.3f}M'.format(tot / 1e6)
        else:
            return '{:.3f}K'.format(tot / 1e3)
    else:
        return tot

def make_optimizer(model, optimizer_spec, load_sd=False):
    param_list = [
        {
            'params': [param for name, param in model.named_parameters() if name.split('.')[-1] in ['s', 'x_s', 'y_s', 'xy_s']],
            'lr': optimizer_spec['s_lr']
        },
        {
            'params': [param for name, param in model.named_parameters() if name.split('.')[-1] not in ['s', 'x_s', 'y_s', 'xy_s']]
        }]

    print(param_list[0])
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    
    return optimizer

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    #ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    ret = torch.stack(torch.meshgrid(*coord_seqs,indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])
    rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb


def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff 
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)

def cheb_loss2(out, n_x, n_y, C=2*torch.pi, x_power = 1, y_power = 1) :
    # x derivative 
    xi_x = (torch.range(0,n_x-1)*(C/n_x)).cuda() # Default C = 2*pi
    fhat_x = torch.fft.fft(out, dim=-2)
    dx = torch.fft.ifft(fhat_x*((1+xi_x**2)**(x_power/2)).view(-1, 1), dim = -2).abs()
    
    # y derivative
    xi_y = (torch.range(0,n_y-1)*(C/n_y)).cuda() # Default C = 2*pi
    fhat_y = torch.fft.fft(out, dim=-1)
    dy = torch.fft.ifft(fhat_y * ((1+xi_y**2)**(y_power/2)), dim=-1).abs()

    return dx, dy

def calc_ssim(sr, hr, dataset=None, scale=1, rgb_range=1):
    sr_norm = sr / rgb_range
    hr_norm = hr / rgb_range

    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if sr_norm.size(1) > 1:
                gray_coeffs = torch.tensor([65.738, 129.057, 25.064],
                                           device=sr_norm.device).view(1, 3, 1, 1) / 256
                sr_norm = (sr_norm * gray_coeffs).sum(dim=1, keepdim=True)
                hr_norm = (hr_norm * gray_coeffs).sum(dim=1, keepdim=True)
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError(f"Not implemented dataset type: {dataset}")

        sr_valid = sr_norm[..., shave:-shave, shave:-shave]
        hr_valid = hr_norm[..., shave:-shave, shave:-shave]
    else:
        sr_valid = sr_norm
        hr_valid = hr_norm

    ssim_val = ssim(sr_valid, hr_valid, data_range=1, size_average=True)

    return ssim_val.item() if isinstance(ssim_val, torch.Tensor) else ssim_val

def calc_lpips(sr, hr, dataset=None, scale=1, rgb_range=1, net='alex'):
    device = sr.device
    lpips_fn = lpips.LPIPS(net=net).to(device)

    sr_norm = sr / rgb_range * 2 - 1
    hr_norm = hr / rgb_range * 2 - 1

    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError(f"Not implemented dataset type: {dataset}")

        sr_valid = sr_norm[..., shave:-shave, shave:-shave]
        hr_valid = hr_norm[..., shave:-shave, shave:-shave]
    else:
        sr_valid = sr_norm
        hr_valid = hr_norm

    lpips_val = lpips_fn(sr_valid, hr_valid)

    lpips_val_mean = lpips_val.mean()

    return lpips_val_mean.item()

def normalize_feature_map(feat_map, dimension=(-2, -1), eps=1e-8):
    """
    Normalize a feature map using z-score normalization over the specified dimensions.

    Args:
        feat_map (Tensor): Feature map tensor of shape (..., H, W)
        dimension (tuple): Dimensions over which to compute mean and std
        eps (float): Small constant to avoid division by zero

    Returns:
        Tensor: Normalized feature map
    """
    mean = feat_map.mean(dim=dimension, keepdim=True)
    std = feat_map.std(dim=dimension, keepdim=True)
    return (feat_map - mean) / (std + eps)

def generate_kernel(feat_map, x_s, y_s, s=1): 
    N_x = feat_map.size(-2)
    N_y = feat_map.size(-1)   

    ind_x = torch.arange(-N_x/2,N_x/2).cuda() * (2 * torch.pi / N_x)
    ind_y = torch.arange(-N_y/2,N_y/2).cuda() * (2 * torch.pi / N_y)

    return (1 + (ind_y**2) + (ind_x[:,None]**2))**(x_s/2)

class PerceptualLoss(nn.Module):
    def __init__(self, layer_ids=[3, 8, 17, 26], use_normalize=True):
        super(PerceptualLoss, self).__init__()
        vgg = torchvision.models.vgg19(pretrained=True).features
        self.layers = nn.ModuleList()
        prev_idx = 0
        for idx in layer_ids:
            block = nn.Sequential(*[vgg[i] for i in range(prev_idx, idx+1)])
            self.layers.append(block.eval()) 
            prev_idx = idx+1
        
        for param in self.parameters():
            param.requires_grad = False

        self.use_normalize = use_normalize
        if self.use_normalize:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x, y):
        """
        x: prediction (B,C,H,W)
        y: ground truth (B,C,H,W)
        return: perceptual loss (scalar)
        """
        if self.use_normalize:
            x = (x - self.mean) / self.std
            y = (y - self.mean) / self.std

        loss = 0.0
        for layer in self.layers:
            x = layer(x)
            y = layer(y)
            loss += F.l1_loss(x, y)
        return loss