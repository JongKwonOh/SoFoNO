import random
import math
from torchvision.transforms import InterpolationMode

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

from datasets import register
from utils import make_coord

import torch
import torch.nn.functional as F

def get_gaussian_kernel(kernel_size=9, sigma=3.0, channels=3):
    x = torch.arange(kernel_size) - kernel_size // 2
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    
    kernel_2d = torch.outer(gauss, gauss)
    kernel_2d = kernel_2d / kernel_2d.sum()
    
    kernel_2d = kernel_2d.expand(channels, 1, kernel_size, kernel_size).clone()
    return kernel_2d

def apply_gaussian_blur(img, kernel_size=9, sigma=3.0):
    channels = img.shape[1]
    kernel = get_gaussian_kernel(kernel_size, sigma, channels).to(img.device)
    return F.conv2d(img, kernel, padding=kernel_size//2, groups=channels)

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img)))

def resize_fn_gaussian(img, size, kernel_size=5, sigma=1.0):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma),
        transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])
    return transform(img)

@register('sr-implicit-downsampled-fast')
class SRImplicitDownsampledFast(Dataset):
    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, additional_augment=False, ranges=None, get_noise=False, get_blur=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.additional_augment = additional_augment
        self.ranges = ranges
        self.get_noise = get_noise
        self.get_blur = get_blur

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)
        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            img = img[:, :h_hr, :w_hr]
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img

        elif self.scale_max > 4:
            h_lr = self.inp_size
            w_lr = self.inp_size
            h_hr = int(round(h_lr * s))
            w_hr = int(round(w_lr * s))
            pad_h = max(0, h_hr - img.shape[-2])
            pad_w = max(0, w_hr - img.shape[-1])
            img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
            H2, W2 = img_pad.shape[-2], img_pad.shape[-1]
            x0 = random.randint(0, H2 - h_hr)
            y0 = random.randint(0, W2 - w_hr)
            crop_hr = img_pad[:, x0:x0 + h_hr, y0:y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        else:
            h_lr = self.inp_size
            w_lr = self.inp_size
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        if self.additional_augment:
            p_rot = 0.5
            p_scale = 0.5

            if random.random() < p_rot:
                k = random.randint(0, 3)
                crop_lr = torch.rot90(crop_lr, k, dims=(-2, -1))
                crop_hr = torch.rot90(crop_hr, k, dims=(-2, -1))

            if random.random() < p_scale:
                scale_noise = random.uniform(0.009, 0.011)

                noisy_h = max(1, int(crop_hr.shape[-2] * scale_noise))
                noisy_w = max(1, int(crop_hr.shape[-1] * scale_noise))
                crop_hr = resize_fn(crop_hr, (noisy_h, noisy_w))
                crop_hr = resize_fn(crop_hr, (h_hr, w_hr))

                noisy_h_lr = max(1, int(h_lr * scale_noise))
                noisy_w_lr = max(1, int(w_lr * scale_noise))
                crop_lr = resize_fn(crop_lr, (noisy_h_lr, noisy_w_lr))
                crop_lr = resize_fn(crop_lr, (h_lr, w_lr))

        if self.get_noise: 
            torch.manual_seed(0)
            sigma = 0.01 
            noise = torch.randn_like(crop_lr) * sigma
            crop_lr = crop_lr + noise
            crop_lr = torch.clamp(crop_lr, 0.0, 1.0)

        if self.get_blur:
            torch.manual_seed(0)
            sigma_b = 0.5
            ksize = 7   
            crop_lr = apply_gaussian_blur(crop_lr.unsqueeze(0), kernel_size=ksize, sigma=sigma_b).squeeze(0)
            crop_lr = torch.clamp(crop_lr, 0.0, 1.0)

        hr_coord = make_coord([h_hr, w_hr], ranges=self.ranges, flatten=False)
        hr_rgb = crop_hr
        if self.inp_size is not None:
            idx = torch.tensor(np.random.choice(h_hr*w_hr, h_lr*w_lr, replace=False))
            #idx,_ = torch.sort(idx)
            hr_coord = hr_coord.view(-1, hr_coord.shape[-1])
            hr_coord = hr_coord[idx, :]
            hr_coord = hr_coord.view(h_lr, w_lr, hr_coord.shape[-1])

            hr_rgb = crop_hr.contiguous().view(crop_hr.shape[0], -1)
            hr_rgb = hr_rgb[:, idx]
            hr_rgb = hr_rgb.view(crop_hr.shape[0], h_lr, w_lr)
        
        cell = torch.tensor([(self.ranges[1] - self.ranges[0]) / crop_hr.shape[-2], (self.ranges[1] - self.ranges[0]) / crop_hr.shape[-1]], dtype=torch.float32)
        
        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
        }    
