import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict
import torch.nn as nn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--model', default='./SoFoNO.pth')
    parser.add_argument('--scale', default=4)
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--domain', default='-1,1')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    scale_max = 4
    
    img = transforms.ToTensor()(Image.open(args.input).convert('RGB'))
    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
    
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model_ = nn.parallel.DataParallel(model)
    else:
        model_ = model
    
    domain_min, domain_max = args.domain.split(',')
    h = int(img.shape[-2] * int(args.scale))
    w = int(img.shape[-1] * int(args.scale))
    scale = h / img.shape[-2]
    coord_full = make_coord((h, w), ranges=[int(domain_min), int(domain_max)], flatten=False).cuda()
    cell = torch.ones(1, 2).cuda()
    cell[:, 0] *= (int(domain_max) - int(domain_min)) / h
    cell[:, 1] *= (int(domain_max) - int(domain_min)) / w
    
    cell_factor = max(scale/scale_max, 1)
    img_lr_norm = ((img - 0.5) / 0.5).cuda().unsqueeze(0)

    tile_h, tile_w = 512, 512
    overlap = 32
    step_h = tile_h - overlap
    step_w = tile_w - overlap

    out_full = torch.zeros(3, h, w)
    H, W = h, w

    for top_t in range(0, H, step_h):
        for left_t in range(0, W, step_w):
            bottom_t = min(top_t + tile_h, H)
            right_t = min(left_t + tile_w, W)

            coord_tile = coord_full[top_t:bottom_t, left_t:right_t, :].unsqueeze(0).cuda()
            pred_tile = model_(img_lr_norm, coord_tile, cell_factor * cell).squeeze(0)
            pred_tile = (pred_tile * 0.5 + 0.5).clamp(0, 1).reshape(3, bottom_t - top_t, right_t - left_t).cpu()
            out_full[:, top_t:bottom_t, left_t:right_t] = pred_tile

            del coord_tile, pred_tile
            torch.cuda.empty_cache()

    transforms.ToPILImage()(out_full).save(args.output)
