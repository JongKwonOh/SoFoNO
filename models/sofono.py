import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
from copy import deepcopy

import models
from models.block import simple_attn, SoFoNO_block
from models import register
from utils import make_coord
from utils import show_feature_map


@register('sofono')
class SoFoNO(nn.Module):
    def __init__(self, encoder_spec, width=256, T=2, ranges=None,
                local_branch='Conv', init_s=0):
        super().__init__()
        self.width = width
        self.T = T
        self.ranges = ranges
        self.encoder = models.make(encoder_spec)
        self.conv00 = nn.Conv2d((64 + 2)*4+2, self.width, 1)
            
        self.fno_kernels = nn.ModuleList()
        self.local_branch = local_branch # Conv or Attn
        self.init_s = init_s
        self.s = torch.nn.Parameter(torch.tensor(self.init_s, dtype=torch.float32, requires_grad=True))
        for t in range(T):
            self.fno_kernels.append(SoFoNO_block(width=self.width, local_branch=self.local_branch))
            
        self.fc1 = nn.Conv2d(self.width, self.width, 1)
        self.fc2 = nn.Conv2d(self.width, 3, 1)
    
    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        return self.feat
        
    def query_rgb(self, coord, cell):      
        feat = (self.feat)
        grid = 0

        pos_lr = make_coord(feat.shape[-2:], flatten=False, ranges = self.ranges).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        if self.ranges == None :
            domain_min, domain_max = -1, 1
        else :
            domain_min, domain_max = self.ranges

        rx = (domain_max - domain_min) / feat.shape[-2] / 2
        ry = (domain_max - domain_min) / feat.shape[-1] / 2
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6
        
        rel_coords = []
        feat_s = []
        areas = []

        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(domain_min + 1e-6, domain_max - 1e-6)

                feat_ = F.grid_sample(feat, coord_.flip(-1), mode='nearest', align_corners=False)

                old_coord = F.grid_sample(pos_lr, coord_.flip(-1), mode='nearest', align_corners=False)
                rel_coord = coord.permute(0, 3, 1, 2) - old_coord
                rel_coord[:, 0, :, :] *= feat.shape[-2] # / (domain_max - domain_min)
                rel_coord[:, 1, :, :] *= feat.shape[-1] # / (domain_max - domain_min)

                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)

                rel_coords.append(rel_coord)
                feat_s.append(feat_)
        
        rel_cell = cell.clone()
        rel_cell[:,0] *= feat.shape[-2] # / (domain_max - domain_min)
        rel_cell[:,1] *= feat.shape[-1] # / (domain_max - domain_min)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t

        for index, area in enumerate(areas):
            feat_s[index] = feat_s[index] * (area / tot_area).unsqueeze(1)
         
        grid = torch.cat([*rel_coords, *feat_s, \
            rel_cell.unsqueeze(-1).unsqueeze(-1).repeat(1,1,coord.shape[1],coord.shape[2])],dim=1)

        x = self.conv00(grid)
        
        for i, kernel in enumerate(self.fno_kernels):
            x = kernel(x, self.s, i, self.T)
            
        feat = x
        ret = self.fc2(F.gelu(self.fc1(feat)))

        ret = ret + F.grid_sample(self.inp, coord.flip(-1), mode='bilinear',\
                                padding_mode='border', align_corners=False)
        
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)

    def get_feature_map(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)

