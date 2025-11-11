import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.weight * out + self.bias
        return out

@register('galerkin')
class simple_attn(nn.Module):
    def __init__(self, midc, heads): # width, blocks
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv2d(midc, 3*midc, 1)
        self.o_proj1 = nn.Conv2d(midc, midc, 1)
        self.o_proj2 = nn.Conv2d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = nn.GELU()
    
    def forward(self, x, name='0'):
        B, C, H, W = x.shape
        bias = x
        qkv = self.qkv_proj(x).permute(0, 2, 3, 1).reshape(B, H*W, self.heads, 3*self.headc)
        qkv = qkv.permute(0, 2, 1, 3) # (B, self.heads, H*W, 3*self.headc)
        q, k, v = qkv.chunk(3, dim=-1)

        v = self.vln(v)
        k = self.kln(k)
        
        v = torch.matmul(k.transpose(-2,-1), v) / (H*W)
        
        v = torch.matmul(q, v)
        v = v.permute(0, 2, 1, 3).reshape(B, H, W, C)

        ret = v.permute(0, 3, 1, 2) + bias
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias
        
        return bias

class AdaIN(nn.Module):
    # https://github.com/CellEight/Pytorch-Adaptive-Instance-Normalization/blob/master/AdaIN.py
    def __init__(self):
        super().__init__()

    def mu(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
        return torch.sum(x,(2,3))/(x.shape[2]*x.shape[3])

    def sigma(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
        across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
        the permutations are required for broadcasting"""
        return torch.sqrt((torch.sum((x.permute([2,3,0,1])-self.mu(x)).permute([2,3,0,1])**2,(2,3))+0.000000023)/(x.shape[2]*x.shape[3]))

    def forward(self, x, y):
        """ Takes a content embeding x and a style embeding y and changes
        transforms the mean and standard deviation of the content embedding to
        that of the style. [See eq. 8 of paper] Note the permutations are
        required for broadcasting"""
        return (self.sigma(y)*((x.permute([2,3,0,1])-self.mu(x))/self.sigma(x)) + self.mu(y)).permute([2,3,0,1])

class SoFoNO_block(nn.Module):
    def __init__(self, width=256, local_branch='Conv', blocks=16):
        super().__init__()
        self.width = width

        self.local_branch = local_branch
        if self.local_branch=='Conv':
            self.local_b = nn.Sequential(
                nn.Conv2d(self.width, self.width, 1),
                nn.GELU(),
            )
        elif self.local_branch=='Attn':
            self.blocks = blocks
            self.headc = self.width // self.blocks
            self.heads = self.blocks
            self.midc = self.width

            self.qkv_proj = nn.Conv2d(self.width, 3*self.width, 1)

            self.kln = LayerNorm((self.heads, 1, self.headc))
            self.vln = LayerNorm((self.heads, 1, self.headc))
            self.o_proj1 = nn.Conv2d(self.midc, self.midc, 1)
            self.o_proj2 = nn.Conv2d(self.midc, self.midc, 1)

            self.act = nn.GELU()

        self.cross_g = nn.Sequential(
                nn.Conv2d(self.width//2, self.width//2, 1),
                nn.GELU()
            )
        self.cross_l = nn.Sequential(
                nn.Conv2d(self.width//2, self.width//2, 1),
                nn.GELU()
            )

        self.adain = AdaIN()

        self.cat_conv = nn.Sequential(
                nn.Conv2d(self.width, self.width, 1),
                nn.GELU(),
                nn.Conv2d(self.width, self.width, 1),
                nn.GELU(),
            )

    def SoFoNO_LayerNorm(self, x, eps=1e-5, dim=1):
        mean = x.mean(dim, keepdim=True) 
        std = x.std(dim, keepdim=True)

        return (x - mean) / (std + eps)

    def our_transform(self, x, domain):
        F_x = torch.fft.fft2(x)
        F_x = torch.fft.fftshift(F_x, dim=(-2, -1))

        F_x *= (domain / domain.mean(dim=(-2, -1))) 

        out_ft = torch.fft.ifftshift(F_x, dim=(-2, -1))
        tran_x = torch.fft.ifft2(out_ft, s=(out_ft.size(-2), out_ft.size(-1)), dim=(-2, -1)).real

        return tran_x

    def forward(self, x, s, idx, T):
        prev_x = x.clone()
        N_y = x.size(-1)
        N_x = x.size(-2)

        ind_x = torch.arange(-N_x/2,N_x/2).cuda() * (2 * torch.pi / N_x)
        ind_y = torch.arange(-N_y/2,N_y/2).cuda() * (2 * torch.pi / N_y)

        xy_domain =  (1 + (ind_y**2) + (ind_x[:,None]**2))**(s/2)

        tran_x = self.our_transform(x, xy_domain)

        if self.local_branch=='Conv':
            x = self.local_b(self.SoFoNO_LayerNorm(x))

        elif self.local_branch=='Attn':
            B, C, H, W = x.shape
            
            prev_x = x.clone()
            qkv = self.qkv_proj(x).permute(0, 2, 3, 1).reshape(B, H*W, self.heads, 3*self.headc)
            qkv = qkv.permute(0, 2, 1, 3) # (B, self.heads, H*W, 3*self.headc)
            q, k, v = qkv.chunk(3, dim=-1)

            v = self.vln(v)
            k = self.kln(k)

            v = torch.matmul(k.transpose(-2,-1), v) / (H*W)

            v = v.permute(0, 3, 1, 2)
            v = torch.matmul(q, v)
            v = v.permute(0, 2, 1, 3).reshape(B, H, W, C)

            ret = v.permute(0, 3, 1, 2) + prev_x
            x = self.o_proj2(self.act(self.o_proj1(ret))) + prev_x

        x, cross_x = x.chunk(2, dim=1)
        tran_x, cross_tran_x = tran_x.chunk(2, dim=1)
        cross_local = self.cross_l(cross_x)
        cross_global = self.cross_g(cross_tran_x)
        x = self.adain(x, cross_global); tran_x = self.adain(tran_x, cross_local)

        cat_x = torch.cat((x, tran_x), dim=1) + prev_x

        x = self.cat_conv(cat_x)
        
        return x + prev_x
    