import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn

import datasets
import models
import utils


def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell)
            ql = qr
            preds.append(pred)
        pred = torch.cat(preds, dim=2)
    return pred


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None, scale_max=4,
              verbose=False,mcell=False):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    cnt = 0
    for batch in pbar:
        cnt+=1

        for k, v in batch.items():
            batch[k] = v.cuda(non_blocking=True)

        inp = (batch['inp'] - inp_sub) / inp_div
        coord = batch['coord']
        cell = batch['cell']

        if mcell == False: c = 1
        else : c = max(scale/scale_max, 1)

        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, coord, cell*c)
        else:
            pred = batched_predict(model, inp, coord, cell*c, eval_bsize)

        with torch.no_grad():
            pred = pred * gt_div + gt_sub
            pred.clamp_(0, 1)
            res = metric_fn(pred, batch['gt'])
            # torch.cuda.empty_cache()

        val_res.add(res.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {:.2f}'.format(val_res.item()))
            
        torch.cuda.empty_cache()

    return val_res.item()

def eval_ssim(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              scale_max=4, verbose=False, mcell=False):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt':  {'sub': [0], 'div': [1]}
        }

    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()

    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_ssim  # dataset=None, scale=1
        scale = 1
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_ssim, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_ssim, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError(f"Unknown eval_type: {eval_type}")

    val_res = utils.Averager() 

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda(non_blocking=True)

        inp = (batch['inp'] - inp_sub) / inp_div
        coord = batch['coord']
        cell = batch['cell']

        if not mcell:
            c = 1
        else:
            c = max(scale / scale_max, 1)

        with torch.no_grad():
            if eval_bsize is None:
                pred = model(inp, coord, cell * c)
            else:
                pred = batched_predict(model, inp, coord, cell * c, eval_bsize)

        with torch.no_grad():
            pred = pred * gt_div + gt_sub
            pred.clamp_(0, 1)

            res = metric_fn(pred, batch['gt'])
            if isinstance(res, torch.Tensor):
                res = res.item()

        val_res.add(res, inp.shape[0])

        if verbose:
            pbar.set_description('val SSIM: {:.4f}'.format(val_res.item()))

        torch.cuda.empty_cache()

    return val_res.item()

def eval_lpips(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
               scale_max=4, verbose=False, mcell=False):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt':  {'sub': [0], 'div': [1]}
        }

    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()

    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_lpips
        scale = 1
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_lpips, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_lpips, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError(f"Unknown eval_type: {eval_type}")

    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda(non_blocking=True)

        inp = (batch['inp'] - inp_sub) / inp_div
        coord = batch['coord']
        cell = batch['cell']

        if not mcell:
            c = 1
        else:
            c = max(scale / scale_max, 1)

        with torch.no_grad():
            if eval_bsize is None:
                pred = model(inp, coord, cell * c)
            else:
                pred = batched_predict(model, inp, coord, cell * c, eval_bsize)

        with torch.no_grad():
            pred = pred * gt_div + gt_sub
            pred.clamp_(0, 1)

            res = metric_fn(pred, batch['gt'])
            if isinstance(res, torch.Tensor):
                res = res.item()

        val_res.add(res, inp.shape[0])

        if verbose:
            pbar.set_description('val LPIPS: {:.4f}'.format(val_res.item()))

        torch.cuda.empty_cache()

    return val_res.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='./configs/test_srno.yaml')
    # parser.add_argument('--model')
    parser.add_argument('--scale_max', default='4')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--mcell', default=False)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #print(os.environ['CUDA_VISIBLE_DEVICES'])

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('Test Config Loaded.')

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True, shuffle=False)
    
    model_spec = torch.load(config['model_path'])['model']
    model_spec['name'] = 'sofono'
    model_spec['args'].pop('blocks', None)
    model_spec['args'].pop('integral', None)
    model = models.make(model_spec, load_sd=True).cuda()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model_ = nn.parallel.DataParallel(model)
    else :
        model_ = model

    min_scale = config['test_dataset']['wrapper']['args']['scale_min']
    max_scale = config['test_dataset']['wrapper']['args']['scale_max']
    print(f'Test Min Scale : {min_scale}')
    print(f'Test Max Scale : {max_scale}')
    import time
    t1= time.time()
    res_psnr = eval_psnr(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        scale_max = max_scale,
        verbose=True,
        mcell=bool(args.mcell))
    t2 =time.time()
    print('PSNR result: {:.2f}'.format(res_psnr), utils.time_text(t2-t1))

    t1= time.time()
    res_ssim = eval_ssim(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        scale_max = max_scale,
        verbose=True,
        mcell=bool(args.mcell))
    t2 =time.time()
    print('SSIM result: {:.4f}'.format(res_ssim), utils.time_text(t2-t1))
    
    t1= time.time()
    res_lpips = eval_lpips(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        scale_max = max_scale,
        verbose=True,
        mcell=bool(args.mcell))
    t2 =time.time()
    print('LPIPS result: {:.4f}'.format(res_lpips), utils.time_text(t2-t1))
    
