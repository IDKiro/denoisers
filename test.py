from __future__ import division
from __future__ import print_function
import os, time, scipy.io, shutil
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from skimage.measure import compare_psnr, compare_ssim
from thop import profile
import numpy as np
import glob
import argparse

from utils import *
from model import *
from setting import model_def


parser = argparse.ArgumentParser(description = 'Test')
parser.add_argument('model', default='unet', type=str, help = 'model name (default: UNet)')
parser.add_argument('--cpu', nargs='?', const=1, help = 'Use CPU')
parser.add_argument('--flops', nargs='?', const=1, help = 'Calculate FLOPs')
parser.add_argument('--fps', nargs='?', const=1, help = 'Measure FPS')
args = parser.parse_args()

def run(input_var):
    # TODO: output
    with torch.no_grad():
        if args.model == 'cbdnet':
            _, output = model(input_var)
        else:
            output = model(input_var)

        return output


input_dir = './dataset/test/'
test_fns = glob.glob(input_dir + 'Batch_*')

model = model_def(args.model)

if args.flops:
    flops, params = profile(model, input_size=(1, 3, 512, 512))
    print('FLOPs: {flops:.1f} G\t'
        'Params: {params:.1f} M'.format(
        flops=flops*1e-9,
        params=params*1e-6))
    exit(1)

checkpoint_dir = os.path.join('./checkpoint/', args.model)

if not args.cpu:
    print('Using GPU!')
    model.cuda()
else:
    print('Using CPU!')

if os.path.exists(os.path.join(checkpoint_dir, 'checkpoint.pth.tar')):
    # load existing model
    model_info = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pth.tar'))
    print('==> loading existing model:', os.path.join(checkpoint_dir, 'checkpoint.pth.tar'))
    model.load_state_dict(model_info['state_dict'])
else:
    print('Error: no trained model detected!')
    exit(1)

model.eval()

if args.fps:
    test_img = hwc_to_chw(
        read_img(glob.glob(test_fns[0] + '/*Reference.bmp')[0])[0:512, 0:512, :]
    )

    input_var = torch.autograd.Variable(
        torch.from_numpy(test_img.copy()).type(torch.FloatTensor).unsqueeze(0)
        )
    
    if not args.cpu:
        input_var = input_var.cuda()

    ITER = 1000

    output = run(input_var)

    print('Testing...')
    st = time.time()
    for _ in range(ITER):
        output = run(input_var)

    print('FPS: {fps:.1f}'.format(
    fps=ITER / (time.time()-st)))
    exit(1)


psnr = AverageMeter()
ssim = AverageMeter()

for i, test_fn in enumerate(test_fns):
    test_origin_fns = glob.glob(test_fn + '/*Reference.bmp')
    test_noise_fns = glob.glob(test_fn + '/*Noisy.bmp')

    origin_img = read_img(test_origin_fns[0])

    for test_noise_fn in test_noise_fns:
        noise_img = read_img(test_noise_fn)

        for ix in range(0, 8, 2):
            for iy in range(0, 4, 2):
                temp_origin_img = origin_img[512*ix:512*(ix+1), 512*iy:512*(iy+1), :]
                temp_noise_img = noise_img[512*ix:512*(ix+1), 512*iy:512*(iy+1), :]

                temp_noise_img_chw = hwc_to_chw(temp_noise_img)

                input_var = torch.autograd.Variable(
                    torch.from_numpy(temp_noise_img_chw.copy()).type(torch.FloatTensor).unsqueeze(0)
                    )

                if not args.cpu:
                    input_var = input_var.cuda()

                output = run(input_var)

                output_np = output.squeeze().cpu().detach().numpy()
                output_np = chw_to_hwc(np.clip(output_np, 0, 1))

                test_psnr = compare_psnr(temp_origin_img, output_np, data_range=1)
                test_ssim = compare_ssim(temp_origin_img, output_np, data_range=1, multichannel=True)
                
                psnr.update(test_psnr)
                ssim.update(test_ssim)

                print('PSNR: {psnr.val:.2f} ({psnr.avg:.2f})\t'
                    'SSIM: {ssim.val:.3f} ({ssim.avg:.3f})'.format(
                    psnr=psnr,
                    ssim=ssim))

print('PSNR: {psnr.avg:.2f}\t'
    'SSIM: {ssim.avg:.3f}'.format(
    psnr=psnr,
    ssim=ssim))