from __future__ import division
from __future__ import print_function
import os, time, scipy.io, shutil
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from skimage.measure import compare_psnr, compare_ssim
import numpy as np
import glob
import argparse

from utils import *
from model import unet, seunet, ssunet, gcunet, cbdnet, dncnn, rdn, n3net

parser = argparse.ArgumentParser(description = 'Test')
parser.add_argument('model', default='unet', type=str, help = 'model name (default: UNet)')
parser.add_argument('--cpu', nargs='?', const=1, help = 'Use CPU')
args = parser.parse_args()

input_dir = './dataset/test/'
test_fns = glob.glob(input_dir + 'Batch_*')

# TODO: model
if args.model == 'unet':
    model = unet.UNet()
elif args.model == 'seunet':
    model = seunet.SEUNet()
elif args.model == 'ssunet':
    model = ssunet.SSUNet()
elif args.model == 'gcunet':
    model = gcunet.GCUNet()
elif args.model == 'cbdnet':
    model = cbdnet.CBDNet()
elif args.model == 'dncnn':
    model = dncnn.DnCNN()
elif args.model == 'rdn':
    model = rdn.RDN()
elif args.model == 'n3net':
    model = n3net.N3Net(3, 3, 3,
                        nblocks=1, 
                        block_opt={'features':64, 'kernel':3, 'depth':17, 'residual':1, 'bn':0}, 
                        nl_opt={'k':4}, residual=False)
else:
    print('Error: no support model detected!')
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

psnr = AverageMeter()
ssim = AverageMeter()
stime = AverageMeter()

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

                st = time.time()

                # TODO: output
                if args.model == 'cbdnet':
                    _, output = model(input_var)
                else:
                    output = model(input_var)

                spend_time = time.time() - st

                output_np = output.squeeze().cpu().detach().numpy()
                output_np = chw_to_hwc(np.clip(output_np, 0, 1))

                test_psnr = compare_psnr(temp_origin_img, output_np, data_range=1)
                test_ssim = compare_ssim(temp_origin_img, output_np, data_range=1, multichannel=True)
                
                psnr.update(test_psnr)
                ssim.update(test_ssim)
                if i > 0:
                    stime.update(spend_time * 1000) # ms

                print('PSNR: {psnr.val:.4f} ({psnr.avg:.4f})\t'
                    'SSIM: {ssim.val:.4f} ({ssim.avg:.4f})\t'
                    'Time: {time.val:.2f} ({time.avg:.2f})'.format(
                    psnr=psnr,
                    ssim=ssim,
                    time=stime))

print('PSNR: {psnr.avg:.4f}\t'
    'SSIM: {ssim.avg:.4f}\t'
    'Time: {time.avg:.2f}'.format(
    psnr=psnr,
    ssim=ssim,
    time=stime))