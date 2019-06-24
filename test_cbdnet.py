from __future__ import division
from __future__ import print_function
import os, time, scipy.io, shutil
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
import numpy as np
import glob
import re
import cv2

from utils import *
from model.cbdnet import CBDNet


def load_checkpoint(checkpoint_dir):
    if os.path.exists(checkpoint_dir + 'checkpoint.pth.tar'):
        # load existing model
        model_info = torch.load(checkpoint_dir + 'checkpoint.pth.tar')
        print('==> loading existing model:', checkpoint_dir + 'checkpoint.pth.tar')
        model = CBDNet()
        model.cuda()
        model.load_state_dict(model_info['state_dict'])
    else:
        print('Error: No trained model detected!')
        exit(1)

    return model
    

if __name__ == '__main__':
    input_dir = './dataset/test/'
    checkpoint_dir = './checkpoint/cbdnet/'

    test_fns = glob.glob(input_dir + 'Batch_*')

    model = load_checkpoint(checkpoint_dir)
    model.eval()

    psnr = AverageMeter()
    ssim = AverageMeter()
    stime = AverageMeter()

    for i, test_fn in enumerate(test_fns):
        test_origin_fns = glob.glob(test_fn + '/*Reference.bmp')
        test_noise_fns = glob.glob(test_fn + '/*Noisy.bmp')

        origin_img = read_img(test_origin_fns[0])
        origin_img = origin_img[0:512, 0:512, :] # TODO

        for test_noise_fn in test_noise_fns:
            noise_img = read_img(test_noise_fn)
            noise_img = noise_img[0:512, 0:512, :]  # TODO
            noise_img_chw = hwc_to_chw(noise_img)

            input_var = torch.autograd.Variable(
                torch.from_numpy(noise_img_chw.copy()).type(torch.FloatTensor).unsqueeze(0)
                )

            input_var = input_var.cuda()

            st = time.time()
            output = model(input_var)
            spend_time = time.time() - st

            output_np = output.squeeze().cpu().detach().numpy()
            output_np = chw_to_hwc(np.clip(output_np, 0, 1))

            test_psnr = compare_psnr(origin_img, output_np, data_range=1)
            test_ssim = compare_ssim(origin_img, output_np, data_range=1, multichannel=True)
            
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
