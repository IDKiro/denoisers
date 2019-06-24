from __future__ import division
from __future__ import print_function
import os, time, scipy.io, shutil
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import glob
import re
import cv2

from utils import *
from model.seunet import SEUNet


def load_checkpoint(checkpoint_dir):
    if os.path.exists(checkpoint_dir + 'checkpoint.pth.tar'):
        # load existing model
        model_info = torch.load(checkpoint_dir + 'checkpoint.pth.tar')
        print('==> loading existing model:', checkpoint_dir + 'checkpoint.pth.tar')
        model = SEUNet()
        model.cuda()
        model.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch']
    else:
        # create model
        model = SEUNet()
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        cur_epoch = 0

    return model, optimizer, cur_epoch

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, checkpoint_dir + 'checkpoint.pth.tar')
	if is_best:
		shutil.copyfile(checkpoint_dir + 'checkpoint.pth.tar',checkpoint_dir + 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, lr_update_freq):
	if not epoch % lr_update_freq and epoch:
		for param_group in optimizer.param_groups:
			param_group['lr'] = param_group['lr'] * 0.1
	return optimizer


if __name__ == '__main__':
    input_dir = './dataset/train/'
    checkpoint_dir = './checkpoint/seunet/'
    result_dir = './result/seunet/'

    ps = 512
    save_freq = 100
    lr_update_freq = 1000

    train_fns = glob.glob(input_dir + 'Batch_*')

    origin_imgs = [None] * len(train_fns)
    noise_imgs = [None] * len(train_fns)

    for i in range(len(train_fns)):
        origin_imgs[i] = []
        noise_imgs[i] = []

    model, optimizer, cur_epoch = load_checkpoint(checkpoint_dir)

    criterion = nn.L1Loss()
    criterion = criterion.cuda()

    for epoch in range(cur_epoch, 2001):
        cnt=0
        losses = AverageMeter()
        optimizer = adjust_learning_rate(optimizer, epoch, lr_update_freq)
        model.train()

        for ind in np.random.permutation(len(train_fns)):
            train_fn = train_fns[ind]

            if not len(origin_imgs[ind]):
                train_origin_fns = glob.glob(train_fn + '/*Reference.bmp')
                train_noise_fns = glob.glob(train_fn + '/*Noisy.bmp')

                origin_imgs[ind] = read_img(train_origin_fns[0])

                for train_noise_fn in train_noise_fns:
                    noise_img = read_img(train_noise_fn)
                    noise_imgs[ind].append(noise_img)

            st = time.time()
            for nind in np.random.permutation(len(noise_imgs[ind])):
                H = origin_imgs[ind].shape[0]
                W = origin_imgs[ind].shape[1]

                ps_temp = min(H, W, ps) - 1

                xx = np.random.randint(0, W-ps_temp)
                yy = np.random.randint(0, H-ps_temp)
                
                temp_origin_img = origin_imgs[ind][yy:yy+ps_temp, xx:xx+ps_temp, :]
                temp_noise_img = noise_imgs[ind][nind][yy:yy+ps_temp, xx:xx+ps_temp, :]
                temp_origin_img, temp_noise_img = data_augment(temp_origin_img, temp_noise_img)
                
                temp_noise_img_chw = hwc_to_chw(temp_noise_img)
                temp_origin_img_chw = hwc_to_chw(temp_origin_img)

                cnt += 1
                st = time.time()

                input_var = torch.autograd.Variable(
                    torch.from_numpy(temp_noise_img_chw.copy()).type(torch.FloatTensor).unsqueeze(0)
                    )
                target_var = torch.autograd.Variable(
                    torch.from_numpy(temp_origin_img_chw.copy()).type(torch.FloatTensor).unsqueeze(0)
                    )

                input_var, target_var = input_var.cuda(), target_var.cuda()

                output = model(input_var)

                loss = criterion(output, target_var)
                losses.update(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print('[{0}][{1}]\t'
                    'lr: {lr:.5f}\t'
                    'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Time: {time:.3f}'.format(
                    epoch, cnt,
                    lr=optimizer.param_groups[-1]['lr'],
                    loss=losses,
                    time=time.time()-st))

                if epoch % save_freq == 0:
                    if not os.path.isdir(result_dir + '%04d'%epoch):
                        os.makedirs(result_dir + '%04d'%epoch)

                    output_np = output.squeeze().cpu().detach().numpy()
                    output_np = chw_to_hwc(np.clip(output_np, 0, 1))

                    temp = np.concatenate((temp_origin_img, temp_noise_img, output_np), axis=1)
                    scipy.misc.toimage(temp*255, high=255, low=0, cmin=0, cmax=255).save(result_dir + '%04d/train_%d_%d.jpg'%(epoch, ind, nind))
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()}, is_best=0)
