from __future__ import division
from __future__ import print_function
import os, time, scipy.io, shutil
import numpy as np
import glob
import argparse
import importlib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from utils import *
from model import *
from loader import loadedDataset


parser = argparse.ArgumentParser(description = 'Train')
parser.add_argument('model', default='unet', type=str, help = 'model name (default: UNet)')
parser.add_argument('-ps', default=512, type=int, help = 'patch size')
parser.add_argument('-lr', default=1e-4, type=float, help = 'learning rate')
parser.add_argument('-epochs', default=2000, type=int, help = 'sum of epochs')
parser.add_argument('-freq', default=1000, type=int, help = 'learning rate update frequency')
args = parser.parse_args()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, os.path.join(checkpoint_dir, 'checkpoint.pth.tar'))
	if is_best:
		shutil.copyfile(os.path.join(checkpoint_dir, 'checkpoint.pth.tar'), os.path.join(checkpoint_dir, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, lr_update_freq):
	if not epoch % lr_update_freq and epoch:
		for param_group in optimizer.param_groups:
			param_group['lr'] = param_group['lr'] * 0.1
	return optimizer


ps = args.ps
save_freq = 100
lr_update_freq = args.freq

input_dir = './dataset/train/'
model = importlib.import_module('.' + args.model, package='model').Network()

checkpoint_dir = os.path.join('./checkpoint/', args.model)
result_dir = os.path.join('./result/', args.model)

model.cuda()

if os.path.exists(os.path.join(checkpoint_dir, 'checkpoint.pth.tar')):
    # load existing model
    model_info = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pth.tar'))
    print('==> loading existing model:', os.path.join(checkpoint_dir, 'checkpoint.pth.tar'))
    model.load_state_dict(model_info['state_dict'])
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(model_info['optimizer'])
    cur_epoch = model_info['epoch']
else:
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # create model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    cur_epoch = 0

# TODO: loss
if args.model == 'cbdnet':
    criterion = importlib.import_module('.cbdnet', package='model').asym_loss()
else:
    criterion = nn.L1Loss()
criterion = criterion.cuda()

train_dataset = loadedDataset(input_dir, patch_size=ps)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1, shuffle=True, pin_memory=True)

for epoch in range(cur_epoch, args.epochs + 1):
    losses = AverageMeter()
    optimizer = adjust_learning_rate(optimizer, epoch, lr_update_freq)
    model.train()

    st = time.time()
    for ind, (noise_img, origin_img) in enumerate(train_loader):
        st = time.time()

        input_var = torch.autograd.Variable(noise_img).cuda()
        target_var = torch.autograd.Variable(origin_img).cuda()

        # TODO: output
        if args.model == 'cbdnet':
            noise_level = noise_img - origin_img
            noise_level_var = torch.autograd.Variable(noise_level).cuda()
            noise_level_est, output = model(input_var)
            loss = criterion(output, target_var, noise_level_est, noise_level_var)
        else:
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
            epoch, ind,
            lr=optimizer.param_groups[-1]['lr'],
            loss=losses,
            time=time.time()-st))

        if epoch % save_freq == 0:
            if not os.path.isdir(os.path.join(result_dir, '%04d'%epoch)):
                os.makedirs(os.path.join(result_dir, '%04d'%epoch))

            origin_np = origin_img.numpy()
            noise_np = noise_img.numpy()
            output_np = output.cpu().detach().numpy()

            for indp in range(output_np.shape[0]):
                origin_np_img = chw_to_hwc(origin_np[indp])
                noise_np_img = chw_to_hwc(noise_np[indp])
                output_img = chw_to_hwc(np.clip(output_np[indp], 0, 1))

                temp = np.concatenate((origin_np_img, noise_np_img, output_img), axis=1)
                scipy.misc.toimage(temp*255, high=255, low=0, cmin=0, cmax=255).save(os.path.join(result_dir, '%04d/train_%d_%d.jpg'%(epoch, ind, indp)))

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict()}, is_best=0)
