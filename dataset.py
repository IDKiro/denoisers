import os
import random
import torch
import numpy as np
import glob
import PIL.Image as Image
from torch.utils.data import Dataset

from utils import read_img, hwc_to_chw

class loadedDataset(Dataset):
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.batches = sorted(os.listdir(self.root_dir))

	def __len__(self):
		l = len(self.batches)
		return l

	def __getitem__(self, idx):
		batch_path = os.path.join(self.root_dir, self.batches[idx])

		origin_fns = glob.glob(batch_path + '/*Reference.bmp')
		noise_fns = glob.glob(batch_path + '/*Noisy.bmp')

		origin_fn = origin_fns[0]
		noise_fn = random.choice(noise_fns)

		origin_img = hwc_to_chw(read_img(origin_fn))
		noise_img = hwc_to_chw(read_img(noise_fn))

		return noise_img, origin_img
