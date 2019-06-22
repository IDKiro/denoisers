import numpy as np
import cv2

class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def read_img(filename):
	img = cv2.imread(filename)
	img = img[:,:,::-1] / 255.0
	img = np.array(img).astype('float32')

	return img

def cal_smooth(patch):
    lamda = 0.1

    patch_gp = np.mean(patch, axis=(0, 1))

    loss_smooth = np.mean(np.abs(
                np.concatenate((np.expand_dims(patch[:, :, 0] - patch_gp[0], axis=2),
                                np.expand_dims(patch[:, :, 1] - patch_gp[1], axis=2),
                                np.expand_dims(patch[:, :, 2] - patch_gp[2], axis=2)), axis=2)
                ))

    loss_color = np.mean(np.abs(patch_gp - np.mean(patch_gp)))
    
    loss = loss_smooth + lamda * loss_color

    return loss


def get_noise(patch):
    H = patch.shape[0]
    W = patch.shape[1]

    ps = 64

    smooth_patch = np.zeros([ps, ps, 3])
    best_smooth = 100
    for xx in range(0, W - ps, ps):
        for yy in range(0, H - ps, ps):
            patch_noisy_img = patch[yy:yy+ps, xx:xx+ps, :]
            temp_smooth = cal_smooth(patch_noisy_img)
            if temp_smooth < best_smooth:
                smooth_patch = patch_noisy_img
                best_smooth = temp_smooth
    
    noise = np.concatenate((np.expand_dims(smooth_patch[:, :, 0] - np.mean(smooth_patch, axis=(0, 1))[0], axis=2),
                            np.expand_dims(smooth_patch[:, :, 1] - np.mean(smooth_patch, axis=(0, 1))[1], axis=2),
                            np.expand_dims(smooth_patch[:, :, 2] - np.mean(smooth_patch, axis=(0, 1))[2], axis=2)), axis=2)
    return noise

def data_augment(temp_origin_img, temp_noise_img):
    if np.random.randint(2, size=1)[0] == 1:
        temp_origin_img = np.flip(temp_origin_img, axis=1)
        temp_noise_img = np.flip(temp_noise_img, axis=1)
    if np.random.randint(2, size=1)[0] == 1: 
        temp_origin_img = np.flip(temp_origin_img, axis=0)
        temp_noise_img = np.flip(temp_noise_img, axis=0)
    if np.random.randint(2, size=1)[0] == 1:
        temp_origin_img = np.transpose(temp_origin_img, (1, 0, 2))
        temp_noise_img = np.transpose(temp_noise_img, (1, 0, 2))
    
    return temp_origin_img, temp_noise_img

def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0])
	