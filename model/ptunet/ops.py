import numpy as np


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