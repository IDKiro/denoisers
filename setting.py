import torch
import torch.nn as nn
from model import *

def model_def(model_name):
    # TODO: model
    if model_name == 'unet':
        model = unet.UNet()
    elif model_name == 'seunet':
        model = seunet.SEUNet()
    elif model_name == 'ssunet':
        model = ssunet.SSUNet()
    elif model_name == 'gcunet':
        model = gcunet.GCUNet()
    elif model_name == 'cbdnet':
        model = cbdnet.CBDNet()
    elif model_name == 'dncnn':
        model = dncnn.DnCNN()
    elif model_name == 'rdn':
        model = rdn.RDN()
    elif model_name == 'n3net':
        model = n3net.N3Net(3, 3, 3,
                            nblocks=1, 
                            block_opt={'features':64, 'kernel':3, 'depth':17, 'residual':1, 'bn':0}, 
                            nl_opt={'k':4}, residual=False)
    elif model_name == 'n3unet':
        model = n3unet.N3UNet()       
    elif model_name == 'mobileunet':
        model = mobileunet.MobileUNet()                 
    elif model_name == 'nmunet':
        model = nmunet.NMUNet() 
    elif model_name == 'durbnet':
        model = durbnet.DuRBNet() 
    else:
        print('Error: no support model detected!')
        exit(1)

    return model

def loss_def(loss_name):
    # TODO: loss
    if loss_name == 'cbdnet':
        criterion = cbdnet.asym_loss()
    else:
        criterion = nn.L1Loss()
    criterion = criterion.cuda()

    return criterion