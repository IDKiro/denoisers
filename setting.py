import torch
import torch.nn as nn
from model import *

def model_def(model_name):
    # TODO: model
    if model_name == 'unet':
        model = unet.UNet()
    elif model_name == 'unets':
        model = unets.UNetS()
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
        model = n3net.N3Net()   
    elif model_name == 'mobileunet':
        model = mobileunet.MobileUNet()                 
    elif model_name == 'nmunet':
        model = nmunet.NMUNet() 
    elif model_name == 'durb':
        model = durb.DuRB() 
    elif model_name == 'memnet':
        model = memnet.MemNet() 
    elif model_name == 'rednet':
        model = rednet.REDNet() 
    elif model_name == 'carn':
        model = carn.CARN() 
    elif model_name == 'carnm':
        model = carnm.CARNM() 
    elif model_name == 'snunet':
        model = snunet.SNUNet() 
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