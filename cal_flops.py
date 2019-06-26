import argparse
from thop import profile
from model import unet, seunet, ssunet, gcunet, cbdnet, dncnn, rdn, n3net

parser = argparse.ArgumentParser(description = 'FLOPS')
parser.add_argument('model', default='unet', type=str, help = 'model name (default: UNet)')
args = parser.parse_args()

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

flops, params = profile(model, input_size=(1, 3, 512, 512))


print('FLOPs: {flops:.1f} G\t'
    'Params: {params:.1f} M'.format(
    flops=flops*1e-9,
    params=params*1e-6))