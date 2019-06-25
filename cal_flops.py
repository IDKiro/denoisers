import argparse
from thop import profile
from model import unet, seunet, ssunet, gcunet, cbdnet, dncnn, rdn

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
    model = dncnn.DnCNN(3)
elif args.model == 'rdn':
    model = rdn.RDN()
else:
    print('Error: no support model detected!')
    exit(1)

flops, params = profile(model, input_size=(1, 3, 512, 512))


print('FLOPS: {flops:.1f}\t'
    'PARAMS: {params:.1f}'.format(
    flops=flops*1e-9,
    params=params*1e-6))