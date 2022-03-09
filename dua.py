from __future__ import print_function
import argparse
from argparse import Namespace
from models.resnet_26 import *
from utils.data_loader import *
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from utils.testing import test
from utils.rotation import *
from models.wide_resnet import WideResNet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--depth', default=26, type=int)
parser.add_argument('--width', default=1, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--group_norm', default=0, type=int)
parser.add_argument('--dataroot', default='/media/mirza/Data/Downloads/test-time-training/')
parser.add_argument('--ckpt_path', default='/media/mirza/Data/Downloads/ttt_adaptation/ckpt/cifar10/corruptions/Hendrycks2020AugMix_WRN.pt')
parser.add_argument('--model', default='wrn')
parser.add_argument('--num_samples', default=80, type=int)
parser.add_argument('--decay_factor', default=0.94, type=float)
parser.add_argument('--min_mom', default=0.005, type=float)
parser.add_argument('--rotation_type', default='rand')
parser.add_argument('--level', default=5, type=int)
args: Namespace = parser.parse_args()

cudnn.benchmark = True

severity = [5]
common_corruptions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
    'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast',
    'elastic_transform', 'pixelate', 'jpeg_compression'
]

tr_transform_adapt = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*NORM)
])

ckpt = torch.load(args.ckpt_path)

if args.group_norm == 0:
    norm_layer = nn.BatchNorm2d
else:
    def gn_helper(planes):
        return nn.GroupNorm(args.group_norm, planes)

if args.model == 'wrn':
    net = WideResNet(widen_factor=2, depth=40, num_classes=10)

elif args.model == 'res':
    net = ResNetCifar(args.depth, args.width, channels=3, classes=10, norm_layer=norm_layer).cuda()

net = net.cuda()
decay_factor = args.decay_factor
min_momentum_constant = args.min_mom

for args.level in severity:
    print(f'Starting DUA for Level {args.level}')
    all_errors = []
    for args.corruption in common_corruptions:
        mom_pre = 0.1
        err = []
        print(f'Corruption - {args.corruption} :::: Level - {args.level}')
        net.load_state_dict(ckpt)
        teset, teloader = prepare_test_data(args)
        err_cls = test(teloader, net)[0] * 100
        print(f'Error Before Adaptation: {err_cls:.1f}')

        for i in tqdm(range(1, args.num_samples + 1)):
            net.eval()
            image = Image.fromarray(teset.data[i - 1])
            mom_new = (mom_pre * decay_factor)
            for m in net.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm3d):
                    m.train()
                    m.momentum = mom_new + min_momentum_constant
            mom_pre = mom_new
            inputs = [(tr_transform_adapt(image)) for _ in range(64)]
            inputs = torch.stack(inputs)
            inputs = inputs.cuda()
            inputs_ssh, labels_ssh = rotate_batch(inputs, 'rand')
            inputs_ssh, labels_ssh = inputs_ssh.cuda(), labels_ssh.cuda()
            _ = net(inputs_ssh)
            err_cls = test(teloader, net)[0] * 100
            err.append(err_cls)
        adaptation_error = min(err)
        print(f'Error After Adaptation: {adaptation_error:.1f}')
        all_errors.append(adaptation_error)
    print(f'Mean Error after Adaptation {(sum(all_errors) / len(all_errors)):.1f}')
