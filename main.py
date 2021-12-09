from __future__ import print_function
import argparse
from argparse import Namespace
import logging
import torch.optim as optim
from model import *
from data_loader import *
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from utils import *
from testing import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')
########################################################################
parser.add_argument('--depth', default=26, type=int)
parser.add_argument('--width', default=1, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--num_samples', default=200, type=int)
########################################################################
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--nepoch', default=150, type=int)
parser.add_argument('--milestone_1', default=75, type=int)
parser.add_argument('--milestone_2', default=125, type=int)
parser.add_argument('--rotation_type', default='rand')
parser.add_argument('--level', default=5, type=int)
parser.add_argument('--dua', default=False, type=bool)
parser.add_argument('--adapt_batch', default=False, type=bool)
parser.add_argument('--train', default=False, type=bool)
parser.add_argument('--test_c', default=False, type=bool)
parser.add_argument('--num_samples_adapt', default=200, type=int)
parser.add_argument('--group_norm', default=0, type=int)
########################################################################
parser.add_argument('--outf', default='.')
args: Namespace = parser.parse_args()
my_makedir(args.outf)
cudnn.benchmark = True
logger = logging.getLogger(__name__)

##############Please Edit This##############
args.dataroot = '/media/mirza/Data/Downloads/test-time-training/'

common_corruptions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
    'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast',
    'elastic_transform', 'pixelate', 'jpeg_compression'
]

severity = [5]

NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

tr_transform_adapt = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*NORM)
])

if args.group_norm == 0:
    norm_layer = nn.BatchNorm2d
else:
    def gn_helper(planes):
        return nn.GroupNorm(args.group_norm, planes)

net = ResNetCifar(args.depth, args.width, channels=3, classes=10, norm_layer=norm_layer).cuda()
net = net.cuda()

parameters = list(net.parameters())

optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, [args.milestone_1, args.milestone_2], gamma=0.1, last_epoch=-1)

criterion = nn.CrossEntropyLoss().cuda()
_, teloader = prepare_test_data(args)
_, trloader = prepare_train_data(args)

if args.train:
    all_err_cls = []
    print('Running...')
    print('Error (%)\t\ttest')
    for epoch in range(1, args.nepoch + 1):
        net.train()
        for batch_idx, (images, labels) in enumerate(trloader):
            optimizer.zero_grad()
            images, labels = images.cuda(), labels.cuda()
            outputs_cls = net(images)
            loss = criterion(outputs_cls, labels)
            loss.backward()
            optimizer.step()

        err_cls = test(teloader, net)[0]
        all_err_cls.append(err_cls)
        scheduler.step()

        print(('Epoch %d/%d:' % (epoch, args.nepoch)).ljust(24) +
              '%.2f' % (err_cls * 100))

        torch.save(all_err_cls, args.outf + '/loss.pth')

        if err_cls <= min(all_err_cls):
            state = {'err_cls': err_cls,
                     'net': net.state_dict(),
                     'optimizer': optimizer.state_dict()}
            torch.save(state, args.outf + '/ckpt.pth')

#######ONLY FOR TESTING CORRUPTIONS WITHOUT ADAPTATION#######
if args.test_c:
    ckpt = torch.load(args.outf + '/ckpt.pth')
    error_dict = {}
    running_mean_list = []
    running_variance = []
    net.load_state_dict(ckpt['net'])
    for index, args.level in enumerate(severity):
        err = []
        for corruption in common_corruptions:
            print(corruption, args.level)
            args.corruption = corruption
            args.batch_size = 10000
            teset, teloader = prepare_test_data(args)

            err_cls = test(teloader, net)[0]
            print('Error: %.2f' %(err_cls * 100))

#######For Adaptation with DUA#######
if args.dua:
    ckpt = torch.load(args.outf + '/ckpt.pth')
    error_dict = {}
    running_mean_list = []
    running_variance = []
    all_errors = []
    decay_factor = 0.94
    min_momentum_constant = 0.005

    for args.level in severity:
        for args.corruption in common_corruptions:
            mom_pre = 0.1
            err = []
            i = 1
            print('#######Beginning DUA#######')
            print(args.corruption, args.level)

            net.load_state_dict(ckpt['net'])

            args.batch_size = 10000
            teset, teloader = prepare_test_data(args)

            args.batch_size = 1
            _, trloader = prepare_train_data(args)

            err_cls = test(teloader, net)[0]
            print('Error Before Adaptation: ', err_cls * 100)

            for i in tqdm(range(1, args.num_samples + 1)):
                net.train()
                image = Image.fromarray(teset.data[i - 1])
                mom_new = (mom_pre * decay_factor)
                for m in net.modules():
                    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm3d):
                        m.momentum = mom_new + min_momentum_constant
                mom_pre = mom_new
                inputs = [(tr_transform_adapt(image)) for _ in range(64)]
                inputs = torch.stack(inputs)
                inputs = inputs.cuda()
                inputs_ssh, labels_ssh = rotate_batch(inputs, 'rand')
                inputs_ssh, labels_ssh = inputs_ssh.cuda(), labels_ssh.cuda()
                outputs_cls = net(inputs_ssh)
                err_cls = test(teloader, net)[0] * 100
                err.append(err_cls)

            plot_adaptation_err(err, args.corruption, args)
            adaptation_error = min(err)

            adaptation_error_min = adaptation_error
            print('Error After Adaptation: %.2f' % adaptation_error_min)

            all_errors.append(adaptation_error_min)
            print('Mean Error after Adaptation %.2f' % (sum(all_errors) / len(all_errors)))
            plot_adaptation_err(err, args.corruption, args)

