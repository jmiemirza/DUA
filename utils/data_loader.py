import copy
import logging
import os
from os.path import exists, join, normpath
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from torch import manual_seed, randperm
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import CIFAR10

import globals
import config
from utils.datasets import CIFAR, ImgNet
from utils.datasets import LoadImagesAndLabels as Kitti
from utils.general import check_img_size, increment_path
from utils.torch_utils import torch_distributed_zero_first

log = logging.getLogger('MAIN.DATA')

NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

te_transforms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(*NORM)
                                    ])

tr_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(*NORM)
                                    ])

NORM_IMGNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

tr_transforms_imgnet = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(*NORM_IMGNET)
                                           ])

te_transforms_imgnet = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(*NORM_IMGNET)
                                           ])


def get_loader(args, split='train', joint=False, shuffle=True, pad=0.0, aug=False, rect=False):
    """
        Create the loader for the specified split (train/val/test) and
        current task (args.task).
        If joint=True the dataset will be created from the current task and
        all tasks which came before it (in globals.TASKS), combined.
        Parameters: padding (pad), augment (aug) and rectangular training (rect)
        only apply to the yolov3 model and are ignored for other models.
        YOLOv3 rectangular training (rect) is incompatible with dataloader
        shuffle (shuffle) and shuffle will be set to False silently if that
        combination of parameters is supplied to this function.
    """
    if args.model == 'yolov3' and rect and shuffle:
        shuffle = False

    if args.dataset == 'kitti' or not joint:
        # Create loader for joint or non-joint KITTI dataset, as well as
        # non-joint loaders for other datasets
        ds = get_dataset(args, split=split, pad=pad, aug=aug, rect=rect, joint=joint)
        collate_fn = Kitti.collate_fn if args.dataset == 'kitti' else None
        loader = DataLoader if args.model != 'yolov3' or args.image_weights else InfiniteDataLoader
        rank = args.global_rank if args.model == 'yolov3' and split == 'train' else -1
        return loader(ds, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.workers, collate_fn=collate_fn,
                      pin_memory=True)
    else:
        # Create joint loaders for datasets other than KITTI
        datasets = []
        current_task = args.task
        for args.task in ['initial'] + globals.TASKS:
            datasets.append(get_dataset(args, split=split))
            if current_task == args.task:
                break
        return DataLoader(ConcatDataset(datasets),
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=args.workers)


def get_dataset(args, split=None, pad=0.0, aug=False, rect=False, joint=False):
    """
        Create dataset based on args and split
        Parameters: padding (pad), augment (aug), rectangular training (rect) and
        joint training (joint) only apply to the yolov3 model and are ignored
        for other models.
    """
    if not hasattr(args, 'task'):
        args.task = 'initial'
    if args.task not in ['initial'] + globals.TASKS:
        raise Exception(f'Invalid task: {args.task}')

    if args.dataset == 'cifar10':
        transform = tr_transforms if split == 'train' else te_transforms
        ds = CIFAR(args.dataroot, args.task, split=split, transform=transform,
                   severity=int(args.severity))
        if split != 'test':
            # train and val split are being created from the train set
            ds = get_split_subset(args, ds, split)

    elif args.dataset in ['imagenet', 'imagenet-mini']:
        transform = tr_transforms_imgnet if split == 'train' else te_transforms_imgnet

        ds = ImgNet(args.dataroot, split, args.task, args.severity, transform)
        if split != 'val':
            # train and test split are being created from the train set
            ds = get_split_subset(args, ds, split)

    elif args.dataset == 'kitti':
        path = join(args.dataroot, f'{split}.txt')
        img_size_idx = split != 'train'
        img_size = check_img_size(img_size=args.img_size[img_size_idx], s=args.gs)
        img_dirs_paths = []
        if joint:
            # put paths to all tasks image directories into img_dirs_paths
            for t in ['initial'] + globals.TASKS:
                if t != 'initial':
                    if args.severity_idx < len(globals.KITTI_SEVERITIES[args.task]):
                        args.severity = globals.KITTI_SEVERITIES[t][args.severity_idx]
                    else:
                        continue
                img_dir = 'images' if t == 'initial' else f'{args.severity}'
                img_dirs_paths.append(join(args.dataroot, f'{t}', img_dir))
                if t == args.task:
                    break
        else:
            img_dir = 'images' if args.task == 'initial' else f'{args.severity}'
            img_dirs_paths.append(join(args.dataroot, f'{args.task}', img_dir))

        with torch_distributed_zero_first(-1):
            ds = Kitti(path, img_size, args.batch_size,
                       augment=aug, hyp=args.yolo_hyp(), rect=rect,
                       stride=int(args.gs), pad=pad, imgs_dir=img_dirs_paths)
    return ds


def get_split_subset(args, ds, split):
    """
        Create a subset of given dataset (ds).
        Specifically defined for CIFAR10 and ImageNet, as they either do not
        have a labeled validation set or test set, therefore we create them
        here from the their train sets.
        args.split_seed is used to define a seed to be able to reproduce a split.
        args.split_ratio defines how much percent of the train set will be used
        as validation/test set (e.g. args.split_ratio = 0.3 for CIFAR10 means
        30% of the train set will be used as validation set and the remaining
        70% will be the train set).
    """
    manual_seed(args.split_seed)
    indices = randperm(len(ds))
    valid_size = round(len(ds) * args.split_ratio)

    if args.dataset == 'cifar10':
        if split == 'train':
            ds = Subset(ds, indices[:-valid_size])
        elif split == 'val':
            ds = Subset(ds, indices[-valid_size:])

    elif args.dataset in ['imagenet', 'imagenet-mini']:
        if split == 'train':
            ds = Subset(ds, indices[:-valid_size])
        elif split == 'test':
            ds = Subset(ds, indices[-valid_size:])

    return ds


def get_image_from_idx(self, idx: int = 0):
    return self.dataset.get_image_from_idx(idx)
Subset.get_image_from_idx = get_image_from_idx


def set_yolo_save_dir(args, baseline, scenario):
    """
        Sets args.save_dir which is used in yolov3 training to save results
    """
    p = join(args.checkpoints_path, args.dataset, args.model, baseline,
             scenario, f'{args.task}_{args.severity}_train_results')
    args.save_dir = increment_path(Path(p), exist_ok=args.exist_ok)


def set_severity(args):
    """
        Sets args.severity to the current severity and returns True on success.
        For the KITTI dataset this will get the appropriate severity for the
        current task. In case of different number of severities among tasks,
        False is returned if current args.severity_idx does not exist for the
        current task.
    """
    if args.dataset != 'kitti':
        args.severity = args.robustness_severities[args.severity_idx]
        return True

    if args.task == 'initial':
        args.severity = '' # TODO not tested thoroughly
        return True

    if args.severity_idx < len(globals.KITTI_SEVERITIES[args.task]):
        args.severity = globals.KITTI_SEVERITIES[args.task][args.severity_idx]
        return True

    return False


def get_all_severities_str(args):
    all_severities_str = ''
    for task in globals.TASKS:
        if args.dataset != 'kitti':
            all_severities_str = f'{args.robustness_severities[args.severity_idx]}_'
            break
        elif args.severity_idx < len(globals.KITTI_SEVERITIES[task]):
            all_severities_str += f'{globals.KITTI_SEVERITIES[task][args.severity_idx]}_'
    return all_severities_str


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers
    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def dataset_checks(args):
    if not args.dataset in config.VALID_DATASETS:
        raise Exception(f'Invalid dataset argument: {args.dataset}')

    error = False
    if args.dataset == 'cifar10':
        error = check_cifar10_c(args)
    elif args.dataset in ['imagenet', 'imagenet-mini']:
        error = check_imgnet_c(args)

    if error:
        raise Exception('Dataset checks unsuccessful!')
    else:
        log.info('Dataset checks successful!')


def check_cifar10_c(args):
    CIFAR10(root=args.dataroot, download=True)
    error = False
    test_set_path = join(args.dataroot, 'CIFAR-10-C', 'test')
    train_set_path = join(args.dataroot, 'CIFAR-10-C', 'train')
    if not exists(test_set_path):
        error = True
        log.error(f'CIFAR-10-C test set not found. Expected at {test_set_path}')
    if not exists(train_set_path):
        error = True
        log.error(f'CIFAR-10-C training set not found. Expected at {train_set_path}')
    missing_files = []
    for task in globals.TASKS:
        test_samples = join(test_set_path, task + '.npy')
        train_samples = join(train_set_path, task + '.npy')
        if not exists(test_samples):
            missing_files.append(test_samples)
        if not exists(train_samples):
            missing_files[:0] = [train_samples]
    if len(missing_files):
        error = True
        log.error('Missing the following CIFAR-10-C samples:')
        for f_path in missing_files:
            log.error(normpath(f_path))
    return error


def check_imgnet_c(args):
    error = False
    val_set_path = join(args.dataroot, args.dataset + '-c', 'val')
    train_set_path = join(args.dataroot, args.dataset + '-c', 'train')

    if not exists(val_set_path):
        error = True
        log.error(f'{args.dataset.capitalize()} validation set not found. '
                  f'Expected at {val_set_path}')
    if not exists(train_set_path):
        error = True
        log.error(f'{args.dataset.capitalize()} training set not found. '
                  f'Expected at {train_set_path}')
    missing_dirs = []
    for task in globals.TASKS:
        for severity in globals.SEVERTITIES:
            val_samples_dir = join(val_set_path, task, str(severity))
            train_samples_dir = join(train_set_path, task, str(severity))
            if not exists(val_samples_dir):
                missing_dirs.append(val_samples_dir)
            if not exists(train_samples_dir):
                missing_dirs[:0] = [train_samples_dir]
    if len(missing_dirs):
        error = True
        log.error(f'Missing the following {args.dataset.capitalize()} directories:')
        for f_path in missing_dirs:
            log.error(normpath(f_path))
    return error

