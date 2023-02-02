VALID_DATASETS = ['cifar10', 'imagenet', 'kitti', 'imagenet-mini']

ROBUSTNESS_TASKS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

KITTI_TASKS = ['fog', 'rain']

PATHS = {
    'jm': {
        'cifar10': {
            'root': '/PATH/TO/cifar10/ROOT',
            'ckpt': '/PATH/TO/cifar10/CHECKPOINT',
        },
        'imagenet-mini': {
            'root': '/PATH/TO/imagenet-mini/ROOT',
            'ckpt': '/PATH/TO/imagenet-mini/CHECKPOINT',
        },
        'imagenet': {
            'root': '/PATH/TO/imagenet/ROOT',
            'ckpt': '/PATH/TO/imagenet/CHECKPOINT',
        },
        'kitti': {
            'root': '/PATH/TO/kitti/ROOT',
            'ckpt': '/PATH/TO/kitti-clear/CHECKPOINT',
        },
    },
}

LOGGER_CFG = {
    'version': 1,
    'formatters': {
        'default': {
            'format': '[%(name)s - %(levelname)s] %(message)s'
        },
        'timestamped': {
            'format': '%(asctime)s [%(name)s - %(levelname)s] %(message)s'
        },
        'minimal': {
            'format': '[%(name)s] %(message)s'
        }
    },
    'filters': {
        'name': {
            '()': 'config.ContextFilter'
        }
    },
    'handlers': {
        'console_handler': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'minimal',
            'stream': 'ext://sys.stdout',
            'filters': ['name']
        },
        'file_handler': {
            'level': 'DEBUG',
            'formatter': 'minimal',
            'class': 'logging.FileHandler',
            'filename': 'log.txt',
            'mode': 'a',
            'filters': ['name']
        },
    },
    'loggers': {
        '': {
            'handlers': ['console_handler', 'file_handler'],
            'level': 'WARNING',
            'propagate': False
        },

        'MAIN': {
            'handlers': ['console_handler', 'file_handler'],
            'level': 'DEBUG',
            'propagate': False
        },
        'MAIN.DISC': {},
        'MAIN.DUA': {},
        'MAIN.DATA': {},
        'MAIN.RESULTS': {},

        'BASELINE': {
            'handlers': ['console_handler', 'file_handler'],
            'level': 'DEBUG',
            'propagate': False
        },
        'BASELINE.FREEZING': {},
        'BASELINE.DISJOINT': {},
        'BASELINE.JOINT_TRAINING': {},
        'BASELINE.SOURCE_ONLY': {},
        'BASELINE.FINE_TUNING': {},

        'TRAINING': {
            'handlers': ['console_handler', 'file_handler'],
            'level': 'DEBUG',
            'propagate': False
        },

        'TESTING': {
            'handlers': ['console_handler', 'file_handler'],
            'level': 'DEBUG',
            'propagate': False
        },
        'TESTING.FILEONLY': {
            'handlers': ['file_handler'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}

# Filtering logger tag prefixes
class ContextFilter:
    def filter(self, record):
        split_name = record.name.split('.', 1)
        if split_name[0] == 'BASELINE' or split_name[0] == 'MAIN':
            if len(split_name) > 1:
                record.name = split_name[1]
        if split_name[0] == 'TESTING':
            if len(split_name) > 1:
                record.name = split_name[0]
        return True


YOLO_HYP = {
    # !! lr0 will be overwritten by args.lr !!
    'lr0': 0.01,            # initial learning rate (SGD=1E-2, Adam=1E-3)
    'lrf': 0.2,             # final OneCycleLR learning rate (lr0 * lrf)
    'momentum': 0.937,      # SGD momentum/Adam beta1
    'weight_decay': 0.0005, # optimizer weight decay 5e-4
    'warmup_epochs': 3.0,   # warmup epochs (fractions ok)
    'warmup_momentum': 0.8, # warmup initial momentum
    'warmup_bias_lr': 0.1,  # warmup initial bias lr
    'box': 0.05,            # box loss gain
    'cls': 0.5,             # cls loss gain
    'cls_pw': 1.0,          # cls BCELoss positive_weight
    'obj': 1.0,             # obj loss gain (scale with pixels)
    'obj_pw': 1.0,          # obj BCELoss positive_weight
    'iou_t': 0.20,          # IoU training threshold
    'anchor_t': 4.0,        # anchor-multiple threshold
    'fl_gamma': 0.0,        # focal loss gamma (efficientDet default gamma=1.5)
    'hsv_h': 0.015,         # image HSV-Hue augmentation (fraction)
    'hsv_s': 0.7,           # image HSV-Saturation augmentation (fraction)
    'hsv_v': 0.4,           # image HSV-Value augmentation (fraction)
    'degrees': 0.0,         # image rotation (+/- deg)
    'translate': 0.1,       # image translation (+/- fraction)
    'scale': 0.5,           # image scale (+/- gain)
    'shear': 0.0,           # image shear (+/- deg)
    'perspective': 0.0,     # image perspective (+/- fraction), range 0-0.001
    'flipud': 0.0,          # image flip up-down (probability)
    'fliplr': 0.5,          # image flip left-right (probability)
    'mosaic': 1.0,          # image mosaic (probability)
    'mixup': 0.0            # image mixup (probability)
}


