from tqdm import tqdm
from utils.data_loader import *
from utils.rotation import *
from utils.testing import test
from utils.testing_yolov3 import test as test_yolo
from utils.torch_utils import select_device
from utils.utils import make_dirs
from utils.results_manager import ResultsManager
from init import init_net
log = logging.getLogger('MAIN.DUA')


def dua(args, net, save_bn_stats=False, use_training_data=False, save_fname=None):
    results_mgr = ResultsManager()
    if args.model == 'yolov3':
        get_adaption_inputs = get_adaption_inputs_kitti
        metric = 'mAP@50'
        tr_transform_adapt = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((224, 640)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        get_adaption_inputs = get_adaption_inputs_default
        metric = 'Error'
        tr_transform_adapt = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*NORM)
        ])
    if not args.dataset == 'imagenet':
        ckpt = torch.load(args.ckpt_path)
    decay_factor = args.decay_factor
    min_momentum_constant = args.min_mom
    no_imp = 0
    no_imp_cnt = 0
    all_results = []
    device = select_device(args.device, batch_size=args.batch_size)

    for args.task in globals.TASKS:
        if not set_severity(args):
            continue
        mom_pre = 0.1
        results = []
        log.info(f'Task - {args.task} :::: Level - {args.severity}')
        if not args.dataset == 'imagenet':
            net.load_state_dict(ckpt)
        else:
            init_net(args)

        net.eval()
        if use_training_data:
            train_loader = get_loader(args, split='train')
            valid_loader = get_loader(args, split='val', pad=0.5, rect=True)
        else:
            # original DUA is run on test data only
            train_loader = valid_loader = get_loader(args, split='test', pad=0.5, rect=True)

        if args.model == 'yolov3':
            res = test_yolo(model=net, dataloader=valid_loader,
                            iou_thres=args.iou_thres, conf_thres=args.conf_thres,
                            augment=args.augment)[0] * 100
        else:
            res = test(valid_loader, net)[0] * 100
        log.info(f'{metric} Before Adaptation: {res:.1f}')

        for i in tqdm(range(1, args.num_samples + 1)):
            net.eval()
            image = train_loader.dataset.get_image_from_idx(i - 1)
            mom_new = (mom_pre * decay_factor)
            for m in net.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.train()
                    m.momentum = mom_new + min_momentum_constant
            mom_pre = mom_new
            inputs = get_adaption_inputs(image, tr_transform_adapt, device)
            _ = net(inputs)
            net.eval()
            if args.model == 'yolov3':
                res = test_yolo(model=net, dataloader=valid_loader,
                                iou_thres=args.iou_thres, conf_thres=args.conf_thres,
                                augment=args.augment)[0] * 100
            else:
                res = test(valid_loader, net)[0] * 100
            results.append(res)
            if result_improved(metric, res, results):
                save_bn_stats_in_model(net, args.task)
                no_imp = 0
                no_imp_cnt = 0
            else:
                no_imp += 1
                if no_imp >= 10:
                    no_imp_cnt += no_imp
                    no_imp = 0
                    log.info(f'Iteration {i}/{args.num_samples}: No Improvement '
                             f'for {no_imp_cnt} consecutive iterations')

        adaptation_result = max(results) if metric == 'mAP@50' else min(results)

        severity_str = '' if args.task == 'initial' else f'{args.severity}'
        results_mgr.add_result('DUA', f'{args.task} {severity_str}', adaptation_result, 'online')

        log.info(f'{metric} After Adaptation: {adaptation_result:.1f}')
        all_results.append(adaptation_result)
    log.info(f'Mean {metric} after Adaptation {(sum(all_results) / len(all_results)):.1f}')

    if save_bn_stats:
        save_bn_stats_to_file(net, args.dataset, args.model, save_fname)


def result_improved(metric, current_result, all_results_for_current_task):
    """
        Check if the result has improved compared to all previous results.
        If metric is 'mAP@50' higher value means better, else
        lower value means better.
    """
    if metric == 'mAP@50':
        return current_result >= max(all_results_for_current_task)
    else:
        return current_result <= min(all_results_for_current_task)


def get_adaption_inputs_default(img, tr_transform_adapt, device):
    inputs = [(tr_transform_adapt(img)) for _ in range(64)]
    inputs = torch.stack(inputs)
    inputs_ssh, _ = rotate_batch(inputs, 'rand')
    inputs_ssh = inputs_ssh.to(device, non_blocking=True)
    return inputs_ssh


def get_adaption_inputs_kitti(img, tr_transform_adapt, device):
    img = img.squeeze(0)
    inputs = [(tr_transform_adapt(img)) for _ in range(64)]
    inputs = torch.stack(inputs)
    inputs_ssh, _ = rotate_batch(inputs, 'rand')
    inputs_ssh = inputs_ssh.to(device, non_blocking=True)
    inputs_ssh /= 255
    return inputs_ssh


def save_bn_stats_in_model(net, task):
    """
        Saves the running estimates of all batch norm layers for a given
        task, in the net.bn_stats attribute.
    """
    state_dict = net.state_dict()
    net.bn_stats[task] = {}
    for layer_name, m in net.named_modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            net.bn_stats[task][layer_name] = {
                'running_mean': state_dict[layer_name + '.running_mean'].detach().clone(),
                'running_var': state_dict[layer_name + '.running_var'].detach().clone()
            }


def save_bn_stats_to_file(net, dataset_str=None, model_str=None, file_name=None):
    """
        Saves net.bn_stats content to a file.
    """
    # ckpt_folder = 'checkpoints/' + dataset_str + '/' + model_str + '/'
    ckpt_folder = join('checkpoints', dataset_str, model_str)
    make_dirs(ckpt_folder)
    if not file_name:
        file_name = 'BN_stats.pt'
    torch.save(net.bn_stats, join(ckpt_folder, file_name))