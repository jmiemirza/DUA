import os
import torch
from colorama import Fore
import numpy as np


def get_grad(params):
    if isinstance(params, torch.Tensor):
        params = [params]
    params = list(filter(lambda p: p.grad is not None, params))
    grad = [p.grad.data.cpu().view(-1) for p in params]
    return torch.cat(grad)


def write_to_txt(name, content):
    with open(name, 'w') as text_file:
        text_file.write(content)


def my_makedir(name):
    try:
        os.makedirs(name)
    except OSError:
        pass


def print_args(opt):
    for arg in vars(opt):
        print('%s %s' % (arg, getattr(opt, arg)))


def mean(ls):
    return sum(ls) / len(ls)


def normalize(v):
    return (v - v.mean()) / v.std()


def flat_grad(grad_tuple):
    return torch.cat([p.view(-1) for p in grad_tuple])


def print_nparams(model):
    nparams = sum([param.nelement() for param in model.parameters()])
    print('number of parameters: %d' % (nparams))


def print_color(color, string):
    print(getattr(Fore, color) + string + Fore.RESET)


def plot_adaptation_err(all_err_cls, corr, args):
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    fig, _ = plt.subplots()

    plt.plot(all_err_cls, color='r', label=corr)
    plt.xlabel('Number of Samples for Adaptation')
    plt.ylabel('Test Error (%)')
    plt.legend()
    plt.savefig(os.path.join(args.outf, corr), format="png")
    plt.close(fig)
