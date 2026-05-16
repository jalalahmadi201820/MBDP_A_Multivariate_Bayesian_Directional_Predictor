# -*- coding: utf-8 -*-
"""
Model utility functions
"""

import torch
import torch.nn as nn
import random
import numpy as np
import os
import copy


def init_seed(seed):
    """Disable cudnn to maximize reproducibility"""
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def init_device(opt):
    """Initialize device (CPU or GPU)"""
    if torch.cuda.is_available():
        opt.cuda = True
        torch.cuda.set_device(int(opt.device[5]))
    else:
        opt.cuda = False
        opt.device = 'cpu'
    return opt


def init_optim(model, opt):
    """Initialize optimizer"""
    return torch.optim.Adam(params=model.parameters(), lr=opt.lr_init)


def init_lr_scheduler(optim, opt):
    """Initialize the learning rate scheduler"""
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optim, 
        milestones=opt.lr_decay_steps,
        gamma=opt.lr_scheduler_rate
    )


def print_model_parameters(model, only_num=True):
    """Print model parameters"""
    print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('*****************Finish Parameter****************')


def get_memory_usage(device):
    """Get GPU memory usage"""
    allocated_memory = torch.cuda.memory_allocated(device) / (1024*1024.)
    cached_memory = torch.cuda.memory_cached(device) / (1024*1024.)
    return allocated_memory, cached_memory


def save_model(model, model_dir, epoch=None):
    """Save model to file"""
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ""
    file_name = os.path.join(model_dir, epoch + "_samba.pt")
    with open(file_name, "wb") as f:
        torch.save(model, f)
