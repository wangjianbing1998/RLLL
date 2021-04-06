# encoding=utf-8
'''
@File    :   __init__.py.py    
@Contact :   jianbingxiaman@gmail.com
@License :   (C)Copyright 2020-2021, John Hopcraft Lab-CV
@Desciption : 
@Modify Time      @Author    @Version
------------      -------    --------
2021/1/23 10:46   jianbingxia     1.0    
'''
import functools
import logging

from torch import nn
from torch.nn import init
from torch.optim import lr_scheduler

"""This package contains networks related to objective functions, and network architectures.

To add a custom net_main class called 'dummy', you need to add a file called 'dummy_net.py' and define a subclass DummyNet inherited from BaseNet.
You need to implement the following five functions:
    -- <__init__>:                      initialize the class; first call BaseNet.__init__(self, opt).
    -- <forward>:                       forward the network
    

In the function <__init__>, you need to define one list:
    -- self.net_names (str list):         define networks used in our training.

In the function <forward>, you need to define the forwarding network:
    -- multi_output_classifier (MultiOutputClassifier): the output of network
Now you can use the net_main class by specifying flag '--net_name dummy1 in dummy2_model.py'.
See our template net_main class 'alexnet_net.py' for more details.
"""

import importlib

from networks.base_net import BaseNet

net_names = []


def find_net_using_name(net_name):
    """Import the module "networks/[net_name]_net.py".

    In the file, the class called DatasetNameNet() will
    be instantiated. It has to be a subclass of BaseNet,
    and it is case-insensitive.
    """
    net_filename = "networks." + net_name + "_net"
    netlib = importlib.import_module(net_filename)
    net = None
    target_net_name = net_name.replace('_', '') + 'net'
    for name, cls in netlib.__dict__.items():
        if 'net' in name.lower():
            net_names.append(name.lower())
            if name.lower() == target_net_name.lower():
                net = cls

    if net is None:
        logging.error(
            "In %s.py, there should be a subclass of BaseModel with class tag that matches %s in lowercase." % (
                net_filename, target_net_name))
        exit(0)

    return net


def get_option_setter(net_name):
    """Return the static method <modify_commandline_options> of the net_main class."""
    net_class = find_net_using_name(net_name)
    return net_class.modify_commandline_options


def create_net(opt):
    """Create a net_main given the option.

    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from networks import create_net
        >>> net_main = create_net(opt)
    """
    net = find_net_using_name(opt.net_name)
    instance = net(opt)
    logging.info("net [%s] was created" % type(instance).__name__)
    return instance



class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the tag of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the tag of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_start - opt.n_epochs) / float(opt.n_epochs_decay + 2)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def _init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the tag of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    logging.info('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
