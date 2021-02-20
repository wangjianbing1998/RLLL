# encoding=utf-8
'''
@File    :   __init__.py.py    
@Contact :   jianbingxiaman@gmail.com
@License :   (C)Copyright 2020-2021, John Hopcraft Lab-CV
@Desciption : 
@Modify Time      @Author    @Version
------------      -------    --------
2021/1/23 12:37   jianbingxia     1.0    
'''
import logging

from losses.base_loss import BaseLoss

"""This package contains losses related to objective functions, optimizations, and network architectures.

To add a custom model class called 'dummy', you need to add a file called 'dummy_loss.py' and define a subclass DummyLoss inherited from BaseLoss.
You need to implement the following five functions:
    -- <__init__>:                      initialize the class; first call BaseLoss.__init__(self, opt).
    -- self.<loss_names>:               set the current class's loss_names, to propose into the naiv model.
    -- <__call__>:                      calculate loss in predictions and ground-truth, kd-loss for other task, ce-loss for current task.
    -- <modify_commandline_options>:    (optionally) add loss-specific options and set default options.

In the function <__init__>, you need to define four lists:
    -- self.loss_names (str list):    define the name of losses need to visualize and plot

Now you can use the model class by specifying flag '--loss_name dummy1 in dummy2_model.py'.
See our template model class 'total_loss.py' for more details.
"""

import importlib

loss_names = []


def find_loss_using_name(loss_name):
    """Import the module "losses/[loss_name]_loss.py".

    In the file, the class called LossNameLoss() will
    be instantiated. It has to be a subclass of BaseLoss,
    and it is case-insensitive.
    """
    losslib = importlib.import_module("losses." + loss_name + "_loss")
    loss = None
    target_loss_name = loss_name.replace('_', '') + 'loss'
    for name, cls in losslib.__dict__.items():
        if 'loss' in name.lower():
            loss_names.append(name.lower())
            if name.lower() == target_loss_name.lower():
                loss = cls

    if loss is None:
        logging.error(
            "In %s.py, there should be a subclass of BaseLoss with class tag that matches %s in lowercase." % (
                ("losses." + loss_name + "_loss"), target_loss_name))
        exit(0)

    return loss


def get_option_setter(loss_name):
    """Return the static method <modify_commandline_options> of the loss class."""
    return find_loss_using_name(loss_name).modify_commandline_options


def create_loss(opt):
    """Create a model given the option.

    This function warps the class DannyLoss.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from  losses import create_loss
        >>> loss = create_loss(opt)
    """
    loss = find_loss_using_name(opt.loss_name)
    instance = loss(opt)
    logging.info(f"loss [{type(instance).__name__}] was created")
    return instance
