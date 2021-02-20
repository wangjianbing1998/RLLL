# encoding=utf-8
'''
@File    :   base_loss.py    
@Contact :   jianbingxiaman@gmail.com
@License :   (C)Copyright 2020-2021, John Hopcraft Lab-CV
@Desciption : 
@Modify Time      @Author    @Version
------------      -------    --------
2021/1/23 12:38   jianbingxia     1.0    
'''
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from util.util import MultiOutput

__all__ = ['BaseLoss', 'KDLoss']


class BaseLoss(ABC):
    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.loss_names = []

    @staticmethod
    def modify_commandline_options(parser, isTrain):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    def get_current_losses(self):
        """Return traning losses_without_lambda / errors. fit.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = getattr(self, name)
        return errors_ret

    @property
    def continued_task_index(self):
        return self._continued_task_index

    @continued_task_index.setter
    def continued_task_index(self, value):
        self._continued_task_index = value

    @property
    def plus_other_loss(self):
        return self._plus_other_loss

    @plus_other_loss.setter
    def plus_other_loss(self, value):
        self._plus_other_loss = value

    @abstractmethod
    def __call__(self, preds: MultiOutput, gts: MultiOutput, task_index):
        pass


class KDLoss():
    """KD loss between """

    def __init__(self, temp=2):
        self.temp = temp
        self.log_sotfmax = nn.LogSoftmax(dim=-1)

    def __call__(self, preds, gts):
        # logging.debug("preds:" + str(preds))
        # logging.debug("gts:" + str(gts))
        preds = F.softmax(preds, dim=-1)
        preds = torch.pow(preds, 1. / self.temp)
        # l_preds = F.softmax(preds, dim=-1)
        l_preds = self.log_sotfmax(preds)

        gts = F.softmax(gts, dim=-1)
        gts = torch.pow(gts, 1. / self.temp)
        gts = F.softmax(gts, dim=-1)
        # l_gts = self.log_sotfmax(gts)

        # l_preds = torch.log(l_preds)
        # l_preds[l_preds != l_preds] = 0.
        loss = torch.mean(torch.sum(-gts * l_preds, dim=(1,)))
        return loss
