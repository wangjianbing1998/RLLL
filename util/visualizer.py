# encoding=utf-8
'''
@File    :   visualizer.py    
@Contact :   jianbingxiaman@gmail.com
@License :   (C)Copyright 2020-2021, John Hopcraft Lab-CV
@Desciption : 
@Modify Time      @Author    @Version
------------      -------    --------
2021/1/24 17:14   jianbingxia     1.0    
'''
import os

import torch
from tensorboardX import SummaryWriter

from util.util import is_gpu_avaliable, rmdirs


class Visualizer(object):
    def __init__(self, opt):
        self.opt = opt
        self.logs_dir = opt.logs_dir
        self.model_name = opt.model_name

        self.summary_writer = SummaryWriter(log_dir=self.logs_dir, comment=self.model_name)

    def setup(self):
        # TODO regular setup: if opt.<continue_train>, add_scalar into the previous scalars.

        pass

    def add_scalar(self, *args):
        self.summary_writer.add_scalar(*args)

    def add_losses(self, losses, epoch):
        """add losses for FloatTensor or list of FloatTensor

        such as
        losses_with_lambda: [losses_with_lambda[index] for index in range(nb_tasks)]
        losses_without_lambda: [losses_without_lambda[index] for index in range(nb_tasks)]
        loss_total: FloatTensor

        """
        for loss_name, loss in losses.items():
            if isinstance(loss, torch.FloatTensor if not is_gpu_avaliable(self.opt) else torch.cuda.FloatTensor):
                self.summary_writer.add_scalar("losses", loss, epoch)
            elif isinstance(loss, list):
                loss = dict([(str(index), l) for index, l in enumerate(loss)])
                self.summary_writer.add_scalars("losses", loss)
            else:
                raise TypeError(f"loss must be float or list, but got {type(loss)}")

    def add_scalars(self, *args):
        self.summary_writer.add_scalars(*args)

    def add_graph(self, *args):
        self.summary_writer.add_graph(*args)

    def reset(self):
        pass

    def __del__(self):
        self.close()

    def close(self):
        self.summary_writer.close()
