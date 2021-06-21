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
import random

random.seed(42)

import pandas as pd

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 20)
pd.set_option('precision', 2)

import matplotlib.pyplot as plt
import torch
from tensorboardX import SummaryWriter

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.max_open_warning'] = 50


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
        from utils.util import is_gpu_avaliable
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


class MetrixResult(object):
    def __init__(self, df, shift_column=True):
        self.df = df
        self.T = len(df) - 1
        self.shift = 0
        if shift_column:
            self.shift = 1

        self._backward_transfer = self.__backward_transfer()
        self._farward_transfer = self.__farward_transfer()
        self._average_accuracy = self.__average_accuracy()

    @property
    def backward_transfer(self):
        return self._backward_transfer

    @property
    def farward_transfer(self):
        return self._farward_transfer

    @property
    def average_accuracy(self):
        return self._average_accuracy

    def __repr__(self):
        return f'bt={self.backward_transfer}, ft={self.farward_transfer}, aa={self.average_accuracy}'

    def __getitem__(self, item):
        assert len(item) == 2, f'Expected len(item)==2, but got {item}'
        row_index, col_index = item
        col_index -= self.shift
        return self.df.iloc[(row_index, col_index)]

    def __backward_transfer(self):
        """

		>>> result=MetrixResult(pd.DataFrame({0:[0,1,2,3,4],1:[10,11,12,13,14],2:[20,21,22,23,24],3:[30,31,32,33,34],4:[40,41,42,43,44]}))
		>>> result.backward_transfer()
		2.0

		"""
        ans = [self[(self.T, i)] - self[(i, i)] for i in range(1, self.T)]
        return sum(ans) / len(ans)

    def __farward_transfer(self):
        """

		>>> result=MetrixResult(pd.DataFrame({0:[0,1,2,3,4],1:[10,11,12,13,14],2:[20,21,22,23,24],3:[30,31,32,33,34],4:[40,41,42,43,44]}))
		>>> result.farward_transfer()
		2.0

		"""
        ans = [self[(i - 1, i)] - self[(0, i)] for i in range(2, self.T + 1)]
        return sum(ans) / len(ans)

    def __average_accuracy(self):
        """

		>>> result=MetrixResult(pd.DataFrame({0:[0,1,2,3,4],1:[10,11,12,13,14],2:[20,21,22,23,24],3:[30,31,32,33,34],4:[40,41,42,43,44]}))
		>>> result.average_accuracy()
		29.0
		"""
        ans = [self[(self.T, i)] for i in range(1, self.T + 1)]
        return sum(ans) / len(ans)
