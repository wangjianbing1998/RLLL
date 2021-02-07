# encoding=utf-8
'''
@File    :   total_loss.py    
@Contact :   jianbingxiaman@gmail.com
@License :   (C)Copyright 2020-2021, John Hopcraft Lab-CV
@Desciption : 
@Modify Time      @Author    @Version
------------      -------    --------
2021/1/23 12:43   jianbingxia     1.0    
'''
import logging

import numpy as np
import torch
from torch import nn

from losses import BaseLoss
from losses.base_loss import KDLoss
from util.util import tensors2str, MultiOutput, un_onehot


class TotalLoss(BaseLoss):
    def __init__(self, opt):
        BaseLoss.__init__(self, opt)

        self.loss_names = ["losses_with_lambda", "losses_without_lambda", "loss_total"]
        self.kd_loss = KDLoss(temp=opt.temp)  # for other loss
        self.ce_loss = nn.CrossEntropyLoss()
        self.nb_tasks = len(opt.num_classes)
        self.lambda_all = opt.lambda_all
        self._plus_other_loss = False  # if plus other kd loss or not
        self._continued_task_index = 0  # currently trained task index
        if self.nb_tasks != len(self.lambda_all):
            raise ValueError(
                f"Expected num_classes = len(lambda_all), but got nb_tasks={self.nb_tasks}, len(lambda_all) = {len(self.lambda_all)}")

    @staticmethod
    def modify_commandline_options(parser):
        """Add new loss-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser


        Returns:
            the modified parser.


        """
        parser.add_argument('--lambda_all', type=list, default=[1., 1., 1.],
                            help='the length must be same to num_classes', )
        parser.add_argument('--temp', type=int, default=2, help='the temp between KD loss', )

        return parser

    def __call__(self, preds: MultiOutput, gts: MultiOutput, task_index):
        """calculate the losses_without_lambda on multi-task

        Debug output the summary on <losses_with_lambda>, <losses_with_lambda>, <losses_without_lambda>

         Return
         if plus_other_loss, then return summary on <losses_with_lambda>
         else, then return <cross_entropy_loss> on current task

         """
        task_preds, task_gts = lambda x:(preds[x]), lambda x:(gts[x])
        self.losses_without_lambda = []
        for i in range(self.continued_task_index + 1):
            if i != task_index:
                if self._plus_other_loss:
                    self.losses_without_lambda.append(self.kd_loss(task_preds(i), task_gts(i)))
                else:
                    self.losses_without_lambda.append(torch.from_numpy(np.array([0])).cuda().float())
            else:
                self.losses_without_lambda.append(self.ce_loss(task_preds(i), un_onehot(task_gts(i))))

        self.losses_with_lambda = [loss * lambda_ for loss, lambda_ in
                                   zip(self.losses_without_lambda, self.lambda_all)]
        self.loss_total = sum(self.losses_with_lambda)
        # logging.debug(
        #     f"total_loss={self.loss_total.item()}, losses_with_lambda={tensors2str(self.losses_with_lambda)}, losses_without_lambda={tensors2str(self.losses_without_lambda)}")

        return self.loss_total
