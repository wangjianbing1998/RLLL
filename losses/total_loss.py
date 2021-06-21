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
from typing import Union

import torch
from torch import nn

from losses import BaseLoss
from losses.base_loss import KDLoss
from utils.util import MultiOutput


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

    @staticmethod
    def default_value(opt):
        return opt

    def __call__(self, preds: Union[MultiOutput, torch.Tensor],
                 gts: Union[MultiOutput, torch.Tensor], task_index=None) -> 'loss Tensor':
        """calculate the losses_without_lambda on multi-task or single task
        loss on multi-task: preds:MultiOutput, gts:MultiOutput, losses_without_lambda and losses_with_lambda will be calculated, Return loss_total
        loss on single-task: preds:scaler, gts:scaler, losses_without_lambda and losses_with_lambda will be None, Return loss_total(eg. ce_loss)

        Debug output the summary on <losses_with_lambda>, <losses_with_lambda>, <losses_without_lambda>

         Return
         if plus_other_loss, then return summary on <losses_with_lambda>
         else, then return <cross_entropy_loss> on current task

         """
        if isinstance(preds, MultiOutput) and isinstance(gts, MultiOutput):
            """get <MultiOutput> loss"""
            task_preds, task_gts = lambda x:(preds[x]), lambda x:(gts[x])
            self.losses_without_lambda = []
            for i in range(self.continued_task_index + 1):
                prediction = task_preds(i)
                target = task_gts(i)
                if i != task_index:
                    if self._plus_other_loss:
                        loss = self.kd_loss(prediction, target)
                    else:
                        continue
                else:
                    # if judge_tensor_value_is_long(target): # target is the ground truth
                    if len(target.size()) == 1:  # target is the ground truth
                        loss = self.ce_loss(prediction, target).cuda()
                    else:  # target is the model(xs), prediction ~= target
                        loss = self.kd_loss(prediction, target).cuda()

                self.losses_without_lambda.append(loss)

            self.losses_with_lambda = self.losses_without_lambda  # TODO implement the losses_with_lambda
            self.loss_total = sum(self.losses_with_lambda)
            return self.loss_total, self.losses_without_lambda

        else:
            """get current task ce-loss"""
            self.loss_need_backward_indices = []
            self.losses_without_lambda = None
            self.losses_with_lambda = None
            self.loss_total = self.ce_loss(preds, gts)
            return self.loss_total
