# encoding=utf-8
'''
@File    :   train_single_task.py    
@Contact :   jianbingxiaman@gmail.com
@License :   (C)Copyright 2020-2021, John Hopcraft Lab-CV
@Desciption : 
@Modify Time      @Author    @Version
------------      -------    --------
2021/3/6 10:44   jianbingxia     1.0    
'''
import logging
import time
from typing import Tuple, List

import torch
from sklearn.utils import Bunch
from torchvision.models import alexnet

from models import BaseModel, create_model
from options.train_options import TrainOptions
from task_datasets import create_task_dataset, PseudoData
from util.util import MatrixItem, MultiOutput
from util.visualizer import Visualizer


class AlexnetNet(torch.nn.Module):
    def __init__(self, opt):
        super(AlexnetNet, self).__init__()
        base_alexnet = alexnet(pretrained=opt.pretrained)
        self.shared_cnn_layers = base_alexnet.features
        self.adap_avg_pool = base_alexnet.avgpool
        self.shared_fc_layers = base_alexnet.classifier[:6]
        self.in_features = base_alexnet.classifier[6].in_features

        self.target_outputs = torch.nn.ModuleList(
            [torch.nn.Linear(self.in_features, num_class) for num_class in [10, 10, 100]])

    def forward(self, _input):
        cnn_out = self.shared_cnn_layers(_input)
        cnn_out = self.adap_avg_pool(cnn_out)
        cnn_out_flatten = cnn_out.contiguous().view(cnn_out.size()[0], -1)
        shared_fc_out = self.shared_fc_layers(cnn_out_flatten)
        return self.target_outputs[0](shared_fc_out)


class Model():

    def __init__(self, opt):
        self.net_main = AlexnetNet(opt)
        self.net_main.cuda()
        self.optimizer = torch.optim.SGD(self.net_main.parameters(), lr=opt.lr)
        self.loss_criterion = torch.nn.CrossEntropyLoss()

    def set_data(self, data):
        self.image = data.image.cuda()
        self.target = data.target.cuda()

    def eval(self):
        self.net_main.eval()

    def fit(self):
        self.net_main.train()

    def forward(self):
        self.output = self.net_main(self.image)

    def optimize_parameters(self):
        self.optimizer.zero_grad()  # clear network's existing gradients
        self.loss = self.loss_criterion(self.output, self.target.squeeze())
        self.optimizer.step()  # update gradients for network

    def train(self):
        self.fit()
        self.forward()
        self.optimize_parameters()

    def test(self):
        self.eval()
        with torch.no_grad():
            self.forward()

    def get_current_losses(self):
        return {"loss_total":self.loss}

    def get_matrix_item(self):
        return MatrixItem(self.output, self.target, self.loss_criterion)


def val(val_dataset: 'Single task_dataset', model: BaseModel, task_index, visualizer=None) -> Tuple[MatrixItem, List]:
    """for validation on one task"""
    logging.info(f"Validating task {task_index}")
    start_time = time.time()  # timer for validate a task

    matrixItems = []
    for i, data in enumerate(val_dataset):  # inner loop within one epoch
        model.set_data(PseudoData(opt, Bunch(**data["data"])))
        model.test(visualizer)
        # Add matrixItem result
        matrixItems.append(model.get_matrix_item(task_index))

    res = sum(matrixItems, MatrixItem()(accuracy=0, loss=0))
    res = res / len(matrixItems)
    logging.info(f"Validation Time Taken: {time.time() - start_time} sec")
    return res, matrixItems


def train(opt, model, task_index, continued_task_index, train_dataset, val_dataset=None):
    model.continued_task_index = continued_task_index
    logging.info(f"Fitting task {task_index}")
    best_matrix_item = None
    for epoch in range(opt.epoch_start, opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs;
        epoch_start_time = time.time()  # timer for entire epoch
        total_loss = 0
        n_batch = 0
        for data in train_dataset:  # inner loop within one epoch
            logging.debug(f'Loading dataset {data["data_name"]}, target={data["data"]["target"]}')
            previous_data: 'image,SingleOutput' = PseudoData(opt, data=Bunch(**data["data"]))
            model.set_data(previous_data)
            model.test()
            data: 'image,MultiOutput' = PseudoData(opt, previous_data, model.output, task_index)  #
            model.set_data(data)

            assert all(data.target[task_index] == previous_data.target)

            model.train(task_index)

            losses = model.get_current_losses()
            total_loss += losses['loss_total']
            n_batch += 1
        total_loss /= n_batch
        if epoch % opt.curve_freq == 0:  # visualizing training losses and save logging information to the disk
            visualizer.add_losses({'loss_total':total_loss}, epoch)
        # Validation
        val_matrix, val_matrix_items = val(val_dataset, model, task_index, visualizer)

        if (epoch + 1) % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            logging.info('saving the model at the end of epoch %d' % (epoch))
            model.save_networks(continued_task_index, epoch)

        if opt.save_best and (best_matrix_item is None or val_matrix > best_matrix_item):
            logging.info(f'saving the best model at the end of epoch {epoch}')
            model.save_networks(continued_task_index, epoch="best")

        logging.info(
            f'End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t train_loss={total_loss.item()},val:{val_matrix}, Time Taken: {time.time() - epoch_start_time} sec')
        model.update_learning_rate()  # update learning rates at the end of every epoch.


def main(_):
    # new model
    opt = TrainOptions().parse()  # get training options
    model = Model(opt)  # create a model given opt.model and other options

    train_datasets = create_task_dataset(opt, phase="train")
    val_datasets = create_task_dataset(opt, phase="val")

    task_index = 0
    train_dataset = train_datasets[task_index]
    val_dataset = val_datasets[task_index]
    logging.info(f"Fitting task {task_index}")
    for epoch in range(10):  # outer loop for different epochs;
        epoch_start_time = time.time()  # timer for entire epoch
        total_loss = 0
        n_batch = 0
        for data in train_dataset:  # inner loop within one epoch
            logging.debug(f'Loading dataset {data["data_name"]}, target={data["data"]["target"]}')
            model.set_data(Bunch(**data["data"]))
            model.train()
            losses = model.get_current_losses()
            total_loss += losses['loss_total']
            n_batch += 1
        total_loss /= n_batch

        # Validation
        matrixItems = []
        for data in val_dataset:
            model.set_data(Bunch(**data["data"]))
            model.test()
            matrixItems.append(model.get_matrix_item())

        val_matrix = sum(matrixItems, MatrixItem()(accuracy=0, loss=0))
        val_matrix = val_matrix / len(matrixItems)

        # logging output
        logging.info(
            f'End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t train_loss={total_loss.item()},val:{val_matrix}, Time Taken: {time.time() - epoch_start_time} sec')


if __name__ == '__main__':
    device = 'cuda:0'
    # main("")

    opt = TrainOptions().parse()  # get training options

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(0)  # regular setup: load and print networks; create schedulers

    visualizer = Visualizer(opt)
    visualizer.setup()  # regular setup:

    train_datasets = create_task_dataset(opt, phase="train")
    val_datasets = create_task_dataset(opt, phase="val")
    test_datasets = create_task_dataset(opt, phase="test")

    nb_tasks = train_datasets.nb_tasks

    task_index = 0
    task_dataset = train_datasets[task_index]
    val_dataset = val_datasets[task_index]
    train(opt,
          model=model,
          task_index=task_index,
          continued_task_index=task_index,
          train_dataset=task_dataset,
          val_dataset=val_dataset,
          )
