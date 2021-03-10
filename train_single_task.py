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

from sklearn.utils import Bunch

from models import create_model, BaseModel
from options.train_options import TrainOptions
from task_datasets import create_task_dataset, PseudoData
from util.util import MatrixItem
from util.visualizer import Visualizer


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


def train(opt, model, task_index, continued_task_index, train_dataset, val_dataset=None, visualizer=None):
    model.continued_task_index = continued_task_index
    logging.info(f"Fitting task {task_index}")
    best_matrix_item = None
    for epoch in range(opt.epoch_start, opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs;
        epoch_start_time = time.time()  # timer for entire epoch
        total_loss = 0
        n_batch = 0
        for data in train_dataset:  # inner loop within one epoch
            logging.debug(f'Loading dataset {data["data_name"]}, target={data["data"]["target"]}')
            previous_data: 'image,SingleOutput' = PseudoData(opt, Bunch(**data["data"]))
            model.set_data(previous_data)
            model.test(visualizer)
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


if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    visualizer = Visualizer(opt)
    visualizer.setup()  # regular setup:

    # visualizer.add_graph(model,) TODO the model graph visualization

    train_datasets = create_task_dataset(opt, phase="train")
    val_datasets = create_task_dataset(opt, phase="val")
    test_datasets = create_task_dataset(opt, phase="test")

    nb_tasks = train_datasets.nb_tasks

    task_index = 0
    task_dataset = train_datasets[task_index]
    val_dataset = val_datasets[task_index]
    model.setup(opt)  # regular setup: load and print networks; create schedulers before training each task
    train(opt,
          model=model,
          task_index=task_index,
          continued_task_index=task_index,
          train_dataset=task_dataset,
          val_dataset=val_dataset,
          visualizer=visualizer,
          )
