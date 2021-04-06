# encoding=utf-8
'''
@File    :   fit.py
@Contact :   jianbingxiaman@gmail.com
@License :   (C)Copyright 2020-2021, John Hopcraft Lab-CV
@Desciption : 
@Modify Time      @Author    @Version
------------      -------    --------
2021/1/24 13:48   jianbingxia     1.0    
'''
import logging
import os
from typing import List, Tuple

from sklearn.utils import Bunch

from task_datasets import PseudoData, create_task_dataset
from utils.util import TestMatrix, MatrixItem

"""General-purpose training script for multi-task learning(or LifeLong Learning) translation.

This script works for various models (with option '--model_name': e.g., lwf, finetune, warmtune, hottune, folwf, tblwf) and
different task_datasets (with option '--task_dataset_name': e.g., custom).

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the model, print/save the loss plot.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a lwf model:
        python train.py --model_name lwf --task_dataset_name custom --continue_train --n_epochs 100 --n_epochs_decay 80
    Train a folwf model:
        python train.py --model_name folwf --task_dataset_name custom --continue_train --n_epochs 100 --n_epochs_decay 80
    and so on.

See options/base_options.py and options/train_options.py for more training options.

"""
import time

from models import create_model, BaseModel
from options.train_options import TrainOptions
from utils.visualizer import Visualizer


# @torchsnooper.snoop()
def fit(opt,
        model,
        task_index,
        continued_task_index,  # currently trained task index
        train_dataset,
        val_dataset=None,
        visualizer=None,
        ):
    model.continued_task_index = continued_task_index

    logging.info(f"Fitting task {task_index}")
    best_matrix_item = None
    for epoch in range(opt.epoch_start, opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs;
        epoch_start_time = time.time()  # timer for entire epoch

        total_loss = 0
        n_batch = 0
        for data in train_dataset:  # inner loop within one epoch
            previous_data: 'image,SingleOutput' = PseudoData(opt, Bunch(**data))
            model.set_data(previous_data)
            model.test(visualizer=visualizer)  # Get model.output
            '''
            data.image=data.image
            data.target=[output,output,...,<data.target>,output,output,...]
            '''
            data: 'image,MultiOutput' = PseudoData(opt, previous_data, model.output, task_index)  #

            # logging.debug("after:"+str(data))
            # assert all(data.target[task_index] == previous_data.target)
            # set data and fit
            model.set_data(data)  # unpack _data from dataset and apply preprocessing
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


def train(opt, model, task_index, continued_task_index, train_datasets, val_datasets=None, visualizer=None):
    fit(opt,
        model=model,
        task_index=task_index,
        continued_task_index=continued_task_index,
        train_dataset=train_datasets[continued_task_index],
        val_dataset=val_datasets[continued_task_index],
        visualizer=visualizer)

    if model.need_backward and task_index >= 1:
        """backward training"""
        train(opt,
              model=model,
              task_index=task_index - 1,
              continued_task_index=continued_task_index,
              train_datasets=train_datasets,
              val_datasets=val_datasets,
              visualizer=visualizer,
              )


def test(opt, test_datasets, model: BaseModel, train_index, visualizer=None):
    """test the model on multi-task test_datasets, after training task indexed with <train_index>

    Return
    None
    the global testMatrix will be updated
    """

    for test_index, test_dataset in enumerate(test_datasets):
        matrixItem, _ = val(test_dataset, model, test_index, visualizer, )
        test_matrix[(train_index, test_index + 1)] = matrixItem


def val(val_dataset: 'Single task_dataset', model: BaseModel, task_index, visualizer=None) -> Tuple[MatrixItem, List]:
    """for validation on one task"""
    logging.info(f"Validating task {task_index}")
    start_time = time.time()  # timer for validate a task

    matrixItems = []
    for i, data in enumerate(val_dataset):  # inner loop within one epoch
        model.set_data(PseudoData(opt, Bunch(**data)))
        model.test(visualizer)
        # Add matrixItem result
        matrixItems.append(model.get_matrix_item(task_index))

    res = sum(matrixItems, MatrixItem()(accuracy=0, loss=0))
    res = res / len(matrixItems)
    logging.info(f"Validation Time Taken: {time.time() - start_time} sec")
    return res, matrixItems


if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup()  # regular setup: load and print networks; create schedulers

    visualizer = Visualizer(opt)
    visualizer.setup()  # regular setup:

    train_datasets = create_task_dataset(opt, phase="train")
    val_datasets = create_task_dataset(opt, phase="val")
    test_datasets = create_task_dataset(opt, phase="test")

    nb_tasks = train_datasets.nb_tasks

    test_matrix = TestMatrix()
    test(opt, test_datasets, model, train_index=0, visualizer=visualizer)

    for task_index in range(nb_tasks):
        # task_dataset = train_datasets[task_index]
        # val_dataset = val_datasets[task_index]
        model.setup(task_index)  # regular setup: load and print networks; create schedulers before training each task
        train(opt,
              model=model,
              task_index=task_index,
              continued_task_index=task_index,
              train_datasets=train_datasets,
              val_datasets=val_datasets,
              visualizer=visualizer,
              )

        if 'alwf' in opt.model_name:
            model.setup(task_index, step=2)
            train(opt,
                  model=model,
                  task_index=task_index,
                  continued_task_index=task_index,
                  train_datasets=train_datasets,
                  val_datasets=val_datasets,
                  visualizer=visualizer,
                  )

        test(opt, test_datasets, model, train_index=task_index + 1, visualizer=visualizer)

    test_matrix.save_matrix(os.path.join(opt.result_dir, f'{opt.name}.xlsx'))
