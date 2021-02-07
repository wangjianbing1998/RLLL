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

from sklearn.utils import Bunch

from task_datasets import PseudoData, create_task_dataset
from util.util import TestMatrix, MatrixItem

"""General-purpose training script for multi-task learning(or LifeLong Learning) translation.

This script works for various models (with option '--model_name': e.g., lwf, finetune, warmtune, hottune, folwf, rlll) and
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
from util.visualizer import Visualizer


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

        for i, data in enumerate(train_dataset):  # inner loop within one epoch
            previous_data = PseudoData(Bunch(**data))
            model.set_data(previous_data)
            model.test(visualizer=visualizer)  # Get model.output
            '''
            data.image=data.image
            data.target=[output,output,...,<data.target>,output,output,...]
            '''
            data = PseudoData(previous_data, model.output, task_index)  #

            # logging.debug("after:"+str(data))
            assert all(data.target[task_index] == previous_data.target)
            # set data and fit
            model.set_data(data)  # unpack _data from dataset and apply preprocessing
            model.train(task_index)

        train_losses = model.get_current_losses()
        if epoch % opt.curve_freq == 0:  # visualizing training losses and save logging information to the disk
            visualizer.add_losses(train_losses, epoch)
        # Validation
        val_matrix_item = val(opt, val_dataset, model, task_index, visualizer)

        if (epoch + 1) % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            logging.info('saving the model at the end of epoch %d' % (epoch))
            model.save_networks(epoch)

        if opt.save_best and (best_matrix_item is None or val_matrix_item > best_matrix_item):
            logging.info(f'saving the best model at the end of epoch {epoch}')
            model.save_networks(epoch="best")

        logging.info(f'End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t train_loss={train_losses["loss_total"].item()}, Time Taken: {time.time() - epoch_start_time} sec' )
        model.update_learning_rate()  # update learning rates at the end of every epoch.


def train(opt, model, task_index, continued_task_index, train_dataset, val_dataset=None, visualizer=None):
    fit(opt,
        model=model,
        task_index=task_index,
        continued_task_index=continued_task_index,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        visualizer=visualizer)

    if model.need_backward and task_index >= 1:
        """backward training"""
        train(opt,
              model=model,
              task_index=task_index - 1,
              continued_task_index=continued_task_index,
              train_dataset=train_dataset,
              val_dataset=val_dataset,
              visualizer=visualizer,
              )


def test(opt, test_datasets, model: BaseModel, train_index, visualizer=None):
    """test the model on multi-task test_datasets, after training task indexed with <train_index>

    Return
    None
    the global testMatrix will be updated
    """

    for test_index, test_dataset in enumerate(test_datasets):
        matrixItem = val(opt, test_dataset, model, test_index, visualizer, )
        test_matrix[(train_index, test_index)] = matrixItem


def val(opt, val_dataset, model: BaseModel, task_index, visualizer=None) -> MatrixItem:
    """for validation on one task"""
    logging.info(f"Validating task {task_index}")
    start_time = time.time()  # timer for validate a task

    matrixItems = []
    for i, data in enumerate(val_dataset):  # inner loop within one epoch
        # if is_gpu_avaliable(opt):
        #     model.cuda()
        # Get output
        model.set_data(PseudoData(Bunch(**data)))
        model.test(visualizer)
        # Add matrixItem result
        matrixItems.append(model.get_matrix_item(task_index))

    res = sum(matrixItems, MatrixItem()(0))
    res = res / len(matrixItems)
    logging.info(f"Validation Time Taken: {time.time() - start_time} sec")
    return res


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

    test_matrix = TestMatrix()
    test(opt, test_datasets, model, train_index=0, visualizer=visualizer)

    for task_index in range(nb_tasks):
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
        test(opt, test_datasets, model, train_index=task_index, visualizer=visualizer)
    test_matrix.save_matrix(f'{opt.name}.xlsx')

