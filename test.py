# encoding=utf-8
'''
@File    :   test.py    
@Contact :   jianbingxiaman@gmail.com
@License :   (C)Copyright 2020-2021, John Hopcraft Lab-CV
@Desciption :  test the model and get the test_matrix.xlsx
@Modify Time      @Author    @Version
------------      -------    --------
2021/2/25 10:41   jianbingxia     1.0    
'''
import logging
import os
import time

from sklearn.utils import Bunch

from models import BaseModel, create_model
from options.test_options import TestOptions
from task_datasets import PseudoData, create_task_dataset
from util.util import TestMatrix, MatrixItem
from util.visualizer import Visualizer


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
        model.set_data(PseudoData(opt, Bunch(**data)))
        model.test(visualizer)
        matrixItems.append(model.get_matrix_item(task_index))

    res = sum(matrixItems, MatrixItem()(0))
    res = res / len(matrixItems)
    logging.info(f"Validation Time Taken: {time.time() - start_time} sec")
    return res


if __name__ == '__main__':
    opt = TestOptions().parse()  # get training options

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    visualizer = Visualizer(opt)
    visualizer.setup()  # regular setup:

    # visualizer.add_graph(model,) TODO the model graph visualization

    test_datasets = create_task_dataset(opt, phase="test")

    nb_tasks = test_datasets.nb_tasks

    test_matrix = TestMatrix()
    test(opt, test_datasets, model, train_index=0, visualizer=visualizer)

    test_matrix.save_matrix(os.path.join(opt.result_dir, f'{opt.name}.xlsx'))
