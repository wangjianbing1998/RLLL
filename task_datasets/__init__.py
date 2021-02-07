# encoding=utf-8
'''
@File    :   __init__.py.py    
@Contact :   jianbingxiaman@gmail.com
@License :   (C)Copyright 2020-2021, John Hopcraft Lab-CV
@Desciption : 
@Modify Time      @Author    @Version
------------      -------    --------
2021/1/23 12:37   jianbingxia     1.0    
'''
import logging

from task_datasets.base_task_dataset import BaseTaskDataset
from util.util import MultiOutput, print_tensor

"""This package contains losses_without_lambda related to objective functions, optimizations, and network architectures.

To add a custom model class called 'dummy', you need to add a file called 'dummy_model.py' and define a subclass DummyModel inherited from BaseModel.
You need to implement the following five functions:
    -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
    -- <set_data>:                     unpack _data from dataset and apply preprocessing.
    -- <forward>:                       produce intermediate results.
    -- <optimize_parameters>:           calculate loss, gradients, and update network weights.
    -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.

In the function <__init__>, you need to define four lists:
    -- self.taskdataset_names (str list):          specify the training losses_without_lambda that you want to plot and save.
    -- self.net_names (str list):         define networks used in our training.
    -- self.visual_names (str list):        specify the label2ImagePaths that you want to display and save.
    -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an usage.

Now you can use the model class by specifying flag '--model dummy'.
See our template model class 'template_model.py' for more details.
"""

import importlib

taskdataset_names = []


def find_taskdataset_using_name(taskdataset_name):
    """Import the module "models/[net_name]_model.py".

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    losslib = importlib.import_module("task_datasets." + taskdataset_name + "_task_dataset")
    task_dataset = None
    target_loss_name = taskdataset_name.replace('_', '') + 'taskdataset'
    for name, cls in losslib.__dict__.items():
        if 'taskdataset' in name.lower() and issubclass(cls, BaseTaskDataset):
            taskdataset_names.append(name.lower())
            if name.lower() == target_loss_name.lower():
                task_dataset = cls

    if task_dataset is None:
        logging.error(
            "In %s.py, there should be a subclass of BaseTaskDataset with class tag that matches %s in lowercase." % (
                ("task_datasets." + taskdataset_name + "_task_dataset"), target_loss_name))
        exit(0)

    return task_dataset


def get_option_setter(task_dataset_name):
    """Return the static method <modify_commandline_options> of the model class."""
    return find_taskdataset_using_name(task_dataset_name).modify_commandline_options


def create_task_dataset(opt, phase="train"):
    """Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'fit.py'/'test.py'

    Example:
        >>> from  task_datasets import create_task_dataset
        >>> loss = create_task_dataset(opt)
    """
    task_dataset = find_taskdataset_using_name(opt.task_dataset_name)
    instance = task_dataset(opt, phase)
    logging.info(f"task_dataset [{type(instance).__name__}]-{phase} was created")
    return instance


class PseudoData(object):
    """ For LifeLongLearning dataset

    pdatas: MultiOutput, the output on model(data.image), if none, it represents the only data from dataset, not contains the pdata amang the other output
    task_index: int, current task index
    data: one element on BaseDataset
    self.image=tensor
    self.target=MultiOutput if pdatas is not None else Tensor

    """

    def __init__(self, data, pdatas: MultiOutput = None, task_index=None):
        self._image = data.image
        self._target = data.target
        if pdatas is not None:
            pdatas[task_index] = data.target
            self._target = pdatas

    @property
    def image(self):
        return self._image

    @property
    def target(self):
        return self._target

    def __repr__(self):
        return (
            f'PseudoData \n\timage={print_tensor(self._image, pt=False)}\n\ttarget={print_tensor(self._target.output if hasattr(self._target,"output") else self._target, pt=False)}')