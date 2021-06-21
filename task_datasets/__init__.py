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
import re
from collections import defaultdict

from VARIABLES import nc_datas, MINIIMAGENET
from datasets import dataset_names
from task_datasets.base_task_dataset import BaseTaskDataset
from utils.util import split2numclasses, MultiOutput

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


def dataname2taskindex(dataset_list):
	"""
	calculate the <dataname2taskIndices>,  # {mnist:[0,1], imagenet:[2]}
	Args:
		dataset_list:

	Returns: dict

	>>> dataname2taskindex(['mnist_1','mnist_2','mnist_3'])
	defaultdict(<class 'list'>, {'mnist': [0, 1, 2]})
	"""
	dataname2taskIndices = defaultdict(list)
	for dataset_name in dataset_names:
		for index, data_name in enumerate(dataset_list):
			data_name = data_name.lower()
			if '_' in data_name:
				data_name = data_name[:data_name.index('_')]
			if dataset_name == data_name:  # split by "_",
				dataname2taskIndices[data_name].append(index)
	return dataname2taskIndices


def get_num_classes_by_data_list(dataset_list):
	"""

	Args:
		dataset_list: [mnist_1,mnist_2,mnist_3]

	Returns:
		the <num_classes> with the input dataset_list
	>>> get_num_classes_by_data_list(["mnist_1","mnist_2","mnist_3"])
	[4, 4, 2]
	>>> get_num_classes_by_data_list(["cifar100","cifar10","mnist"])
	[100, 10, 10]
	"""
	nb_tasks = len(dataset_list)
	dataname2taskIndices = dataname2taskindex(dataset_list)

	num_classes = [0] * nb_tasks
	for data_name, indices in dataname2taskIndices.items():

		if data_name == MINIIMAGENET and len(indices) == nb_tasks:
			num_classes = [64, 20, 16]
		else:
			classes = split2numclasses(range(nc_datas[data_name]), len(indices))
			for index, cls in zip(indices, classes):
				num_classes[index] = cls

	return num_classes


def get_data_list_by_str(dataset_list_str: str) -> [str]:
	"""

	Args:
		dataset_list: "mnist_cifar10_cifar100" or "mnist_10cifar100"

	Returns:
		dataset_list
	>>> get_data_list_by_str('mnist_cifar10_cifar100')
	['mnist', 'cifar10', 'cifar100']
	>>> get_data_list_by_str('mnist_3cifar100')
	['mnist', 'cifar100_1', 'cifar100_2', 'cifar100_3']
	"""
	datas = dataset_list_str.split('_')
	dataset_list = []
	for data in datas:
		g = re.match('^(\d+)?(.*?)$', data)
		count, name = g.groups()
		if count is None:
			count = 1
		count = int(count)
		dataset_list.extend([f'{name}{f"_{index + 1}" if count > 1 else ""}' for index in range(count)])

	return dataset_list


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
		if 'taskdataset' in name.lower():
			taskdataset_names.append(name.lower())
			if name.lower() == target_loss_name.lower():
				task_dataset = cls

	if task_dataset is None:
		logging.error(
			"In %s.py, there should be a subclass of BaseTaskDataset with class tag that matches %s in lowercase." % (
				("task_datasets." + taskdataset_name + "_task_dataset"), target_loss_name))
		exit(0)

	return task_dataset


def get_cls(task_dataset_name):
	"""Return the static method <modify_commandline_options> of the model class."""
	return find_taskdataset_using_name(task_dataset_name)


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
	logging.info(f"task_dataset [{type(instance).__name__}]({instance})-{phase} was created")

	return instance


class PseudoData(object):
	""" For LifeLongLearning dataset

	pdatas: MultiOutput, the output on model(data.image), if none, it represents the only data from dataset, not contains the pdata amang the other output
	task_index: int, current task index
	data: one element on BaseDataset
	self.image=tensor
	self.target=MultiOutput if pdatas is not None else Tensor

	"""

	def __init__(self, opt, image, target=None, pdatas: MultiOutput = None, task_index=None):

		self.opt = opt
		self._image = image
		self._target: 'Single Output' = target

		self.cuda_image = False
		self.cuda_target = False

		if pdatas is not None:
			if target is not None:
				pdatas[task_index] = target

			self._target: MultiOutput = pdatas

	@property
	def image(self):
		return self._image

	@property
	def target(self):
		return self._target

	def __repr__(self):
		return f'PseudoData(image,{"MultiOutput" if isinstance(self._target, MultiOutput) else "SingleOutput"})'

	def cuda(self, device=None):
		if device is None:
			device = self.opt.device

		if not self.cuda_image:
			self.cuda_image = True
			self._image = self._image.to(device, non_blocking=True)
		if not self.cuda_target:
			self.cuda_target = True
			self._target = self._target.cuda(device, non_blocking=True)
