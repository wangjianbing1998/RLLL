"""This module implements an abstract base class (ABC) 'BaseDataset' for task_datasets.

It also includes common transformation functions (e.rg., get_transform, __scale_width), which can be later used in subclasses.
"""
import logging
from abc import ABC, abstractmethod
from collections import defaultdict

import torch.utils.data as data

__all__ = ['BaseDataset']

import PIL
from utils.util import split2n


class BaseDataset(data.Dataset, ABC):
	"""This class is an abstract base class (ABC) for task_datasets.

	To create a subclass, you need to implement the following four functions:
	-- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
	-- <__len__>:                       return the size of dataset.
	-- <__getitem__>:                   get a _data point.
	-- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
	"""

	def __repr__(self):
		return f'dataset {self.data_name}'

	def __init__(self, opt, phase):
		"""Initialize the class; save the options in the class

		Parameters:
			opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
		"""
		self.isTrain = False
		if 'train' in phase:
			self.isTrain = True
			self.phase = 'train'
		elif 'val' in phase:
			self.phase = 'val'
		elif 'test' in phase:
			self.phase = 'test'

		self.opt = opt
		self.max_dataset_size = -1
		self.set_max_dataset_size()

		if opt.max_dataset_size != -1:
			self.max_dataset_size = opt.max_dataset_size

		# specified in sub-class
		self.data_dir = None
		self._retargets = None
		self.data = None

		self.x_transforms_train = None
		self.x_transforms_test = None
		self.y_transforms = None

		self.target2label = None
		self.label2Indices = None
		self._labels = None

		self.choices_ps = None

	@abstractmethod
	def set_max_dataset_size(self):
		"""set max_dataset_size"""
		pass

	@staticmethod
	def modify_commandline_options(parser, isTrain):
		"""Add new dataset-specific options, and rewrite default values for existing options.

		Parameters:
			parser          -- original option parser
			is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

		Returns:
			the modified parser.
		"""

		# dataset parameters
		parser.add_argument('--load_size', type=int, default=64, help='scale image_size to this size')
		parser.add_argument('--crop_size', type=int, default=64, help='then crop to this size')
		parser.add_argument('--max_dataset_size', type=int, default=-1,
							help='Maximum number of samples allowed per label. -1 is [:-1], up load to n-1 samles, similarity to all')
		parser.add_argument('--preprocess', type=str, default='resize_and_crop',
							help='scaling and cropping of label2ImagePaths at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
		parser.add_argument('--no_flip', action='store_true',
							help='if specified, do not flip the label2ImagePaths for _data augmentation')
		parser.add_argument('--mnist_dataset_dir', type=str, default='./data/MNIST/')
		parser.add_argument('--cifar10_dataset_dir', type=str, default='./data/CIFAR10/')
		parser.add_argument('--cifar100_dataset_dir', type=str, default='./data/CIFAR100/')
		parser.add_argument('--imagenet_dataset_dir', type=str, default='/home/lichao/circle_conv/dataset/imagenet/')
		parser.add_argument('--miniimagenet_dataset_dir', type=str, default='./data/MINIImageNet/')
		parser.add_argument('--cub_dataset_dir', type=str, default='./data/CUB/CUB_200_2011/images/')

		parser.add_argument('--load_dataset_mode', type=str, default='reader',
							help="the mode for load dataset, 'dir' for directory loading or 'reader' for pytorch.data",
							choices=['dir', 'reader'])

		return parser

	@staticmethod
	def default_value(opt):
		return opt

	@property
	def retargets(self):
		return self._retargets

	@retargets.setter
	def retargets(self, value):
		self._retargets = value

	# logging.debug(f'dataset {self.data_name}.retargets = {value}')

	def __getitem__(self, item):
		"""Get items on label index

		"""
		data = self.data[item]
		if self.opt.load_dataset_mode == 'dir':
			image_path, target = data
			image = PIL.Image.open(image_path)

		elif self.opt.load_dataset_mode == 'reader':
			image, target = data
		else:
			raise ValueError(
				f'Expected load_dataset_mode choice from dir or reader, but got {self.opt.load_dataset_mode}')

		image = image.convert('RGB')  # need use three channels, instead of one channel
		if self.x_transforms_train is not None and self.phase == 'train':
			image = self.x_transforms_train(image)
		elif self.x_transforms_test is not None and self.phase in ['test', 'val']:
			image = self.x_transforms_test(image)

		target = self._retargets[target]

		if self.y_transforms is not None:
			target = self.y_transforms(target)
		return image, target

	def __len__(self):
		return len(self.data)

	def __call__(self, label, **kwargs):
		"""MnistDataset()(label) return a variety of indices, not real data"""

		indices = self.label2Indices[label]
		if self.max_dataset_size != -1:
			indices = indices[: self.max_dataset_size]
		return indices

	@property
	def labels(self):
		return self._labels

	def split2n_on_tasks(self, nb_tasks):
		"""split the dataset into <nb_tasks> splits, on labelsOnTask
		Examples

		self.labelsOnTask = [labels for i in range(nb_tasks)]
		nb_tasks=3

		Return
		str for all([[0,1,2,3],[4,5,6,7],[8,9]])
		"""
		labelsOnTask = split2n(self.labels, nb_tasks)
		return labelsOnTask

	def get_label2filepaths(self, label_filepaths, phase=None):
		res_label2filepaths = defaultdict(list)
		for label in label_filepaths:
			filepaths = label_filepaths[label]
			len_filepaths = len(filepaths)
			if phase == 'train':
				offset = 0
			elif phase == 'val':
				offset = len_filepaths * self.choices_ps['train']
			elif phase == 'test':
				offset = len_filepaths * (self.choices_ps['train'] + self.choices_ps['val'])
			else:
				raise ValueError(f'Expected phase in train|val|test , but got {phase}')

			offset = int(offset)

			res_label2filepaths[label] = filepaths[offset:offset + int(len_filepaths * self.choices_ps[phase])]

		logging.debug(f'load dataset {self.data_name}.{phase}')
		return res_label2filepaths
