# encoding=utf-8
'''
@File    :   base_task_dataset.py
@Contact :   jianbingxiaman@gmail.com
@License :   (C)Copyright 2020-2021, John Hopcraft Lab-CV
@Desciption : 
@Modify Time      @Author    @Version
------------      -------    --------
2021/1/26 17:26   jianbingxia     1.0    
'''
"""This module implements an abstract base class (ABC) 'BaseTaskDataset' for task_datasets.
"""
from abc import ABC, abstractmethod


class BaseTaskDataset(ABC):
    """This class is an abstract base class (ABC) for task_datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a _data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt,phase="train"):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.phase=phase

    @staticmethod
    def modify_commandline_options(parser,isTrain):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """

		return parser

	@staticmethod
	def default_value(opt):
		return opt

	@abstractmethod
	def __len__(self):
		"""Return the total number of tasks."""
		return 0

	@abstractmethod
	def __getitem__(self, task_index):
		"""Return a SimpleDataset for current task(index of task_index).

		Parameters:
			task_index - - a random integer for task indexing

		Returns:
			a SimpleDataset
		"""
        pass
