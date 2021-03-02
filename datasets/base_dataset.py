"""This module implements an abstract base class (ABC) 'BaseDataset' for task_datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
from abc import ABC, abstractmethod

import torch.utils.data as data

__all__ = ['BaseDataset']


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for task_datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a _data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.isTrain = opt.isTrain
        self.opt = opt
        self.data_dir = None  # specified in sub-class

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

        parser.add_argument('--load_dataset_mode', type=str, default='reader',
                            help="the mode for load dataset, 'dir' for directory loading or 'reader' for pytorch.data",
                            choices=['dir', 'reader'])

        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of label2ImagePaths in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a _data point and its metadata information.

        Parameters:
            index - - a random integer for _data indexing

        Returns:
            a dictionary of _data with their names. It ususally contains the _data itself and its metadata information.
        """
        pass
