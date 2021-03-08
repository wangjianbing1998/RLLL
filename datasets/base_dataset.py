"""This module implements an abstract base class (ABC) 'BaseDataset' for task_datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
from abc import ABC

import torch
import torch.utils.data as data

__all__ = ['BaseDataset']

from PIL.Image import Image
from sklearn.utils import Bunch

from util.util import is_gpu_avaliable, split2n


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for task_datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a _data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt, phase):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        self.isTrain = True if phase == 'train' else False
        self.phase = phase
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

    def __getitem__(self, item):
        """Get items on label index

        """
        if len(item) == 2:
            data_index, relabels = item
        else:
            raise ValueError(f'relabels assignment error, please check SimpleDataset.__getitem__(item)')

        data = self.data[data_index]
        if self.opt.load_dataset_mode == 'dir':
            image_path, target = data.image_path, data.target
            image = Image.open(image_path)

        elif self.opt.load_dataset_mode == 'reader':
            image, target = data
        else:
            raise ValueError(
                f'Expected load_dataset_mode choice from dir or reader, but got {self.opt.load_dataset_mode}')

        image = image.convert('RGB')  # need use three channels, instead of one channel
        if self.x_transforms is not None:
            image = self.x_transforms(image)

        target = relabels[self.target2label[target]]

        if self.y_transforms is not None:
            target = self.y_transforms(target)
        target = torch.LongTensor([target])

        if is_gpu_avaliable(self.opt):
            image = image.to(self.opt.device)
            target = target.to(self.opt.device)
        return Bunch(image=image,
                     target=target)

    def __len__(self):
        return len(self.data)

    def __call__(self, label, **kwargs):
        """MnistDataset()(label) return a variety of indices, not real data"""
        indices = self.label2Indices[label][:self.opt.max_dataset_size]
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
