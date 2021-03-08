"""This package includes all the modules related to _data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a _data point from _data dataloader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import logging
import os
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data

from datasets.base_dataset import BaseDataset
from util.util import relabel

dataset_names = [dir.replace("_dataset.py", "").lower() for dir in
                 os.listdir(os.path.dirname(os.path.abspath(__file__))) if
                 "_dataset.py" in dir and "base" not in dir]


def find_dataset_using_name(dataset_name):
    """Import the module "_data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "datasets." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if 'dataset' in name.lower():
            if name.lower() == target_dataset_name.lower():
                dataset = cls

    if dataset is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of BaseDataset with class tag that matches %s in lowercase." % (
                dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


class SimpleDataset(data.Dataset):
    """Get Task-Labeled DataItem accroding to the index inside original dataset"""

    def __init__(self, data_index, dataset, labels, shuffle=False):
        self._data_index = data_index
        self._dataset = dataset
        self.data_name = dataset.data_name
        self.shuffle = shuffle

        if self.shuffle:
            import random
            random.shuffle(self._data_index)

        self._relabels = relabel(labels)

    @property
    def relabels(self):
        return self._relabels

    @relabels.setter
    def relabels(self, value):
        self._relabels = value

    def __getitem__(self, item):
        index = self._data_index[item]
        return self._dataset[(index, self._relabels)]

    def __len__(self):
        return len(self._data_index)


def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    """
    opt.preprocess:
                    resize: resize to [opt.load_size,opt.load_size],
                    scale_width:
    Args:
        opt:
        params:
        grayscale:
        method:  the
        convert:

    Returns:

    """
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img:__scale_width_height(img, opt.load_size, method)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img:__crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img:__make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img:__flip(img, params['flip'], Image.FLIP_LEFT_RIGHT)))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __one_hot(Ys, nb_classes):
    return np.eye(nb_classes)[Ys]


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width_height(img, target_width=None, target_height=None, method=Image.BICUBIC):
    """resize image into (target_width,target_height)
    target_width/target_height = ow/oh

    raise ValueError, if target_height<=0 or target_width<=0
    """
    if target_height > 0 and target_width:
        raise ValueError(
            f"Expected target_width>0 and target_height>0, but got target_width={target_width}, target_height={target_height}")

    ow, oh = img.size
    if target_width is not None and target_height is None:
        target_height = int(target_width * oh) / ow
    elif target_width is None and target_height is not None:
        target_width = int(target_height * ow) / oh
    elif target_width is None and target_height is None:
        return img

    return img.resize((target_width, target_height), method)


def __crop(img, pos, size):
    """crop the image at position [pos,pos+size]"""
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip, flip_type=Image.FLIP_LEFT_RIGHT):
    """ flip the image if flip is True

    flip_type(int): the flip type on img, only work if the flip is True
    """
    if flip:
        return img.transpose(flip_type)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        logging.warning(
            f"The loaded image size was ({ow}, {oh}), so it was adjusted to ({w}, {h}).This adjustment will be done to all label2ImagePaths")
        __print_size_warning.has_printed = True


def prepare_datas_by_standard_data(d: 'torchvision.datasets.XXXX'):
    """
    Returns:
        data, list(Bunch(image,target))
        labels, list(str)
        label2Indices, {str:list(int),}
        label2target, {str:int}

    """

    data = d
    labels = d.classes
    label2target = d.class_to_idx
    target2label = dict([(target, label) for label, target in label2target.items()])

    label2Indices = defaultdict(list)
    for index, target in enumerate(d.targets):
        if isinstance(target, torch.Tensor):
            target = target.item()
        label2Indices[target2label[target]].append(index)

    return data, labels, label2Indices, label2target, target2label
