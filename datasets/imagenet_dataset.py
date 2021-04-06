# encoding=utf-8
'''
@File    :   imagenet_dataset.py
@Contact :   jianbingxiaman@gmail.com
@License :   (C)Copyright 2020-2021, John Hopcraft Lab-CV
@Desciption :
@Modify Time      @Author    @Version
------------      -------    --------
2021/2/7 19:54   jianbingxia     1.0
'''
import os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from sklearn.utils import Bunch
from torchvision.transforms import transforms

from VARIABLES import IMAGENET
from datasets import BaseDataset
from utils.util import split2n, is_gpu_avaliable


class ImagenetDataset(BaseDataset):
    """Imagenet Dataset

    phase: "fit", "val", "test"
    transform: Resize(), ToTensor(), Normalize()
    _data:[Bunch]
        {
            image_paths: [Path,...]
            image_names: [FileName,...]
            targets: [int,...], for examples, [0,1,2,3,...], not label name, but label index
        }
    label2Indices: {str:[0,1,2,3],...}
    labels:[str]
    label2target:{str:int}, {"0":0,"1":1,...}

    """

    def __init__(self, opt, phase="train"):
        super(ImagenetDataset, self).__init__(opt)
        self.data_dir = opt.imagenet_dataset_dir
        # TODO split the dataset of val and test
        if phase == "val":
            phase = "test"
        self.data_name = IMAGENET
        self.phase = phase
        self.x_transforms = transforms.Compose(
            [
                transforms.Resize((opt.imsize, opt.imsize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]
        )

        self.data = []  # image_paths,image_names,targets

        self.label2Indices = defaultdict(list)
        image_dir = os.path.join(self.data_dir, phase)

        self._labels = os.listdir(image_dir)
        # get label to targets, dict type
        self.label2target = dict([(label, target) for target, label in enumerate(self.labels)])

        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image Dir {image_dir} not exists, please check it")
        for root, label_dirs, files in os.walk(image_dir):
            for file in files:
                label = os.path.basename(root)

                image_path = os.path.join(root, file)
                image_name = os.path.splitext(file)[0]
                target = self.label2target[label]

                self.label2Indices[label].append(len(self.data))

                self.data.append(Bunch(image_path=image_path,
                                       image_name=image_name,
                                       target=target)
                                 )

    def __getitem__(self, item):
        """Get items on label index"""
        data = self.data[item]
        image_path, image_name, target = data.image_path, data.image_name, data.target

        image = Image.open(image_path).convert('RGB')
        if self.x_transforms is not None:
            image = self.x_transforms(image)
        target = torch.from_numpy(np.array(target, dtype=np.float))

        if is_gpu_avaliable(self.opt):
            image = image.to(self.opt.device)
            target = target.to(self.opt.device)
        return Bunch(image_name=image_name,
                     image=image,
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
        str for all([[0,1,2,3],[4,5,6,7],[8,9,10,11],...])
        """
        labelsOnTask = split2n(self.labels, nb_tasks)
        return labelsOnTask
