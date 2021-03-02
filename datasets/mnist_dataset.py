import os
from collections import defaultdict

import torch
from PIL import Image
from sklearn.utils import Bunch
from torchvision import datasets
from torchvision.transforms import transforms

from VARIABLES import MNIST
from datasets import BaseDataset, prepare_datas_by_standard_data
from util.util import split2n, is_gpu_avaliable


class MnistDataset(BaseDataset):
    """Mnist Dataset

    phase: "fit", "val", "test"
    transform: Resize(), ToTensor(), Normalize()
    _data:[Bunch]
        {
            image_paths: [Path,...]
            targets: [int,...], for examples, [0,1,2,3,...], not label name, but label index
        }
    label2Indices: {str:[0,1,2,3],...}
    labels:[str]
    label2target:{str:int}, {"0":0,"1":1,...}

    """

    def __init__(self, opt, phase="train"):
        super(MnistDataset, self).__init__(opt)
        self.data_dir = opt.mnist_dataset_dir
        # TODO split the dataset of val and test
        if phase == "val":
            phase = "test"
        self.data_name = MNIST
        self.phase = phase
        self.x_transforms = transforms.Compose(
            [
                transforms.Resize((opt.imsize, opt.imsize)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (1.0,))
            ]
        )
        self.y_transforms = None
        if self.opt.load_dataset_mode == 'dir':

            self.data = []  # image_paths,targets
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
                    target = self.label2target[label]

                    self.label2Indices[label].append(len(self.data))

                    self.data.append(Bunch(image_path=image_path,
                                           target=target)
                                     )
        elif self.opt.load_dataset_mode == 'reader':
            dataset = datasets.MNIST(root=os.path.join(self.data_dir, 'raw_data'), train=self.isTrain,
                                     download=True)
            self.data, self._labels, self.label2Indices, self.label2target = prepare_datas_by_standard_data(dataset)
        else:
            raise ValueError(f"Expected load_dataset_mode in [dir,reader], but got {self.opt.load_dataset_mode}")

    def __getitem__(self, item):
        """Get items on label index"""
        data = self.data[item]

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

        if self.y_transforms is not None:
            target = self.y_transforms(target)
        target = torch.LongTensor([target])
        if is_gpu_avaliable(self.opt):
            image = image.cuda()
            target = target.cuda()
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
