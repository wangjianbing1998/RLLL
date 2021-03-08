import os
from collections import defaultdict

from sklearn.utils import Bunch
from torchvision import datasets
from torchvision.transforms import transforms

from VARIABLES import CIFAR10
from datasets import BaseDataset, prepare_datas_by_standard_data


class Cifar10Dataset(BaseDataset):
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
        # TODO split the dataset of val and test
        if phase == "val":
            phase = "test"

        super(Cifar10Dataset, self).__init__(opt, phase)
        self.data_dir = opt.cifar10_dataset_dir

        self.data_name = CIFAR10

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
            self.target2label = dict([(target, label) for target, label in enumerate(self.labels)])

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
            dataset = datasets.CIFAR10(root=os.path.join(self.data_dir, 'raw_data'), train=self.isTrain,
                                       download=True)
            self.data, self._labels, self.label2Indices, self.label2target, self.target2label = prepare_datas_by_standard_data(
                dataset)
        else:
            raise ValueError(f"Expected load_dataset_mode in [dir,reader], but got {self.opt.load_dataset_mode}")
