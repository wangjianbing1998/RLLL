import csv
import os
from collections import defaultdict

import torch.utils.data as data
from sklearn.utils import Bunch
from torchvision.transforms import transforms

# from torchvision.datasets import VisionDataset
from VARIABLES import MINIIMAGENET
from datasets import BaseDataset, prepare_datas_by_standard_data


class MINIImageNet(data.Dataset):

    def __init__(self, label2filepaths):
        self.classes = list(set(label2filepaths.keys()))
        self.class_to_idx = dict([(label, target) for target, label in enumerate(self.classes)])

        self.data, self.targets = self.build_data(label2filepaths)

    def __getitem__(self, item):
        image_path, target = self.data[item]
        return Bunch(image_path=image_path, target=target)

    def __len__(self):
        return len(self.data)

    def build_data(self, label2filepaths):
        data = []
        targets = []
        for label, filepaths in label2filepaths.items():
            data.extend([(filepath, self.class_to_idx[label]) for filepath in filepaths])
            targets.extend([self.class_to_idx[label] for filepath in filepaths])

        return data, targets


class MiniImagenetDataset(BaseDataset):
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

        opt.load_dataset_mode = 'dir'
        super(MiniImagenetDataset, self).__init__(opt, phase)
        self.data_dir = opt.miniimagenet_dataset_dir

        self.data_name = MINIIMAGENET

        self.x_transforms = transforms.Compose([transforms.Resize((opt.imsize, opt.imsize)),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                ])
        self.y_transforms = None

        dataset = self.load_dataset()
        self.data, self._labels, self.label2Indices, self.label2target, self.target2label = prepare_datas_by_standard_data(
            dataset)

    def loadCSV(self, csvf):
        """
        return a dict saving the information of csv
        :param splitFile: csv file name
        :return: {label:[file1, file2 ...]}
        """
        path = os.path.join(self.data_dir, 'images')

        dictLabels = defaultdict(list)
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[0]
                label = row[1]
                dictLabels[label].append(os.path.join(path, filename))
        return dictLabels

    def load_dataset(self):
        label2filepaths = self.loadCSV(os.path.join(self.data_dir, self.phase + '.csv'))

        dataset = MINIImageNet(label2filepaths)
        return dataset
