import csv
import logging
import os
from collections import defaultdict

from torchvision.transforms import transforms

from VARIABLES import MINIIMAGENET
from datasets import BaseDataset, prepare_datas_by_standard_data, FolderDataset


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
    label2filepaths = {
        'train.csv': None,
        'val.csv': None,
        'test.csv': None,
    }

    def __init__(self, opt, phase="64-train"):

        nb_cls_of_task, phase = phase.split('-')
        nb_cls_of_task = int(nb_cls_of_task)

        if nb_cls_of_task == 64:
            self.task_file = 'train.csv'
        elif nb_cls_of_task == 20:
            self.task_file = 'test.csv'
        elif nb_cls_of_task == 16:
            self.task_file = 'val.csv'
        else:
            raise ValueError(f'Expected nb_cls_of_task in {64, 20, 16}, but got {nb_cls_of_task}')

        opt.load_dataset_mode = 'dir'
        super(MiniImagenetDataset, self).__init__(opt, phase)

        self.choices_ps = {
            'train': .8,
            'val': .1,
            'test': .1,
        }

        self.data_dir = opt.miniimagenet_dataset_dir

        self.data_name = MINIIMAGENET

        self.x_transforms = transforms.Compose([transforms.Resize((opt.imsize, opt.imsize)),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                ])
        self.y_transforms = None

        # load all dataset
        label2filepaths = MiniImagenetDataset.get_dataset(self.task_file,
                                                          os.path.join(self.data_dir, self.task_file),
                                                          self.data_name, self.data_dir
                                                          )

        # load phase dataset
        dataset = self.load_dataset(label2filepaths, phase)

        self.data, self._labels, self.label2Indices, self.label2target, self.target2label = prepare_datas_by_standard_data(
            dataset)

    @classmethod
    def get_dataset(cls, task_file, csvf, data_name, data_dir, ):
        """
        return a dict saving the information of csv
        :param splitFile: csv file name
        :return: {label:[file1, file2 ...]}
        """

        label2filepaths = cls.label2filepaths[task_file]
        if label2filepaths is None:

            logging.debug(f"load all {task_file} dataset {data_name}")

            path = os.path.join(data_dir, 'images')

            label2filepaths = defaultdict(list)
            with open(csvf) as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                next(csvreader, None)  # skip (filename, label)
                for i, row in enumerate(csvreader):
                    filename = row[0]
                    label = row[1]
                    label2filepaths[label].append(os.path.join(path, filename))
        cls.label2filepaths[task_file] = label2filepaths
        return label2filepaths

    def load_dataset(self, label2filepaths, phase):
        label2filepaths_phase = self.get_label2filepaths(label2filepaths, phase)

        dataset = FolderDataset(label2filepaths_phase)
        return dataset

    def set_max_dataset_size(self):
        super().set_max_dataset_size()
        self.max_dataset_size = 5000  # number of images per class
