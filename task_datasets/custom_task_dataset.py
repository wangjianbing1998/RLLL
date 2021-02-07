# encoding=utf-8
'''
@File    :   custom_task_dataset.py    
@Contact :   jianbingxiaman@gmail.com
@License :   (C)Copyright 2020-2021, John Hopcraft Lab-CV
@Desciption : 
@Modify Time      @Author    @Version
------------      -------    --------
2021/1/26 17:25   jianbingxia     1.0    
'''
import logging
from collections import defaultdict

from datasets import dataset_names, create_splited_datasets, SimpleDataset
from task_datasets.base_task_dataset import BaseTaskDataset
from util.util import log


class CustomTaskDataset(BaseTaskDataset):
    """Custom defined tasks task_datasets


    for example, if opt.<dataset_list> specifies with ["mnist1","mnist2","imagenet"], then the task dataset will be [MnistDataset,MnistDataset,ImagenetDataset]
    and two MnistDataset splits from seperate labels on MnistDataset


    task2dataset: [SimpleDataset_fortask0,SimpleDataset_fortask1,SimpleDataset_fortask2,...]
    """
    def __init__(self, opt, phase="train"):
        BaseTaskDataset.__init__(self, opt, phase)
        self.dataset_list = opt.dataset_list
        self.nb_tasks = len(self.dataset_list)
        self._build_task_dataset()

    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--dataset_list', type=list, default=['mnist1', 'mnist2', 'mnist3'],
                            help='chooses which any task_datasets are loaded. ')
        parser.add_argument('--num_classes', type=list, default=[10, 10, 10], help="the num_class per task")
        return parser

    @log(level = "debug")
    def _build_task_dataset(self):
        """build <self.task2dataset> """
        # calculate the <dataname2taskIndices>,  # {mnist:[0,1], imagenet:[2]}

        dataname2taskIndices = defaultdict(list)
        for dataset_name in dataset_names:
            for index, data_name in enumerate(self.dataset_list):
                if (dataset_name.replace("dataset", "")) in data_name:
                    dataname2taskIndices[dataset_name].append(index)

        # build task2dataset
        self.task2dataset = [SimpleDataset] * self.nb_tasks
        for data_name, indices in dataname2taskIndices.items():
            datasets_gen = create_splited_datasets(self.opt, self.phase, data_name, len(indices))
            for task_index in indices:
                try:
                    self.task2dataset[task_index] = next(datasets_gen)
                except StopIteration:
                    logging.error(f'Please check function <create_splited_datasets>')
                    exit(0)

    def __len__(self):
        return self.nb_tasks

    def __getitem__(self, task_index):
        return self.task2dataset[task_index]
