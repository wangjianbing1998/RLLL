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

from VARIABLES import MINIIMAGENET
from datasets import SimpleDataset, find_dataset_using_name, DataLoaderX
from task_datasets import dataname2taskindex, get_num_classes_by_data_list, get_data_list_by_str
from task_datasets.base_task_dataset import BaseTaskDataset
from utils.util import flat_iterators


class CustomTaskDataset(BaseTaskDataset):
    """Custom defined tasks task_datasets


    for example, if opt.<dataset_list> specifies with ["mnist1","mnist2","imagenet"], then the task dataset will be [MnistDataset,MnistDataset,ImagenetDataset]
    and two MnistDataset splits from seperate labels on MnistDataset


    task2dataset: [SimpleDataset_fortask0,SimpleDataset_fortask1,SimpleDataset_fortask2,...]
    """

    def __repr__(self):
        s = []
        for _, dataset in self.task2dataset:
            s.append((dataset.data_name, dataset.len_data))
        return f'{s}'

    def __init__(self, opt, phase="train"):
        BaseTaskDataset.__init__(self, opt, phase)
        self.dataset_list = opt.dataset_list
        self.nb_tasks = len(self.dataset_list)
        self._build_task_dataset()

    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--dataset_list', type=str, default='mnist_cifar10_cifar100',
                            help='chooses which any task_datasets are loaded. ')
        parser.add_argument('--num_classes', type=list, default=None,
                            help='chooses which any task_datasets are loaded. ')

        return parser

    @staticmethod
    def default_value(opt):
        opt.dataset_list = get_data_list_by_str(opt.dataset_list)
        opt.num_classes = get_num_classes_by_data_list(opt.dataset_list)
        return opt

    def _build_task_dataset(self):
        """build <self.task2dataset> """
        dataname2taskIndices = dataname2taskindex(self.dataset_list)

        # build task2dataset
        self.task2dataset = [(DataLoaderX, SimpleDataset)] * self.nb_tasks
        for data_name, indices in dataname2taskIndices.items():
            if data_name == MINIIMAGENET and len(indices) == self.nb_tasks:  # only run for a loop
                if self.nb_tasks <= 3:
                    logging.info(
                        f"Load phase={self.phase}-{MINIIMAGENET} train.csv(64) for task1, test.csv(20) for task 2, val.csv(16) for task 3")

                    simple_datasets = [
                        self.create_splited_datasets(f'64-{self.phase}', data_name, 1)[0],  # 64-train.csv for task1
                        self.create_splited_datasets(f'20-{self.phase}', data_name, 1)[0],  # 20-test.csv for task2
                        self.create_splited_datasets(f'16-{self.phase}', data_name, 1)[0],  # 16-val.csv for task3
                    ]
                else:
                    raise ValueError(
                        f'Expected nb_tasks<=3 if dataset_list = {self.dataset_list}, but got nb_tasks={self.nb_tasks}')

            else:

                simple_datasets = self.create_splited_datasets(self.phase, data_name, len(indices))
            logging.info(simple_datasets)
            for index, task_index in enumerate(indices):
                simple_dataset = simple_datasets[index]
                dataloader = DataLoaderX(simple_dataset, batch_size=self.opt.batch_size, shuffle=True,
                                         num_workers=self.opt.num_workers,
                                         pin_memory=False if self.opt.num_workers < 4 else True)
                self.task2dataset[task_index] = dataloader, simple_dataset

            if len(indices) == self.nb_tasks:  # only run for a loop
                return

    def __len__(self):
        return self.nb_tasks

    def __getitem__(self, task_index):

        dataloader, simple_dataset = self.task2dataset[task_index]

        simple_dataset.reset_relabels()

        return dataloader

    def create_splited_datasets(self, phase, dataset_name, nb_to_tasks) -> '[SimpleDataset]':
        """create task_datasets on <dataset_name>, which has split into number of <nb_tasks>

        Return Dataloader of (SimpleDataset_of_task0, SimpleDataset_of_task1, ...)

        """

        dataset = find_dataset_using_name(dataset_name)(self.opt, phase)

        if nb_to_tasks == 1:
            targets = list(range(len(dataset.labels)))
            simple_dataset = SimpleDataset(dataset, targets, shuffle=False)
            return [simple_dataset]

        labelsOnTask = dataset.split2n_on_tasks(nb_to_tasks)
        # logging.debug(f'dataset {dataset_name} has been convert into {labelsOnTask}')
        simple_datasets = []
        for task_index, labels in enumerate(labelsOnTask):
            data_indices = flat_iterators((dataset(label) for label in labels))
            targets = [dataset.label2target[label] for label in labels]
            simple_dataset = SimpleDataset(dataset, targets, data_indices, shuffle=False)
            simple_datasets.append(simple_dataset)

        return simple_datasets
