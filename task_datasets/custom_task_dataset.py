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

import torch
from torch.utils.data import DataLoader

from datasets import SimpleDataset, find_dataset_using_name
from task_datasets import dataname2taskindex
from task_datasets.base_task_dataset import BaseTaskDataset
from utils.util import flat_iterators, is_new_distributed_avaliable


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
        parser.add_argument('--dataset_list', nargs='+', type=str, default=['mnist1_1', 'mnist_2', 'mnist_3'],
                            help='chooses which any task_datasets are loaded. ')
        parser.add_argument('--num_classes', type=list, default=[4, 4, 2],
                            help='chooses which any task_datasets are loaded. ')

        return parser

    def _build_task_dataset(self):
        """build <self.task2dataset> """
        dataname2taskIndices = dataname2taskindex(self.dataset_list)

        # build task2dataset
        self.task2dataset = [SimpleDataset] * self.nb_tasks
        for data_name, indices in dataname2taskIndices.items():
            datasets_gen = self.create_splited_datasets(self.phase, data_name)
            for task_index in indices:
                try:
                    self.task2dataset[task_index] = next(datasets_gen)
                except StopIteration:
                    logging.error(f'Please check function <create_splited_datasets>')
                    exit(0)

    def __len__(self):
        return self.nb_tasks

    def __getitem__(self, task_index):

        dataloader, simple_dataset = self.task2dataset[task_index]

        simple_dataset.reset_relabels()

        # print(f'task {task_index}: ',simple_dataset._dataset.relabels)
        return dataloader

    def create_splited_datasets(self, phase, dataset_name) -> 'DataLoader':
        """create task_datasets on <dataset_name>, which has split into number of <nb_tasks>

        Return Dataloader of (SimpleDataset_of_task0, SimpleDataset_of_task1, ...)

        """
        dataset = find_dataset_using_name(dataset_name)(self.opt, phase)
        labelsOnTask = dataset.split2n_on_tasks(self.nb_tasks)
        for task_index, labels in enumerate(labelsOnTask):
            data_indices = flat_iterators((dataset(label) for label in labels))
            simple_dataset = SimpleDataset(data_indices, dataset, labels, shuffle=True)
            if is_new_distributed_avaliable(self.opt):
                data_sampler = torch.utils.data.distributed.DistributedSampler(simple_dataset)
                data_loader = DataLoader(simple_dataset, batch_size=self.opt.batch_size,
                                         num_workers=self.opt.num_workers,
                                         sampler=data_sampler)
            else:
                # for old-distributed-data and none-distributed-data
                data_loader = DataLoader(simple_dataset, batch_size=self.opt.batch_size, shuffle=True,
                                         num_workers=self.opt.num_workers,
                                         pin_memory=False if self.opt.num_workers < 4 else True)
            yield data_loader, simple_dataset
