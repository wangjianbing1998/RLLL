# encoding=utf-8
'''
@File    :   visualizer.py    
@Contact :   jianbingxiaman@gmail.com
@License :   (C)Copyright 2020-2021, John Hopcraft Lab-CV
@Desciption : 
@Modify Time      @Author    @Version
------------      -------    --------
2021/1/24 17:14   jianbingxia     1.0    
'''
import os
from collections import defaultdict

import pandas as pd
import seaborn as sns

from utils.util import ListDict

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 20)
pd.set_option('precision', 2)
import matplotlib.pyplot as plt
import torch
from tensorboardX import SummaryWriter

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


class Visualizer(object):
    def __init__(self, opt):
        self.opt = opt
        self.logs_dir = opt.logs_dir
        self.model_name = opt.model_name

        self.summary_writer = SummaryWriter(log_dir=self.logs_dir, comment=self.model_name)

    def setup(self):
        # TODO regular setup: if opt.<continue_train>, add_scalar into the previous scalars.

        pass

    def add_scalar(self, *args):
        self.summary_writer.add_scalar(*args)

    def add_losses(self, losses, epoch):
        """add losses for FloatTensor or list of FloatTensor

        such as
        losses_with_lambda: [losses_with_lambda[index] for index in range(nb_tasks)]
        losses_without_lambda: [losses_without_lambda[index] for index in range(nb_tasks)]
        loss_total: FloatTensor

        """
        from utils.util import is_gpu_avaliable
        for loss_name, loss in losses.items():
            if isinstance(loss, torch.FloatTensor if not is_gpu_avaliable(self.opt) else torch.cuda.FloatTensor):
                self.summary_writer.add_scalar("losses", loss, epoch)
            elif isinstance(loss, list):
                loss = dict([(str(index), l) for index, l in enumerate(loss)])
                self.summary_writer.add_scalars("losses", loss)
            else:
                raise TypeError(f"loss must be float or list, but got {type(loss)}")

    def add_scalars(self, *args):
        self.summary_writer.add_scalars(*args)

    def add_graph(self, *args):
        self.summary_writer.add_graph(*args)

    def reset(self):
        pass

    def __del__(self):
        self.close()

    def close(self):
        self.summary_writer.close()


class ResultVisualizer(object):
    MODEL_STYLE = {
        'tblwf':'k*',
        'finetune':'b.',
        'warmtune':'go',
        'hottune':'rv',
        'lwf':'y^',
        'folwf':'m<',
        'nllwf':'c1',
        'fonllwf':'mp',
        'jt':'pink|',
        'falwf':'r^',
    }

    KEYS = ['tblwf', 'finetune', 'warmtune', 'hottune', 'lwf', 'folwf', 'nllwf', 'fonllwf', 'falwf', 'jt']
    assert len(KEYS) == len(MODEL_STYLE)

    def __init__(self, dir_path, result_path):

        self.result_path = result_path
        self.defaultdict = defaultdict(list)
        models = set()
        dfs = defaultdict(list)

        for file in os.listdir(dir_path):
            if file.endswith('.csv'):
                model, dataset_list = file.split('_')
                datasetlist_str = dataset_list.replace('.csv', '')
                dataset_list = datasetlist_str.split('-')
                file = os.path.join(dir_path, file)
                df = pd.read_csv(file)
                dfs[datasetlist_str].append((model, df))
                nb_tasks = len(dataset_list)
                ys = [df[str(task)].values.tolist() for task in range(1, nb_tasks + 1)]
                self.defaultdict[datasetlist_str].append((model, ys))
                models.add(model)
        for datasetlist_str, model_ys in self.defaultdict.items():
            dataset_list = datasetlist_str.split('-')

            # fig = plt.figure(figsize=(10, 10))
            # self.plot_line(dataset_list, datasetlist_str, fig, model_ys, models)
            # plt.savefig(datasetlist_str + '_line.png')
            # plt.show()

            fig = plt.figure(figsize=(10, 10))
            self.plot_bar(dataset_list, datasetlist_str, fig, model_ys)
            plt.savefig(datasetlist_str + '_bar.png')
            plt.show()

            # df = dfs[datasetlist_str]
            # df.sort(key=lambda x:self.KEYS.index(x[0]))
            # res = self.concat_df(df)
            # res.to_csv(f"{datasetlist_str}_" + self.result_path)

    def plot_line(self, dataset_list, datasetlist_str, fig, model_ys, models):
        axes = fig.subplots(len(dataset_list), 1, sharex=True)
        for model, ys in model_ys:
            self.ax_plot_line(axes, ys, y_labels=dataset_list, model=model)

        ax = axes[-1]
        ax.set_xlabel('Trained Tasks')
        x_ticks = ['Random'] + dataset_list
        if 'tblwf' in models:
            x_ticks += dataset_list[-2::-1]
        ax.set_xticks(range(len(x_ticks)))
        ax.set_xticklabels(x_ticks)

        axes[0].legend(loc='best')
        axes[0].set_title(datasetlist_str)

    def plot_bar(self, dataset_list, datasetlist_str, fig, model_ys):
        test_score_df = ListDict('model', 'train_dataset', 'test_dataset', 'test_score')
        for model, ys in model_ys:
            for test_index, Y in enumerate(ys):
                test_dataset = dataset_list[test_index]
                if 'tblwf' == model:
                    # Reconstruct the `Random, TASK_1, Task_2, Task_3`
                    Y = [Y[0]] + Y[-1:- len(dataset_list) - 1:-1]
                for train_index, y in enumerate(Y):
                    if train_index == 0:
                        train_dataset = 'Random'
                    else:
                        train_dataset = dataset_list[train_index - 1]

                    test_score_df.insert_one_kwargs(model=model, train_dataset=train_dataset, test_dataset=test_dataset,
                                                    test_score=y)
        test_score_df = test_score_df.convert2df()

        axes = fig.subplots(len(dataset_list), 1, sharex=True)
        for index, test_dataset in enumerate(dataset_list):
            df = test_score_df[test_score_df['test_dataset'] == test_dataset]
            df.drop(columns=['test_dataset'], inplace=True)
            self.ax_plot_bar(axes[index], df, test_dataset)

        ax = axes[-1]
        ax.set_xlabel('Trained Tasks')
        # x_ticks = ['Random'] + dataset_list

        # ax.set_xticks(range(len(x_ticks)))
        # ax.set_xticklabels(x_ticks)

        axes[0].legend(loc='best')
        axes[0].set_title(datasetlist_str)

    def concat_df(self, dfs):
        res = pd.DataFrame({"1":[], "2":[], "3":[]})

        for model, df in dfs:
            print(model)
            res = pd.concat([res, df], axis=0, )

        return res

    def ax_plot_line(self, axes, ys, y_labels, model):

        assert len(y_labels) == len(ys), f'Expected len({ys}) == len({y_labels}), but got ys={ys}, y_labels={y_labels}'
        y_style = self.MODEL_STYLE[model]

        nb_tasks = len(ys)

        for task_index in range(nb_tasks):
            ax = axes[task_index]
            ax.plot(range(len(ys[task_index])), ys[task_index], y_style + '-', label=model)

            ax.set_ylabel(y_labels[task_index])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    def ax_plot_bar(self, ax, df, test_dataset):

        sns.barplot(ax=ax, x='train_dataset', y='test_score', hue='model', data=df)
        ax.set_title(f'Test in {test_dataset}')
        ax.set_xlabel('')
        ax.get_legend().set_visible(False)


if __name__ == '__main__':
    ResultVisualizer(r'../train_results', 'results.csv')
