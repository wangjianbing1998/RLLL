# encoding=utf-8
'''
@File    :   utils.py
@Contact :   jianbingxiaman@gmail.com
@License :   (C)Copyright 2020-2021, John Hopcraft Lab-CV
@Desciption : 
@Modify Time      @Author    @Version
------------      -------    --------
2021/1/21 19:04   jianbingxia     1.0    
'''
import logging
import os
import random
import shutil
import time
from collections import defaultdict
from functools import reduce
from itertools import chain
from math import ceil

import pandas as pd

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 20)
pd.set_option('precision', 2)
import numpy as np
import torch
import wrapt
from sklearn.utils import Bunch


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the tag of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad._data))
            count += 1
    if count > 0:
        mean = mean / count
    logging.debug(f"name:{name}, mean:{mean}")


def print_numpy(x: np.ndarray, val=True, shp=True):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        logging.debug('shape,', x.shape)
    if val:
        x = x.flatten()
        logging.debug('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            x.mean(), x.min(), x.max(), x.median(), x.std()))


def print_tensor(x: torch.tensor, pt=True, val=True, shp=True):
    """Print the mean, min, median, std, and size of a tensor tensor

    Args:
        x:
        val: if print the values of the tensor
        shp: if print the shape of the tensor

    Returns: None

    """

    x = x.float()
    message = ''
    if shp:
        message = str(x.shape) + '\n'
    if val:
        x = x.flatten()
        if len(x) != 1:
            message += ('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
                x.mean(), x.min(), x.max(), x.median(), x.std()))
        else:
            message += (f'one element {x[0]}')
    if pt:
        logging.debug(message)
    return message


def tensors2str(ts, verbose=False):
    res = [t.item() for t in ts]
    if verbose:
        logging.debug(res)
    return res


def tensor2im(input_image, imtype=np.uint8, need_scaling=True):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
        need_scaling (bool)  --  if the input image scaling between -1 and 1 or not
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the _data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))

        transpose_image = np.transpose(image_numpy, (1, 2, 0))
        if need_scaling:
            image_numpy = (transpose_image + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        else:
            image_numpy = transpose_image
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logging.info(f'Make Directory {path}')


def rmfile(path):
    if os.path.exists(path):
        os.remove(path)
        logging.warning(f'Delete File {path}')


def rmfiles(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            rmfile(path)
    else:
        rmfile(paths)


def rmdir(path):
    """remove a single directory if it exist

    Parameters:
        path (str) -- a single directory path
    """
    if os.path.exists(path):
        shutil.rmtree(path)
        logging.warning(f'Delete Directory {path}')


def rmdirs(paths):
    """remove directories if they exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            rmdir(path)
    else:
        rmdir(paths)


def seed_everything(seed):
    """set seed for everything
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # if benchmark=True, deterministic will be False
    # torch.backends.cudnn.deterministic = False


def split2numclasses(A, n) -> []:
    """split list <A> into n splits

    >>> split2numclasses(range(10),3)
    [4, 4, 2]
    """
    res = []
    m = int(ceil(len(A) / float(n)))
    s = 0
    S = len(A)
    for i in range(n):
        res.append(min(m, S - s))
        s += m
    return res


def split2n(A, n) -> []:
    """split list <A> into n splits"""
    res = []
    m = int(ceil(len(A) / float(n)))
    for i in range(n):
        res.append(A[i * m:(i + 1) * m])
    return res


def flat_iterators(A):
    """flatten several iterators into only a iterator"""
    return list(chain(*A))


class MultiOutput(object):
    """Multi Output Classifier, network.forward(input) -> <MultiOutputClassifier>

    self.nb_tasks

    self.task_outputs: output per task, [(batch_size,num_class) for num_class in self.num_classes]
    self.output: total output, [batch_size,num_class*nb_tasks]
    """

    def __repr__(self):
        return f'MultiOutput({self.nb_outputs}: [{["(" + str(task_output.shape) + ")" for task_output in self.task_outputs]}])'

    def __init__(self, task_outputs: []):
        self.task_outputs = task_outputs
        self.nb_outputs = len(self.task_outputs)
        self.is_cuda = False

    def __getitem__(self, task_index):
        return self.task_outputs[task_index]

    def __setitem__(self, task_index, output):
        self.task_outputs[task_index] = output

    def __len__(self):
        return self.nb_outputs

    def cuda(self, device, non_blocking=True):
        if not self.is_cuda:
            self.is_cuda = True
            self.task_outputs = [task_output.to(device, non_blocking=non_blocking) for task_output in self.task_outputs]
        return self

    def __eq__(self, other):
        if hasattr(self, 'task_outputs') and hasattr(other, "task_outputs"):
            return [task_output_1 == task_output_2 for task_output_1, task_output_2 in
                    zip(self.task_outputs, other.task_outputs)]
        return self is None and other is None


class MatrixItem(object):
    """For per task"""

    def __repr__(self):
        return f'loss={self.loss},acc={self.accuracy}'

    def __init__(self, preds=None, gts=None, loss_criterion=None):
        self.accuracy = 0
        self.loss = 0
        if preds is not None and gts is not None:
            self.accuracy = self.cal_accuracy(preds, gts)
            loss = loss_criterion(preds, gts)
            self.loss = loss.detach()

    def __add__(self, other):
        accuracy = self.accuracy + other.accuracy
        loss = self.loss + other.loss
        return MatrixItem()(accuracy, loss)

    def __lt__(self, other):
        return self.accuracy < other.accuracy

    def __truediv__(self, other: 'Scaler'):
        return MatrixItem()(self.accuracy / other, self.loss / other)

    def __call__(self, accuracy, loss, **kwargs):
        """set self.accuracy directly"""
        self.accuracy = accuracy
        self.loss = loss
        return self

    @staticmethod
    def cal_accuracy(preds: torch.Tensor, gts: torch.Tensor):
        """calculate the accuracy on predictions and ground-truths

        preds can be (batch_size,) or (batch_size,num_class)

        >>> matrixitem=MatrixItem()
        >>> targets=torch.LongTensor([1,2,6,4,5])
        >>> preds=torch.FloatTensor([1,2,3,4,5])
        >>> matrixitem.cal_accuracy(preds,targets)
        tensor(80.)
        >>> preds=torch.FloatTensor([
        ...            [0,1,0,0,0,0],
        ...            [0,0,1,0,0,0],
        ...            [0,0,0,1,0,0],
        ...            [0,0,0,0,1,0],
        ...            [0,0,0,0,0,1],
        ...            ])
        >>> matrixitem.cal_accuracy(preds,targets)
        tensor(80.)
        """
        arg_maxs = preds
        if len(preds.shape) == 2:
            (max_vals, arg_maxs) = torch.max(preds.detach(), dim=1)
        num_correct = torch.sum(gts.squeeze() == arg_maxs.long())
        acc = (num_correct * 100.0 / len(preds))
        return acc.detach()


class TestMatrix(object):
    """Measure for testing

    self.matrix:{(train_index,test_index):MatrixItem}
    """

    def __init__(self):
        self.matrix = defaultdict(MatrixItem)

    def __getitem__(self, item: tuple):
        """item is (train_index,test_index), which means after training task indexed with <train_index>, and then test the model at task indexed with <test_index>"""
        return self.matrix[item]

    def __setitem__(self, item, *values):
        """ set the item in matrix
        values can be <MatrixItem> or <preds,gts>

        Raise:
            ValueError, if values is not <accuracy> or <predictions,ground-truths>
        """
        if len(values) == 1 and isinstance(values[0], MatrixItem):
            self.matrix[item] = values[0]
        elif len(values) == 2 and isinstance(values[0], float) and isinstance(values[1], float):
            self.matrix[item] = MatrixItem(*values)
        else:
            raise ValueError(f'Expected values is <MatrixItem> or <predictions,gound-truths>, but got {values}')

    def __get_df_matrix(self):
        nb_tasks = max(max(self.matrix.keys()))
        ans = np.zeros((nb_tasks + 1, nb_tasks + 1))
        for (x, y), item in self.matrix.items():
            ans[x, y] = item.accuracy
        df = pd.DataFrame(ans)
        return df

    def save_matrix(self, path):
        """only can save the .xlsx"""
        dirname = os.path.dirname(path)
        if dirname != '':
            mkdir(dirname)
        df = self.__get_df_matrix()
        df.to_excel(path, index=False)


def is_gpu_avaliable(opt):
    return len(opt.gpu_ids) > 0 and torch.cuda.is_available()


def unsqueeze0(data: Bunch):
    data.image = data.image.unsqueeze(0)
    data.targets = data.targets.unsqueeze(0)


def un_onehot(target: torch.Tensor):
    if len(target.shape) == 0:
        target = torch.unsqueeze(target, dim=0)
        return target.long()
    elif len(target.shape) == 1:
        return target.long()
    elif len(target.shape) == 2:
        return target.argmax(dim=1).long()
    else:
        raise ValueError(f'Expected 1<=target.size<=2 ,but got {len(target.shape)}')


def log(level="info"):
    level = get_log_level(level)

    @wrapt.decorator
    def _wrapper(func, instance, args, kwargs):
        start_time = time.time()
        logging.log(level, f'Enter {func.__name__}')
        res = func(*args, **kwargs)
        logging.log(level, f'{func.__name__} runs {time.time() - start_time} sec')
        return res

    return _wrapper


def get_log_level(level: str) -> int:
    level = level.upper()
    if level == 'DEBUG':
        level = logging.DEBUG
    elif level == 'INFO':
        level = logging.INFO
    elif level in ['WARNING', 'WARN']:
        level = logging.WARNING
    elif level == 'ERROR':
        level = logging.ERROR
    elif level in ['CRITICAL', 'FATAL']:
        level = logging.CRITICAL
    else:
        raise ValueError(f'Expected debug|info|warning|warn|error|critical|fatal, but got {level}')
    return level


def retarget(labels):
    '''
    re-code the targets into range(0,len(targets))
    Args:
        labels:

    Returns:
        recoded target2index

    >>> retarget([1,5,3,6,7])
    {1: 0, 5: 1, 3: 2, 6: 3, 7: 4}
    '''
    label2index = dict([(label, index) for index, label in enumerate(labels)])
    return label2index


def load_best_ckptname(ckpts_path):
    ls = []
    for file in os.listdir(ckpts_path):
        if file.endswith('.pth'):
            logging.info(f'loading file {file}')
            task_index, step, epoch, *others = file.split('_')
            ls.append((int(task_index), int(step), int(epoch) if epoch != 'best' else float('inf')))
    if len(ls) == 0:
        logging.warning(f'NotFoundBestCheckpoint at {ckpts_path}')
        return None, None, None
    best_taskindex, best_step, best_epoch = max(ls)
    if best_epoch == float('inf'):
        best_epoch = 'best'
    return int(best_taskindex), int(best_step), best_epoch


class Checker(object):

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if k == 'dataset_list':
                self.check_dataset_list(v)
            elif k == 'labeled_ratio':
                self.check_labeled_ratio(v)
            else:
                raise ValueError(f'UnExpected checker {k}')

    def check_dataset_list(self, dataset_list):
        from datasets import dataset_names
        for index, data_name in enumerate(dataset_list):
            data_name = data_name.lower()
            if '_' in data_name:
                data_name = data_name[:data_name.index('_')]

            if data_name not in dataset_names:
                raise ValueError(f'Dataset named {data_name} not found!')

    def check_labeled_ratio(self, v):
        v = float(v)
        if v < 0 or v > 1:
            raise ValueError(f'Expected labeled_ratio between 0 and 1, but got {v}')


def exec_times(times):
    def _exec_times(target):
        _times = [0]

        def _warp(*args, **kwargs):
            if _times[0] >= times:
                logging.warning(
                    f'func {target.__name__} has been called {times} times, it will not be called and return None')
                return None
            ans = target(*args, **kwargs)
            _times[0] += 1
            return ans

        return _warp

    return _exec_times


class ListDict(object):
    def __init__(self, *keys):
        self.dict = dict((k, []) for k in keys)

    def load_data(self, D: 'Dict'):
        assert isinstance(D, dict), f'Expected data in load_data is dict, but got {type(D)}'
        ListDict.check_dict(D, error_raise=True)
        self.dict = D

    def insert_one_dict(self, item: 'dict,Bunch'):
        for k in self.dict.keys():
            if k in item:
                self.dict[k].append(item[k])
            else:
                raise ValueError(f'Expected key in dict.keys({self.dict.keys()}), but got {item.keys()}')

    def insert_one_kwargs(self, **kwargs):
        self.insert_one_dict(dict(**kwargs))

    def insert_items(self, items: 'dict,Bunch'):

        ListDict.check_dict(items, error_raise=True)
        for k in self.dict.keys():
            if k in items:
                self.dict[k].extend(items[k])
            else:
                raise ValueError(f'Expected key in dict.keys({self.dict.keys()}), but got {items.keys()}')

    def convert2df(self):
        return pd.DataFrame(self.dict)

    @staticmethod
    def check_dict(dict, error_raise=False):
        """check the len of list on dict is all same or not"""

        len_of_v = (len(v) for k, v in dict.items())
        res = len(set(len_of_v)) == 1
        if not res and error_raise:
            raise ValueError(f'Expected the len of list on dict is all same, but got {len_of_v}')
        return res


def random_choice(choices, rd):
    """

    >>> for _ in range(10):
    ...     random_choice([1,2,3],[.1,.1,.8])

    Args:
        choices:
        ps:

    Returns:

    """
    assert len(choices) == len(ps), f'Expected len(choices) == len(rd), but got choices={choices}, rd={ps}'
    return np.random.choice(choices, p=ps)


def my_sum(*args):
    return reduce(lambda x, y: x + y, *args)


def get_basedirname(path, delimeter='/'):
    dirs = path.split(delimeter)
    if len(dirs) == 1:
        dirs = path.split('\\')

    index = len(dirs) - 1
    while index >= 0:
        if dirs[index] != '':
            return dirs[index]
        index -= 1
    return None


def judge_tensor_value_is_long(T):
    """
    >>> judge_tensor_value_is_long(torch.tensor([[ 4.2, 32.3, 32.4]]))
    False
    >>> judge_tensor_value_is_long(torch.tensor([[ 4., 32., 32.]]))
    True


    """
    if isinstance(T, torch.LongTensor) or isinstance(T, torch.IntTensor):
        return True
    t = T == T.long().float()
    return bool(t.max() == 1) and bool(t.min() == 1)
