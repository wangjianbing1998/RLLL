# encoding=utf-8
'''
@File    :   util.py    
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
        os.makedirs(path)
        logging.info(f'Make Directory {path}')


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
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True


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
        return f'MultiOutput({self.nb_outputs}: [{["(" + task_output.shape + ")" for task_output in self.task_outputs]}])'

    def __init__(self, task_outputs: []):
        self.task_outputs = task_outputs
        self.nb_outputs = len(self.task_outputs)

    def __getitem__(self, task_index):
        return self.task_outputs[task_index]

    def __setitem__(self, task_index, output):
        self.task_outputs[task_index] = output

    def __len__(self):
        return self.nb_outputs

    def cuda(self, device):
        self.task_outputs = [task_output.cuda(device) for task_output in self.task_outputs]


class MatrixItem(object):
    """For per task"""

    def __repr__(self):
        return f'loss={self.loss},acc={self.accuracy}'

    def __init__(self, preds=None, gts=None, loss_criterion=None):
        self.accuracy = 0
        self.loss = 0
        if preds is not None and gts is not None:
            self.accuracy = self.__cal_accuracy(preds, gts)
            loss = loss_criterion(preds, gts)
            self.loss = loss.item()

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

    def __cal_accuracy(self, preds: torch.Tensor, gts: torch.Tensor):
        """calculate the accuracy on predictions and ground-truths

        preds can be (batch_size,) or (batch_size,num_class)
        """
        arg_maxs = preds
        if len(preds.shape) == 2:
            (max_vals, arg_maxs) = torch.max(preds.data, dim=1)
        gts.squeeze_()
        num_correct = torch.sum(gts == arg_maxs.long())
        acc = (num_correct * 100.0 / len(preds))
        return acc.item()


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
        nb_tasks = max(max(self.matrix.keys())) + 1
        ans = np.zeros((nb_tasks, nb_tasks))
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
        df.to_excel(path)


def is_gpu_avaliable(opt):
    return len(opt.gpu_ids) > 0 and torch.cuda.is_available()


def is_new_distributed_avaliable(opt):
    """for newly DistributedDataParallel"""
    return is_gpu_avaliable(opt) and torch.distributed.is_available() and opt.use_distributed == 'new'


def is_old_distributed_avaliable(opt):
    """for old DistributedDataParallel"""
    return is_gpu_avaliable(opt) and torch.distributed.is_available() and opt.use_distributed == 'old'


def is_distributed_avaliable(opt):
    return is_new_distributed_avaliable(opt) or is_old_distributed_avaliable(opt)


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


def relabel(targets):
    '''
    re-code the targets into range(0,len(targets))
    Args:
        targets:

    Returns:
        recoded target2index

    >>> relabel([1,5,3,6,7])
    {1: 0, 5: 1, 3: 2, 6: 3, 7: 4}
    '''
    target2index = dict([(label, index) for index, label in enumerate(targets)])
    return target2index


def load_best_ckptname(ckpts_path):
    ls = []
    for file in os.listdir(ckpts_path):
        task_index, epoch, *others = file.split('_')
        ls.append((int(task_index), float(epoch.replace('best', float('inf')))))
    if len(ls) == 0:
        logging.warning(f'NotFoundBestCheckpoint at {ckpts_path}')
        return None, None
    best_taskindex, best_epoch = max(ls)
    if best_epoch == float('inf'):
        best_epoch = 'best'
    return int(best_taskindex), str(best_epoch)
