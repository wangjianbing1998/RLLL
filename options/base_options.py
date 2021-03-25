import argparse
import copy
import logging
import os

import torch

import datasets
import losses
import models
import networks
import task_datasets
from util.util import rmdirs, get_log_level, seed_everything, load_best_ckptname, Checker


class BaseOptions(object):
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class, model class , loss class and network class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.isTrain = None

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""

        # Multi-GPUs Distributed
        parser.add_argument('--use_distributed', type=str, default='new', choices=['new', 'old', 'none'],
                            help='if use newly distributed GPUs calculating for apex.DistributedDataParallel, or old for `DataParallelModel` or none for DataParallel')
        parser.add_argument('--init_method', type=str, default="tcp://127.0.0.1:46622",
                            help='the main machine or process ip:port, all machine or process is same as the main one')
        parser.add_argument('--rank', type=int, default=0, help='rank of current machine or process')
        parser.add_argument('--word_size', type=int, default=1, help='the number of machine or process')

        # logging configuration

        parser.add_argument('--log_filemode', type=str, default='a', help='logging filemode')
        parser.add_argument('--log_format', type=str, default='%(asctime)s - %(levelname)s - %(message)s',
                            help='logging format')
        parser.add_argument('--log_level', type=str, default="debug", help='logging level')

        # basic parameters
        parser.add_argument('--name', type=str, default='',
                            help='name of the experiment. opt.name=opt.model_name+"_"+opt.dataset_list')
        parser.add_argument('--device', type=str, default='cuda:0', help='cuda:0 or cpu')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--logs_dir', type=str, default='./logs',
                            help='logs are saved here, such as acc/loss curve, graph of model')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')

        parser.add_argument('--conffix', default='_', type=str,
                            help='the connected str between {a} and {b}')

        parser.add_argument('--suffix', default='', type=str,
                            help='customized suffix: opt.name = opt.name + suffix: e.g., {opt.name}_{opt.suffix}')
        parser.add_argument('--preffix', default='', type=str,
                            help='customized preffix: opt.log_filename = opt.preffix+opt.name: e.g., {opt.log_filename}={opt.preffix}_{opt.name}')

        # for setting inputs
        parser.add_argument('--imsize', type=int, default=256)

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = self.initialize(argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter))

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options

        model_name = opt.model_name
        parser = models.get_option_setter('base')(parser, self.isTrain)
        parser = models.get_option_setter(model_name)(parser)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify loss-related parser options
        loss_name = opt.loss_name
        parser = losses.get_option_setter('base')(parser, self.isTrain)
        parser = losses.get_option_setter(loss_name)(parser)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify network-related parser options
        net_name = opt.net_name
        parser = networks.get_option_setter('base')(parser, self.isTrain)
        parser = networks.get_option_setter(net_name)(parser)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify task-dataset-related parser options
        dataset_name = opt.task_dataset_name
        parser = datasets.get_option_setter('base')(parser, self.isTrain)
        parser = task_datasets.get_option_setter('base')(parser, self.isTrain)
        parser = task_datasets.get_option_setter(dataset_name)(parser)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        logging.info(message)

        # save to the disk
        file_name = os.path.join(opt.checkpoints_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        # seed and torch.backends
        seed_everything(opt.seed)

        # format input
        opt.use_distributed = opt.use_distributed.lower()
        opt.log_filename = opt.log_filename.strip().lower()

        # initialize num_classes, get info from dataset_list, and it will be used in networks

        checker = Checker(dataset_list=opt.dataset_list)

        opt.num_classes = task_datasets.get_num_classes_by_data_list(opt.dataset_list)

        opt.isTrain = self.isTrain  # train or test
        opt.name = opt.model_name + "_" + "-".join(opt.dataset_list)
        # process opt.suffix
        if opt.suffix:
            opt.name = opt.name + "_" + opt.suffix

        # log_filename
        if opt.log_filename != 'none':
            opt.log_filename = opt.log_filename.format(f'{opt.preffix}_{opt.name}')
            if hasattr(opt, "dels") and "output" in opt.dels and os.path.isfile(opt.log_filename):
                os.remove(opt.log_filename)
            os.makedirs(os.path.dirname(opt.log_filename), exist_ok=True)
        else:
            opt.log_filename = None
        opt.checkpoints_dir = os.path.join(opt.checkpoints_dir, opt.name + '/')
        opt.logs_dir = os.path.join(opt.logs_dir, opt.name + '/')

        log_level = get_log_level(opt.log_level)
        logging.basicConfig(filename=opt.log_filename,
                            filemode=opt.log_filemode,
                            format=opt.log_format,
                            level=log_level,
                            )
        logging.info(f'Name={opt.name:*^50}')

        # clear(delete) the directories
        if hasattr(opt, "dels") and "log" in opt.dels and os.path.isdir(opt.logs_dir):
            rmdirs(opt.logs_dir)

        if hasattr(opt, "dels") and "ckpt" in opt.dels and os.path.isdir(opt.checkpoints_dir):
            rmdirs(opt.checkpoints_dir)

        os.makedirs(opt.checkpoints_dir, exist_ok=True)
        os.makedirs(opt.logs_dir, exist_ok=True)
        os.makedirs(opt.result_dir, exist_ok=True)

        load_taskindex, load_epoch = load_best_ckptname(opt.checkpoints_dir)

        if opt.load_taskindex == 0 and load_taskindex is not None:
            opt.load_taskindex = load_taskindex
        if opt.load_epoch == 'best' and load_epoch is not None:
            opt.load_epoch = load_epoch

        self.print_options(opt)

        # set GPUs
        gpu_ids_str = copy.copy(opt.gpu_ids)
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)

        # set gpu ids
        os.environ[
            "CUDA_VISIBLE_DEVICES"] = gpu_ids_str  # 当指定这个的时候，比如gpu_ids=[1,2], 那系统会把1当做0,2当做1，则在DataParallel中的device_ids就是[0,1]才可以，而不能是[1,2]
        if min(opt.gpu_ids) >= 0:
            # torch.cuda.set_device(min(opt.gpu_ids))  # 这样就不会出现那种问题
            # opt.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device(
            #     'cpu')  # get device tag: CPU or GPU

            opt.device = torch.device('cuda:0')

        self.opt = opt
        return self.opt
