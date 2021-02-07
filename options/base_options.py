import argparse
import logging
import os

import datasets
import losses
import models
import networks
import task_datasets
from util import util
from util.util import rmdirs


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
        # logging configuration
        # parser.add_argument('--log_filename', type=str, default="output/{}.txt", help='logging filename')
        parser.add_argument('--log_filename', type=str, default=None, help='logging filename')
        parser.add_argument('--log_filemode', type=str, default='a', help='logging filemode')
        parser.add_argument('--log_format', type=str, default='%(asctime)s - %(levelname)s - %(message)s',
                            help='logging format')
        parser.add_argument('--log_level', type=int, default=logging.DEBUG, help='logging level')
        # training before
        parser.add_argument('--dels', type=str, default="",
                            help='which need to be clear delete, strings contains ckpts, logs or outputs ')
        # basic parameters
        parser.add_argument('--name', type=str, default='',
                            help='name of the experiment. opt.name=opt.model_name+"_"+opt.dataset_list')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--logs_dir', type=str, default='./logs',
                            help='logs are saved here, such as acc/loss curve, graph of model')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str,
                            help='customized suffix: opt.name = opt.name + suffix: e.g., {opt.name}_{opt.suffix}')

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
        opt.isTrain = self.isTrain  # fit or test
        opt.name = opt.model_name + "_" + "-".join(opt.dataset_list)
        # process opt.suffix
        if opt.suffix:
            opt.name = opt.name + "_" + opt.suffix

        if opt.log_filename is not None:
            opt.log_filename = opt.log_filename.format(opt.name)
        logging.basicConfig(filename=opt.log_filename,
                            filemode=opt.log_filemode,
                            format=opt.log_format,
                            level=opt.log_level,
                            )
        logging.info(f'Name={opt.name:*^50}')
        # clear(delete) the directories

        opt.checkpoints_dir = os.path.join(opt.checkpoints_dir, opt.name + '/')
        opt.logs_dir = os.path.join(opt.logs_dir, opt.name + '/')

        if "logs" in opt.dels:
            rmdirs(opt.logs_dir)
        if "ckpts" in opt.dels:
            rmdirs(opt.checkpoints_dir)
        if "outputs" in opt.dels:
            os.remove(opt.log_filename)

        # prepare checkpoints_dir
        os.makedirs(opt.checkpoints_dir, exist_ok=True)
        os.makedirs(opt.logs_dir, exist_ok=True)

        self.print_options(opt)

        # set gpu ids
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids

        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)

        self.opt = opt
        return self.opt