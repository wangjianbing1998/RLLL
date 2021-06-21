import argparse
import copy
import logging
import os

import datasets
import losses
import models
import networks
import task_datasets
from VARIABLES import JOINT_TRAIN, E_TUNE, FINE_TUNE, FALWF, HOT_TUNE, WARM_TUNE, LWF, TBLWF, FFLWF, FCLWF, WALWF, \
	HALWF, FOLWF
from utils.util import rmdirs, get_log_level, seed_everything, load_best_ckptname, Checker


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
		parser.add_argument('--model_name', type=str, default="falwf",
							choices=[FINE_TUNE, WARM_TUNE, HOT_TUNE,
									 LWF,
									 FOLWF, FFLWF, FCLWF,
									 TBLWF,
									 FALWF, WALWF, HALWF,
									 E_TUNE,
									 JOINT_TRAIN],
							help='model choice from finetune|warmtune|lwf|tblwf|nllwf|fonllwf|jointtrain', )

		parser.add_argument('--name', type=str, default='',
							help='name of the experiment. opt.name=opt.model_name+"_"+opt.dataset_list')
		parser.add_argument('--device', type=str, default='cuda:0', help='cuda:0 or cpu')
		parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.rg. 0  0,1,2, 0,2. use -1 for CPU')
		parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
		parser.add_argument('--logs_dir', type=str, default='./logs',
							help='logs are saved here, such as acc/loss curve, graph of model')
		parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')

		parser.add_argument('--conffix', default='_', type=str,
							help='the connected str between {a} and {b}')

		parser.add_argument('--suffix', default='', type=str,
							help='customized suffix: opt.name = opt.name + suffix: e.rg., {opt.name}_{opt.suffix}')
		parser.add_argument('--preffix', default='', type=str,
							help='customized preffix: opt.log_filename = opt.preffix+opt.name: e.rg., {opt.log_filename}={opt.preffix}_{opt.name}')

		# for setting inputs
		parser.add_argument('--imsize', type=int, default=256)

		return parser

	def default_value(self, opt):
		opt.isTrain = self.isTrain  # train or test

		opt.name = str(opt.labeled_ratio) + "_" + opt.model_name + "_" + opt.dataset_list
		if opt.unlabeled:
			opt.name += '_unlabeled'
		# process opt.suffix
		if opt.suffix:
			opt.name = opt.name + f"_{opt.suffix}"

		opt.checkpoints_dir = os.path.join(opt.checkpoints_dir, opt.name + '/')
		opt.logs_dir = os.path.join(opt.logs_dir, opt.name + '/')

		return opt

	def gather_options(self):
		"""Initialize our parser with basic options(only once).
		Add additional model-specific and dataset-specific options.
		These options are defined in the <modify_commandline_options> function
		in model and dataset classes.
		"""
		if not self.initialized:  # check if it has been initialized
			parser = self.initialize(argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter))
			self.initialized = True
		else:
			raise ValueError(f'Parser has been initialized')

		# get the basic options
		opt, _ = parser.parse_known_args()

		# set GPUs
		os.environ['CUDA_VISIBLE_DEVICES'] = copy.copy(opt.gpu_ids)

		# modify model-related parser options

		opt, parser = self.gather_parser(models, opt.model_name, parser)
		opt, parser = self.gather_parser(losses, opt.loss_name, parser)
		opt, parser = self.gather_parser(networks, opt.net_name, parser)

		parser = datasets.get_cls('base').modify_commandline_options(parser, self.isTrain)
		opt, parser = self.gather_parser(task_datasets, opt.task_dataset_name, parser)

		# save and return the parser
		self.parser = parser
		return parser.parse_args()

	def gather_parser(self, pckgs, pckg_name, parser):
		parser = pckgs.get_cls('base').modify_commandline_options(parser, self.isTrain)
		parser = pckgs.get_cls(pckg_name).modify_commandline_options(parser)
		opt, _ = parser.parse_known_args()
		return opt, parser

	def gather_defaultvalue(self, pckgs, pckg_name, opt):
		opt = pckgs.get_cls('base').default_value(opt)
		opt = pckgs.get_cls(pckg_name).default_value(opt)
		return opt

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

		opt = self.default_value(opt)
		opt = self.gather_defaultvalue(models, opt.model_name, opt)
		opt = self.gather_defaultvalue(losses, opt.loss_name, opt)
		opt = self.gather_defaultvalue(networks, opt.net_name, opt)
		opt = datasets.get_cls('base').default_value(opt)
		opt = self.gather_defaultvalue(task_datasets, opt.task_dataset_name, opt)

		# seed and torch.backends
		seed_everything(opt.seed)

		checker = Checker(dataset_list=opt.dataset_list, labeled_ratio=opt.labeled_ratio)

		# log_filename
		if opt.log_filename != 'none':
			opt.log_filename = opt.log_filename.format(f'{opt.preffix}{opt.name}')
			if hasattr(opt, "dels") and "output" in opt.dels and os.path.isfile(opt.log_filename):
				os.remove(opt.log_filename)
			os.makedirs(os.path.dirname(opt.log_filename), exist_ok=True)
		else:
			opt.log_filename = None

		logging.basicConfig(filename=opt.log_filename,
							filemode=opt.log_filemode,
							format=opt.log_format,
							level=(get_log_level(opt.log_level)),
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

		load_taskindex, load_step, load_epoch = load_best_ckptname(opt.checkpoints_dir)

		if opt.load_taskindex == 0 and load_taskindex is not None:
			opt.load_taskindex = load_taskindex

		if opt.load_step == 1 and load_step is not None:
			opt.load_step = load_step

		if opt.load_epoch == 'best' and load_epoch is not None:
			opt.load_epoch = load_epoch

		if opt.load_epoch != 'best' and opt.epoch_start == 0:
			opt.epoch_start = int(opt.load_epoch)

		self.print_options(opt)

		# set GPUs
		str_ids = opt.gpu_ids.split(',')
		opt.gpu_ids = []
		for str_id in str_ids:
			id = int(str_id)
			if id >= 0:
				opt.gpu_ids.append(id)

		if min(opt.gpu_ids) >= 0:
			# torch.cuda.set_device(min(opt.gpu_ids))  # 这样就不会出现那种问题
			# opt.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device(
			#     'cpu')  # get device tag: CPU or GPU
			import torch
			opt.device = torch.device('cuda:0')

		self.opt = opt
		return self.opt
