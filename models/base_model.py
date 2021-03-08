import logging
import os
from abc import ABC
from collections import OrderedDict

import torch
from apex import amp
# torch.multiprocessing.set_start_method('spawn')
from apex.parallel import DistributedDataParallel
from torch.nn import DataParallel

from networks import get_scheduler, _init_weights
from task_datasets import PseudoData
from util.data_parallel_util import DataParallelModel, DataParallelCriterion
from util.util import MatrixItem, is_gpu_avaliable, is_new_distributed_avaliable, is_old_distributed_avaliable,\
    is_distributed_avaliable, MultiOutput


class BaseModel(ABC):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain

        self.save_dir = opt.checkpoints_dir  # save all the checkpoints to save_dir

        self.loss_names = []
        self.net_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

        self.shared_cnn_layers = []
        self.shared_fc_layers = []
        self.other_layers = []

        self._plus_other_loss = False  # need to plus the other loss or not,
        self._need_backward = False  # need to backward to optimize the previous task

        self.target, self.image = None, None
        self.output = None
        self._continued_task_index = 0  # continued trained task index

        self.network_printed = False

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """

        return parser

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.load_taskindex, opt.load_epoch)
        self.print_networks(repeat=False)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.net_names:
            if isinstance(name, str):
                net = getattr(self, "net_" + name)
                net.eval()

    def fit(self):
        """Make models train mode during train time"""
        for name in self.net_names:
            if isinstance(name, str):
                net = getattr(self, "net_" + name)
                net.train()

    def train(self, task_index):

        self.fit()  # set train mode
        self.forward()  # first call forward to calculate intermediate results
        self.optimize_parameters(task_index)

    def test(self, visualizer=None):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        self.eval()  # set eval mode
        with torch.no_grad():
            self.forward()
            if visualizer is not None:
                self.compute_visuals(visualizer)

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        logging.info('learning rate = %.7f' % lr)

    def get_current_losses(self):
        """Return traning losses_without_lambda / errors. fit.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = getattr(self, name)  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, task_index, epoch):
        """Save all the networks to the disk.

        Parameters:
            task_index (int) -- current trained task_index
            epoch (Union[int,str]) -- current epoch; used in the file tag '%s_%s_net_%s.pth' % (task_index,epoch, tag)
        """
        for name in self.net_names:
            if isinstance(name, str):
                save_filename = '%s_%s_net_%s.pth' % (task_index, epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, "net_" + name)
                if is_gpu_avaliable(self.opt):
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.to(device=self.opt.device)
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, taks_index, epoch):
        """Load all the networks from the disk.

        Parameters:
            task_index (int) -- current trained task_index
            epoch (int or str) -- current epoch or best epoch; used in the file tag '{taks_index}_{epoch}_net_{name}.pth' % (epoch, tag)
        """
        for name in self.net_names:
            if isinstance(name, str):
                load_filename = f'{taks_index}_{epoch}_net_{name}.pth'
                load_path = os.path.join(self.save_dir, load_filename)
                if not os.path.exists(load_path):
                    logging.warning(f'checkpoint path {load_path} is not exists, please checkout it!')
                else:
                    net = getattr(self, "net_" + name)
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.module
                    logging.info('loading the model from %s' % load_path)
                    state_dict = torch.load(load_path, map_location=str(self.opt.device))
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata

                    # patch InstanceNorm checkpoints prior to 0.4
                    for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                        self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                    net.load_state_dict(state_dict)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and\
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and\
                    (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def print_networks(self, repeat=True):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        if self.network_printed and not repeat:
            return
        self.network_printed = True
        logging.info('---------- Networks initialized -------------')
        for name in self.net_names:
            if isinstance(name, str):
                net = getattr(self, "net_" + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                logging.info(net)
                logging.info('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        logging.info('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_matrix_item(self, task_index) -> MatrixItem:
        """for current task_index, after calling set_data() and forward(), then call <get_matrix_item()> will get MatrixItem """
        if self.output is None or self.target is None:
            raise ValueError(
                f"Expected model.set_data(data) and model.forward() be called before model.get_matrix_item(), but not called {'forward()' if self.output is None else ''} {'and set_data()' if self.target is None else ''}")
        return MatrixItem(self.output[task_index], self.target, self.loss_criterion)

    def init_net_optimizer_with_apex(self, opt, net, optimizer, criterion):
        """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
        Parameters:
            net (network)      -- the network to be initialized
            init_type (str)    -- the tag of an initialization method: normal | xavier | kaiming | orthogonal
            gain (float)       -- scaling factor for normal, xavier and orthogonal.
            gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

        Return an initialized network.
        """

        if len(opt.gpu_ids) > 0:
            assert (torch.cuda.is_available())

            if is_new_distributed_avaliable(opt):
                # bn sync, only if distributed is available
                from apex.parallel import convert_syncbn_model
                net = convert_syncbn_model(net)

            if is_distributed_avaliable(opt):
                net.to(device=self.opt.device)
                net, optimizer = amp.initialize(net, optimizer,
                                                opt_level="O1")  # here, the net can be replaced with list of nets
                if is_new_distributed_avaliable(opt):
                    # for newly distributed data parallel
                    torch.distributed.init_process_group(backend='nccl', init_method=opt.init_method, rank=opt.rank,
                                                         world_size=opt.word_size)
                    net = DistributedDataParallel(net, delay_allreduce=True)  # must be after amp.initialize

                elif is_old_distributed_avaliable(opt):
                    # for old distributed data parallel
                    net = DataParallelModel(net, device_ids=opt.gpu_ids)
                    criterion = DataParallelCriterion(criterion, device_ids=opt.gpu_ids)
                    criterion = criterion.module
                else:
                    raise ValueError(f'Expected normal distributed `use_distributed`, but got {opt.use_distributed}')
            else:
                # for none distributed data parallel, only DataParallel
                net = DataParallel(net, device_ids=opt.gpu_ids)
                net.to(device=self.opt.device)

        else:
            # use CPU
            net, optimizer = amp.initialize(net, optimizer, opt_level="O1")  # 这里多个net就用列表

        _init_weights(net, opt.init_type, init_gain=opt.init_gain)
        return net, optimizer, criterion

    def __getattr__(self, item):
        if "loss" in item and hasattr(self.loss_criterion, item):
            return getattr(self.loss_criterion, item)
        for net in self._get_all_nets():
            if "net" in item and hasattr(net, item):
                return getattr(net, item)
        raise AttributeError(f"'LwfModel' object has no attribute '{item}'")

    def set_data(self, data: PseudoData):
        """Unpack input _data from the dataloader and perform necessary pre-processing steps.
        """
        data.cuda(self.opt.device)

        self.image = data.image
        self.target: MultiOutput = data.target

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        if self.image is None:
            raise ValueError(f"Expected model.set_data(data) be called before forward(), to get data")
        task_outputs = self.net_main(self.image)
        self.output = MultiOutput(task_outputs)

    # @torchsnooper.snoop()
    def backward(self, task_index):
        """Calculate losses_without_lambda, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss_total, self.losses_without_lambda = self.loss_criterion(self.output, self.target, task_index)
        if is_distributed_avaliable(self.opt):
            with amp.scale_loss(self.loss_total, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # backward method 1
            # self.loss_total.backward()
            # backward method 2
            for loss in self.losses_without_lambda[:-1]:
                loss.backward(retain_graph=True)
            self.losses_without_lambda[-1].backward()

    def optimize_parameters(self, task_index):
        """Update network weights; it will be called in every training iteration."""

        self.optimizer.zero_grad()  # clear network's existing gradients
        self.backward(task_index)  # calculate gradients for network main
        self.optimizer.step()  # update gradients for network

    @property
    def continued_task_index(self):
        return self._continued_task_index

    @continued_task_index.setter
    def continued_task_index(self, value):
        self._continued_task_index = value
        self.loss_criterion.continued_task_index = value

    @property
    def plus_other_loss(self):
        return self._plus_other_loss

    @plus_other_loss.setter
    def plus_other_loss(self, value):
        self._plus_other_loss = value
        self.loss_criterion.plus_other_loss = value

    def _get_all_nets(self):
        nets = self.__dict__
        return [nets["net_" + net_name] for net_name in self.net_names]

    def compute_visuals(self, visualizer):
        """Calculate additional visualization"""

        pass
