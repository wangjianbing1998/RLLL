import logging
import os
from abc import ABC
from collections import OrderedDict

import torch
# torch.multiprocessing.set_start_method('spawn')
from torch.cuda.amp import autocast as autocast, GradScaler
from torch.nn import DataParallel

from networks import get_scheduler
from task_datasets import PseudoData
from utils.util import MatrixItem, is_gpu_avaliable, MultiOutput, exec_times


class ConfigModel(ABC):
    _shared_cnn_layers = False
    _shared_fc_layers = False
    _plus_other_loss = False
    _other_layers = False
    _task_layer = False

    _need_backward = False  # need to backward to optimize the previous task
    _continued_task_index = 0  # continued trained task index

    task_index = None
    loss_criterion = None
    net_main = None

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

    @property
    def plus_other_loss(self):
        return self._plus_other_loss

    @plus_other_loss.setter
    def plus_other_loss(self, value):
        self._plus_other_loss = value
        self.loss_criterion.plus_other_loss = value

    @property
    def shared_cnn_layers(self):
        return self._shared_cnn_layers

    @shared_cnn_layers.setter
    def shared_cnn_layers(self, value):
        if self._shared_cnn_layers != value:
            self._shared_cnn_layers = value
            self.set_requires_grad(self.net_main.module.shared_cnn_layers, requires_grad=value)

    @property
    def shared_fc_layers(self):
        return self._shared_fc_layers

    @shared_fc_layers.setter
    def shared_fc_layers(self, value):
        if self._shared_fc_layers != value:
            self._shared_fc_layers = value
            self.set_requires_grad(self.net_main.module.shared_fc_layers, requires_grad=value)

    @property
    def other_layers(self):
        return self._other_layers

    @other_layers.setter
    def other_layers(self, value):
        if self._other_layers != value:
            self._other_layers = value
            self.set_requires_grad(self.net_main.module.other_layers(self.task_index), requires_grad=value)

    @property
    def task_layer(self):
        return self._task_layer

    @task_layer.setter
    def task_layer(self, value):
        if self._task_layer != value:
            self._task_layer = value
            self.set_requires_grad(self.net_main.module.task_layer(self.task_index), requires_grad=value)

    @property
    def continued_task_index(self):
        return self._continued_task_index

    @continued_task_index.setter
    def continued_task_index(self, value):
        self._continued_task_index = value
        self.loss_criterion.continued_task_index = value


class BaseModel(ConfigModel):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.nb_tasks = len(opt.num_classes)
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain

        self.save_dir = opt.checkpoints_dir  # save all the checkpoints to save_dir

        self.loss_names = []
        self.net_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

        self.target, self.image = None, None
        self.output = None

        self.max_step = 0

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

    @staticmethod
    def default_value(opt):
        return opt

    @exec_times(1)
    def setup(self, task_index):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.task_index = task_index
        logging.info("BaseModel set up")
        if self.isTrain:
            self.schedulers = [get_scheduler(optimizer, self.opt) for optimizer in self.optimizers]

        if not self.isTrain or self.opt.continue_train:
            self.load_networks(self.opt.load_taskindex, self.opt.load_step, self.opt.load_epoch)

        if self.opt.amp:
            self.scaler = GradScaler()
        self.print_networks()

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
        self.optimizer.zero_grad(set_to_none=True)  # clear network's existing gradients, torch.version>=1.7 is ok
        # self.optimizer.zero_grad(set_grads_to_None=True)  # clear network's existing gradients
        # self.optimizer.zero_grad()
        if self.opt.amp:
            with autocast():
                self.forward()  # first call forward to calculate intermediate results
                self.loss_total, self.losses_without_lambda = self.loss_criterion(self.output, self.target, task_index)

        else:
            self.forward()  # 多次执行得到的self.output会不一样？？？
            self.loss_total, self.losses_without_lambda = self.loss_criterion(self.output, self.target, task_index)

        self.backward()  # calculate gradients for network main
        # self.scaler.unscale_(self.optimizer)

        self.step()

        if self.opt.amp:
            self.scaler.update()

    def step(self):
        if self.opt.amp:
            for optimizer in self.optimizers:
                self.scaler.step(optimizer)
        else:
            for optimizer in self.optimizers:
                optimizer.step()  # update gradients for network

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

    def save_networks(self, task_index, step, epoch):
        """Save all the networks to the disk.

        Parameters:
            task_index (int) -- current trained task_index
            step (int) -- current trained step
            epoch (Union[int,str]) -- current epoch; used in the file tag '%s_%s_net_%s.pth' % (task_index,epoch, tag)
        """

        task_index, step = int(task_index), int(step)

        for file in os.listdir(self.save_dir):
            if file.endswith('.pth'):
                __task_index, __step, __epoch, *others = file.split('_')
                __task_index, __step, __epoch = int(__task_index), int(__step), int(__epoch)

                if __task_index < task_index \
                        or __task_index == task_index and __step < step \
                        or __task_index == task_index and __step == step and (
                        __epoch < int(epoch) if epoch != 'best' else True):
                    path = os.path.join(self.save_dir, file)
                    logging.info(f'rmdir {path}')
                    # rm_dirs.append(path)
                    os.remove(path)

        for name in self.net_names:
            if isinstance(name, str):
                save_filename = '%s_%s_%s_net_%s.pth' % (task_index, step, epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, "net_" + name)
                if is_gpu_avaliable(self.opt):
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.to(device=self.opt.device)
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, taks_index, step, epoch):
        """Load all the networks from the disk.

        Parameters:
            task_index (int) -- current trained task_index
            epoch (int or str) -- current epoch or best epoch; used in the file tag '{taks_index}_{epoch}_net_{name}.pth' % (epoch, tag)
        """
        for name in self.net_names:
            if isinstance(name, str):
                load_filename = f'{taks_index}_{step}_{epoch}_net_{name}.pth'
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

    def print_networks(self):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
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

    def get_matrix_item(self, task_index) -> MatrixItem:
        """for current task_index, after calling set_data() and forward(), then call <get_matrix_item()> will get MatrixItem """
        if self.output is None or self.target is None:
            raise ValueError(
                f"Expected model.set_data(data) and model.forward() be called before model.get_matrix_item(), but not called {'forward()' if self.output is None else ''} {'and set_data()' if self.target is None else ''}")
        return MatrixItem(self.output[task_index], self.target, self.loss_criterion)

    def init_net_with_dataparaller(self, opt, net):
        """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
        Parameters:
            net (network)      -- the network to be initialized
            init_type (str)    -- the tag of an initialization method: normal | xavier | kaiming | orthogonal
            gain (float)       -- scaling factor for normal, xavier and orthogonal.
            gpu_ids (int list) -- which GPUs the network runs on: e.rg., 0,1,2

        Return an initialized network.
        """

        if len(opt.gpu_ids) > 0:
            assert (torch.cuda.is_available())
            # for none distributed data parallel, only DataParallel
            # net = DataParallel(net, device_ids=opt.gpu_ids)
            # net = DataParallel(net, device_ids=[0])
            net = DataParallel(net)

            # self.cuda()
            # net.to(device=self.opt.device)
            logging.info(f'Data Paraller: DataParallel')
            return net
        else:
            raise ValueError(f'Expected GPU training, but got none gpu_ids')

    def __getattr__(self, item):
        if "loss" in item and hasattr(self.loss_criterion, item):
            return getattr(self.loss_criterion, item)
        for net in self._get_all_nets():
            if "net" in item and hasattr(net, item):
                return getattr(net, item)
        raise AttributeError(f"Model {self.opt.model_name} object has no attribute '{item}'")

    def set_data(self, data: PseudoData):
        """Unpack input _data from the dataloader and perform necessary pre-processing steps.
        """
        self.image = data.image
        self.target: MultiOutput = data.target

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        if self.image is None:
            raise ValueError(f"Expected model.set_data(data) be called before forward(), to get data")

        task_outputs = self.net_main(self.image)
        self.output: MultiOutput = MultiOutput(task_outputs)

    # @torchsnooper.snoop()
    def backward(self):
        """Calculate losses_without_lambda, gradients, and update network weights; called in every training iteration"""
        if self.opt.amp:
            for loss in self.losses_without_lambda[:-1]:
                self.scaler.scale(loss).backward(retain_graph=True)
            self.scaler.scale(self.losses_without_lambda[-1]).backward()

        else:
            for loss in self.losses_without_lambda[:-1]:
                loss.backward(retain_graph=True)
            self.losses_without_lambda[-1].backward()

    def _get_all_nets(self):
        nets = self.__dict__
        return [nets["net_" + net_name] for net_name in self.net_names]

    def compute_visuals(self, visualizer):
        """Calculate additional visualization"""

        pass

    def cuda(self, device=None):
        """cuda: net"""
        if device is None:
            device = self.opt.device
        for name in self.net_names:
            if isinstance(name, str):
                net = getattr(self, "net_" + name)
                # net=net.cuda(device)
                net.to(device)
                logging.info(f'[Network {name}] move to cuda({device})')

    def init_optimizers(self, opt):

        self.cuda()

        if self.isTrain:
            if self.opt.optimizer_type == 'adam':
                self.optimizer = torch.optim.Adam(self.net_main.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optimizer_type == 'sgd':
                self.optimizer = torch.optim.SGD(self.net_main.parameters(), lr=opt.lr)
            elif opt.optimizer_type == 'adamw':
                self.optimizer = torch.optim.AdamW(self.net_main.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                raise ValueError(f"Expected opt.optimizer_type in ['adam','sgd'], but got {opt.optimizer_type}")
            self.net_main = self.init_net_with_dataparaller(opt, self.net_main)
            self.optimizers = [self.optimizer]
