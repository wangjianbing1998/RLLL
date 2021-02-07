import torch

from losses import create_loss
from models.base_model import BaseModel
from networks import create_net
from task_datasets import PseudoData


class FinetuneModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser):
        """Add new dataset-specific options, and rewrite default values for existing options.

        num_classes is the number of classes  per task
        for example, num_classes = [10,10,10], means the number of classes on taks1 is 10, and then so on.

        Parameters:
            parser          -- original option parser


        Returns:
            the modified parser.

        """
        parser.add_argument('--net_name', type=str, default="alexnet", choices=["alexnet", "imagenet"],
                            help='network select from alexnet|imagenet', )
        parser.add_argument('--loss_name', type=str, default="total", choices=["total"],
                            help='loss select from total', )
        parser.add_argument('--taskdataset_name', type=str, default="total", choices=["total"],
                            help='loss from total', )

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel

        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = []

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        self.net_names = ["main"]
        self.net_main = create_net(opt)

        self.loss_criterion = create_loss(opt)
        if self.opt.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.net_main.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        elif opt.optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(self.net_main.parameters(), lr=opt.lr)
        else:
            raise ValueError(f"Expected opt.optimizer_type in ['adam','sgd'], but got {opt.optimizer_type}")
        self.optimizers = [self.optimizer]
        self.loss_names = getattr(self.loss_criterion, "loss_names")

        """
        unfreeze shared_cnn_layers, shared_fc_layers and other_layers,
            calculate other loss
            not backward
        """

        self.set_requires_grad(self.net_main.module.shared_cnn_layers, requires_grad=False)
        self.set_requires_grad(self.net_main.module.shared_fc_layers, requires_grad=False)

        self.plus_other_loss = False
        self.need_backward = False

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
        self.image = data.image
        self.target = data.target
        if not self.image.is_cuda:
            self.image = self.image.cuda()
        if not self.target.is_cuda:
            self.target = self.target.cuda()

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        if self.image is None:
            raise ValueError(f"Expected model.set_data(data) be called before forward(), to get data")
        self.output = self.net_main(self.image)

    # @torchsnooper.snoop()
    def backward(self, task_index):
        """Calculate losses_without_lambda, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss_total = self.loss_criterion(self.output, self.target, task_index)
        self.loss_total.backward()  # calculate gradients of network G w.r.t. loss_total

    def optimize_parameters(self, task_index):
        """Update network weights; it will be called in every training iteration."""
        self.optimizer.zero_grad()  # clear network G's existing gradients
        self.backward(task_index)  # calculate gradients for network G
        self.optimizer.step()  # update gradients for network G

    def train(self, task_index):
        self.set_requires_grad(self.net_main.module.multi_output_classifier.other_layers(task_index),requires_grad=False)
        self.set_requires_grad(self.net_main.module.multi_output_classifier.task_layer(task_index), requires_grad=True)
        BaseModel.train(self, task_index)  # call the initialization method of BaseModel

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
