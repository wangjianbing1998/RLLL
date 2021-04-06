import torch

from losses import create_loss
from models.base_model import BaseModel
from networks import create_net


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

        if self.isTrain:
            if self.opt.optimizer_type == 'adam':
                self.optimizer = torch.optim.Adam(self.net_main.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optimizer_type == 'sgd':
                self.optimizer = torch.optim.SGD(self.net_main.parameters(), lr=opt.lr)
            else:
                raise ValueError(f"Expected opt.optimizer_type in ['adam','sgd'], but got {opt.optimizer_type}")
            self.net_main, self.optimizer, self.loss_criterion = self.init_net_optimizer_with_apex(opt, self.net_main,
                                                                                                   self.optimizer,
                                                                                                   self.loss_criterion)
            self.optimizers = [self.optimizer]
        self.loss_names = getattr(self.loss_criterion, "loss_names")

        """
        unfreeze shared_cnn_layers, shared_fc_layers and other_layers,
            calculate other loss
            not backward
        """

        self.plus_other_loss = False
        self.need_backward = False

    def setup(self, task_index=0, step=1):
        if step == 1:
            BaseModel.setup(self, task_index)  # call the initialization method of BaseModel

            self.shared_fc_layers = False
            self.shared_cnn_layers = False
            self.other_layers = False
            self.task_layer = True
        else:
            raise ValueError(f'finetune Expected 1<=step<=2, but got {step}')
