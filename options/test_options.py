import platform

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        # name specifics
        parser.add_argument('--model_name', type=str, default="lwf",
                            choices=["finetune", "warmtune", "hottune", "folwf", "lwf", "rlll", ],
                            help='model choice from finetune|warmtune|lwf|rlll', )
        parser.add_argument('--task_dataset_name', type=str, default="custom", choices=["custom"],
                            help='task dataset choice from custom', )

        # experiment specifics
        parser.add_argument('--seed', type=int, default=42)

        parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
        parser.add_argument('--num_workers', type=int, default=0 if "Windows" in platform.platform() else 8,
                            help='num workers for data reading, if set value >= 4, the pin_memory will be True, otherwise, it will be False')
        parser.add_argument('--load_epoch', type=str, default='best',
                            help='which epoch to load? set to latest to use best cached model')

        # for setting inputs
        parser.add_argument('--imsize', type=int, default=256)

        # model and optimizer
        parser.add_argument('--optimizer_type', type=str, default='adam', choices=["sgd", "adam"],
                            help='Name of the optimizer')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--gamma', type=float, default=0.1)
        parser.add_argument('--lr_factor', type=float, default=0.5)
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        parser.add_argument('--pool_size', type=int, default=50,
                            help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear',
                            help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='multiply by a gamma every lr_decay_iters iterations')

        parser.add_argument('--result_dir', type=str, default='./results/', help='saves results here.')

        parser.set_defaults(preffix='test')
        self.isTrain = False
        return parser
