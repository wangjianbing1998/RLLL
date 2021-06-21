import platform

from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # name specifics

        parser.add_argument('--task_dataset_name', type=str, default="custom", choices=["custom"],
                            help='task dataset choice from custom', )
        parser.add_argument('--continue_train', action="store_true", help="if continuely train the model or not")

        parser.add_argument('--log_filename', type=str, default="output/{}_train.txt", help='logging filename')
        # parser.add_argument('--log_filename', type=str, default=None, help='logging filename')

        # experiment specifics
        parser.add_argument('--seed', type=int, default=42)

        # training labeled dataset ratio
        parser.add_argument('--labeled_ratio', type=float, default=1.,
                            help='the labeled ratio of per dataset, 1-labeled_ratio is the unlabeled ratio')
        parser.add_argument('--unlabeled', action='store_true', help='use unlabeled model or not')

        # training specifics

        parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
        parser.add_argument('--num_workers', type=int, default=0 if "Windows" in platform.platform() else 8,
                            help='num workers for data reading, if set value >= 4, the pin_memory will be True, otherwise, it will be False')

        parser.add_argument('--save_best', type=bool, default=True, help='if save the best model or not')
        parser.add_argument('--load_taskindex', type=int, default=0,
                            help='which trained-to-task_index model to load? set to latest to use best cached model')
        parser.add_argument('--load_step', type=int, default=1,
                            help='which trained-to-step model to load? set to latest step model')

        parser.add_argument('--load_epoch', type=str, default='best',
                            help='which epoch to load? set to latest to use best cached model')
        # training before
        parser.add_argument('--dels', type=str, default="",
                            help='which need to be clear delete, strings contains ckpt, log or output ')

        parser.add_argument('--curve_freq', type=int, default=1,
                            help='frequence of plotting the loss or accuracy curve')
        parser.add_argument('--epoch_start', type=int, default=0,
                            help='the starting epoch count, we save the model by <epoch_start>, <epoch_start>+<save_latest_freq>, ...')
        parser.add_argument('--n_epochs', type=int, default=100,
                            help='number of epochs with the initial learning rate, total training epochs equals to <n_epochs>+<n_epochs_decay>')
        parser.add_argument('--n_epochs_decay', type=int, default=100,
                            help='number of epochs to linearly decay learning rate to zero')

        # for displays
        parser.add_argument('--save_epoch_freq', type=int, default=float('inf'),
                            help='frequency of saving checkpoints at the end of epochs')

        # model and optimizer
        parser.add_argument('--optimizer_type', type=str, default='adamw', choices=["sgd", "adam", 'adamw'],
                            help='Name of the optimizer')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--gamma', type=float, default=0.1)
        parser.add_argument('--lr_factor', type=float, default=0.5)
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        parser.add_argument('--pool_size', type=int, default=50,
                            help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='cyclic',
                            help='learning rate policy. [linear | step | plateau | cosine|cyclic]')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--result_dir', type=str, default='./train_results/', help='saves results here.')
        parser.add_argument('--amp', action='store_true', help='where to use amp or not')

        self.isTrain = True
        return parser

    def default_value(self, opt):
        opt = BaseOptions.default_value(self, opt)

        opt.log_filename = opt.log_filename.strip().lower()

        return opt
