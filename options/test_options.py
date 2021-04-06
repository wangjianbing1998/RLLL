import platform

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        # name specifics
        parser.add_argument('--task_dataset_name', type=str, default="custom", choices=["custom"],
                            help='task dataset choice from custom', )

        # experiment specifics
        parser.add_argument('--seed', type=int, default=42)

        parser.add_argument('--log_filename', type=str, default="output/test_{}.txt", help='logging filename')
        # parser.add_argument('--log_filename', type=str, default=None, help='logging filename')

        parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
        parser.add_argument('--num_workers', type=int, default=0 if "Windows" in platform.platform() else 8,
                            help='num workers for data reading, if set value >= 4, the pin_memory will be True, otherwise, it will be False')
        parser.add_argument('--load_epoch', type=str, default='best',
                            help='which epoch to load? set to latest to use best cached model')
        parser.add_argument('--load_taskindex', type=int, default=0,
                            help='which trained-to-task_index model to load? set to latest to use best cached model')

        # for setting inputs

        # model and optimizer

        parser.add_argument('--pool_size', type=int, default=50,
                            help='the size of image buffer that stores previously generated images')

        parser.add_argument('--result_dir', type=str, default='./test_results/', help='saves results here.')

        parser.set_defaults(preffix='test')
        self.isTrain = False
        return parser
