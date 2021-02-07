# encoding=utf-8
'''
@File    :   base_net.py    
@Contact :   jianbingxiaman@gmail.com
@License :   (C)Copyright 2020-2021, John Hopcraft Lab-CV
@Desciption : 
@Modify Time      @Author    @Version
------------      -------    --------
2021/1/23 10:48   jianbingxia     1.0    
'''


from abc import ABC
from abc import abstractmethod
from torch import nn



class BaseNet(nn.Module, ABC):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--init_type', type=str, default='normal',
                            help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    @abstractmethod
    def forward(self, _input):
        pass

