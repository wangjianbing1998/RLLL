# encoding=utf-8
'''
@File    :   alexnet_net.py    
@Contact :   jianbingxiaman@gmail.com
@License :   (C)Copyright 2020-2021, John Hopcraft Lab-CV
@Desciption : 
@Modify Time      @Author    @Version
------------      -------    --------
2021/1/23 10:48   jianbingxia     1.0    
'''

from torch import nn
from torchvision.models import alexnet

from networks.base_net import BaseNet


class AlexnetNet(BaseNet):

    @staticmethod
    def modify_commandline_options(parser):
        """Add new dataset-specific options, and rewrite default values for existing options.

        num_classes is the number of classes  per task
        Parameters:
            parser          -- original option parser

        Returns:
            the modified parser.

        """
        parser.add_argument("--pretrained", type=bool, default=False, help="if pretrained the base alexnet or not")
        return parser

    def __init__(self, opt):
        BaseNet.__init__(self, opt)  # call the initialization method of BaseModel
        self.net_names = ['alexnet']

        base_alexnet = alexnet(pretrained=opt.pretrained)
        self.shared_cnn_layers = base_alexnet.features
        self.adap_avg_pool = base_alexnet.avgpool
        self.shared_fc_layers = base_alexnet.classifier[:6]
        self.in_features = base_alexnet.classifier[6].in_features

        self.target_outputs = nn.ModuleList([nn.Linear(self.in_features, num_class) for num_class in opt.num_classes])

    def forward(self, _input):
        cnn_out = self.shared_cnn_layers(_input)
        cnn_out = self.adap_avg_pool(cnn_out)
        cnn_out_flatten = cnn_out.contiguous().view(cnn_out.size()[0], -1)
        shared_fc_out = self.shared_fc_layers(cnn_out_flatten)
        return [target_output(shared_fc_out) for target_output in self.target_outputs]

    def other_layers(self, task_index):
        return [classifier for index, classifier in enumerate(self.target_outputs) if index != task_index]

    def task_layer(self, task_index):
        for index, classifier in enumerate(self.target_outputs):
            if index == task_index:
                return classifier
