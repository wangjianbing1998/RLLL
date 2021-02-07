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
from torchvision.models import alexnet, vgg16_bn

from networks import MultiOutputClassifier
from networks.base_net import BaseNet


class Vgg16bnNet(BaseNet):

    @staticmethod
    def modify_commandline_options(parser):
        """Add new dataset-specific options, and rewrite default values for existing options.

        num_classes is the number of classes  per task

        Parameters:
            parser          -- original option parser

        Returns:
            the modified parser.

        """
        parser.add_argument("--pretrained", type=bool, default=True, help="if pretrained the base alexnet or not")
        return parser

    def __init__(self, opt):
        BaseNet.__init__(self, opt)  # call the initialization method of BaseModel
        self.net_names = ['vgg16bn']

        base_vgg16 = vgg16_bn(pretrained=opt.pretrained)
        self.shared_cnn_layers = base_vgg16.features
        self.adap_avg_pool = base_vgg16.avgpool
        self.shared_fc_layers = base_vgg16.classifier[:6]
        self.in_features = base_vgg16.classifier[6].in_features

        self.multi_output_classifier = MultiOutputClassifier(self.in_features, opt.num_classes)
        self.multi_output_classifier.cuda()

    def forward(self, _input):
        cnn_out = self.shared_cnn_layers(_input)
        cnn_out = self.adap_avg_pool(cnn_out)
        cnn_out_flatten = cnn_out.contiguous().view(cnn_out.size()[0], -1)
        shared_fc_out = self.shared_fc_layers(cnn_out_flatten)
        multi_output_classifer = self.multi_output_classifier(shared_fc_out)
        return multi_output_classifer