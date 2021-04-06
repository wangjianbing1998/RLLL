# encoding=utf-8
'''
@File    :   jointtrain_model.py    
@Contact :   jianbingxiaman@gmail.com
@License :   (C)Copyright 2020-2021, John Hopcraft Lab-CV
@Desciption : 
@Modify Time      @Author    @Version
------------      -------    --------
2021/3/25 16:50   jianbingxia     1.0    
'''
import torch

from losses import create_loss
from models.base_model import BaseModel
from networks import create_net


class JointTrainModel(BaseModel):

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

        self.plus_other_loss = True
        self.need_backward = False

    def setup(self, task_index=0):
        BaseModel.setup(self, task_index)  # call the initialization method of BaseModel
        if task_index > 0:
            # self.set_requires_grad(self.net_main.module.shared_cnn_layers, requires_grad=False)
            # self.set_requires_grad(self.net_main.module.shared_fc_layers, requires_grad=False)
            pass
        self.set_requires_grad(self.net_main.module.other_layers(task_index), requires_grad=True)
        self.set_requires_grad(self.net_main.module.task_layer(task_index), requires_grad=True)

    def joint_train(self, task_index, train_datasets, ):
        for train_dataset in train_datasets:
            pass
            for data in train_dataset:  # inner loop within one epoch
                previous_data: 'image,SingleOutput' = PseudoData(opt, Bunch(**data["data"]))
                model.set_data(previous_data)
                model.test(visualizer=visualizer)  # Get model.output
                '''
                data.image=data.image
                data.target=[output,output,...,<data.target>,output,output,...]
                '''
                data: 'image,MultiOutput' = PseudoData(opt, previous_data, model.output, task_index)  #

                # logging.debug("after:"+str(data))
                # assert all(data.target[task_index] == previous_data.target)
                # set data and fit
                model.set_data(data)  # unpack _data from dataset and apply preprocessing
                model.train(task_index)

                losses = model.get_current_losses()
                total_loss += losses['loss_total']
                n_batch += 1
            total_loss /= n_batch
            if epoch % opt.curve_freq == 0:  # visualizing training losses and save logging information to the disk
                visualizer.add_losses({'loss_total':total_loss}, epoch)
            # Validation
            val_matrix, val_matrix_items = val(val_dataset, model, task_index, visualizer)

            if (epoch + 1) % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
                logging.info('saving the model at the end of epoch %d' % (epoch))
                model.save_networks(continued_task_index, epoch)

            if opt.save_best and (best_matrix_item is None or val_matrix > best_matrix_item):
                logging.info(f'saving the best model at the end of epoch {epoch}')
                model.save_networks(continued_task_index, epoch="best")

            logging.info(
                f'End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t train_loss={total_loss.item()},val:{val_matrix}, Time Taken: {time.time() - epoch_start_time} sec')
            model.update_learning_rate()  # update learning rates at the end of every epoch.
