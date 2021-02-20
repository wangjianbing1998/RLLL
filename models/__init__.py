"""This package contains modules related to objective functions, optimizations, and network architectures.

To add a custom model class called 'dummy', you need to add a file called 'dummy_model.py' and define a subclass DummyModel inherited from BaseModel.
You need to implement the following five functions:
    -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
    -- <set_data>:                      unpack _data from PseudoData and apply preprocessing.
    -- <forward>:                       produce intermediate results.
    -- <backward>:                      calculate the loss within the input of model.output and model.target on task indexed with task_index
    -- <optimize_parameters>:           calculate loss, gradients, and update network weights.
    -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    -- <continued_task_index.setter>:   set the <self.loss_dummy>.continued_task_index
    -- <plus_other_loss.setter>:        set the <self.loss_dummy>.plus_other_loss
    -- <_get_all_net>:                  get all nets from the model
    -- <compute_visuals>:               compute additional visuals for images and output, loss


In the function <__init__>, you need to define four lists:
    -- self.loss_names (str list):          specify the training losses that you want to plot and save.
    -- self.net_names (str list):           define networks used in our training.
    -- self.visual_names (str list):        specify the visuals that you want to display and save.
    -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them.

Now you can use the model class by specifying flag '--model_name dummy'.
See our template model class 'lwf_model.py' for more details.
"""

import importlib
import logging

from models.base_model import BaseModel

model_names = []


def find_model_using_name(model_name):
    """Import the module "models/[net_name]_model.py".

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if 'model' in name.lower():
            model_names.append(name.lower())
            if name.lower() == target_model_name.lower():
                model = cls

    if model is None:
        logging.error(
            "In %s.py, there should be a subclass of BaseModel with class tag that matches %s in lowercase." % (
                model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    """Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'fit.py'/'test.py'

    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    model = find_model_using_name(opt.model_name)
    instance = model(opt)
    logging.info("model [%s] was created" % type(instance).__name__)
    return instance
