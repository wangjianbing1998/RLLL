# encoding=utf-8
'''
@File    :   VARIABLES.py    
@Contact :   jianbingxiaman@gmail.com
@License :   (C)Copyright 2020-2021, John Hopcraft Lab-CV
@Desciption : 
@Modify Time      @Author    @Version
------------      -------    --------
2021/1/24 10:51   jianbingxia     1.0    
'''

# DATA NAME
MNIST = "mnist"
CIFAR10 = "cifar10"
CIFAR100 = "cifar100"
IMAGENET = "imagenet"
MINIIMAGENET = "miniimagenet"
CUB = "cub"

NB_MNIST = 10
NB_CIFAR10 = 10
NB_CIFAR100 = 100
NB_IMAGENET = 1000
NB_MINIIMAGENET = 100
NB_CUB = 200

nc_datas = {
	'mnist': NB_MNIST,
	'cifar10': NB_CIFAR10,
	'cifar100': NB_CIFAR100,
	'imagenet': NB_IMAGENET,
	'miniimagenet': NB_MINIIMAGENET,
	'cub': NB_CUB,
}

# MODEL NAME
LWF = "lwf"
U_LWF = "ulwf"
TBLWF = "tblwf"
U_TBLWF = "utblwf"
FOLWF = "folwf"
FFLWF = "fflwf"
FCLWF = "fclwf"
FALWF = "falwf"
WALWF = "walwf"
HALWF = "halwf"
U_FALWF = "ufalwf"
U_WALWF = "uwalwf"
U_HALWF = "uhalwf"
FINE_TUNE = "finetune"
WARM_TUNE = "warmtune"
HOT_TUNE = "hottune"
JOINT_TRAIN = "jointtrain"
E_TUNE = "etune"
U_E_TUNE = "uetune"
