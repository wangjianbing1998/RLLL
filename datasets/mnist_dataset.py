import os

from torchvision import datasets
from torchvision.transforms import transforms

from VARIABLES import MNIST
from datasets import BaseDataset, prepare_datas_by_standard_data


class MnistDataset(BaseDataset):
    """Mnist Dataset

    phase: "fit", "val", "test"
    transform: Resize(), ToTensor(), Normalize()
    _data:[Bunch]
        {
            image_paths: [Path,...]
            targets: [int,...], for examples, [0,1,2,3,...], not label name, but label index
        }
    label2Indices: {str:[0,1,2,3],...}
    labels:[str]
    label2target:{str:int}, {"0":0,"1":1,...}

    """

    def __init__(self, opt, phase="train"):
        # TODO split the dataset of val and test
        if phase == "val":
            phase = "test"

        opt.load_dataset_mode = 'reader'
        super(MnistDataset, self).__init__(opt, phase)
        self.data_dir = opt.mnist_dataset_dir

        self.data_name = MNIST

        self.x_transforms = transforms.Compose(
            [
                transforms.Resize((opt.imsize, opt.imsize)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (1.0,))
            ]
        )
        self.y_transforms = None

        dataset = datasets.MNIST(root=os.path.join(self.data_dir, 'raw_data'), train=self.isTrain, download=True)
        self.data, self._labels, self.label2Indices, self.label2target, self.target2label = prepare_datas_by_standard_data(
            dataset)

    def set_max_dataset_size(self):
        super().set_max_dataset_size()
        self.max_dataset_size = 5000  # number of images per class
