import logging
import os
from collections import defaultdict

from torchvision import transforms

from VARIABLES import CUB
from datasets import BaseDataset, prepare_datas_by_standard_data, FolderDataset


class CubDataset(BaseDataset):
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
	label2filepaths = None

	def __init__(self, opt, phase="train"):

		opt.load_dataset_mode = 'dir'
		super(CubDataset, self).__init__(opt, phase)

		self.choices_ps = {
			'train': .8,
			'val': .1,
			'test': .1,
		}

		self.data_dir = opt.cub_dataset_dir
		self.data_name = CUB
		if phase == 'train':

			self.x_transforms = transforms.Compose([
				# my_transforms.ToCVImage(),
				transforms.RandomResizedCrop((448, 448)),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize([0.48560741861744905, 0.49941626449353244, 0.43237713785804116],
									 [0.2321024260764962, 0.22770540015765814, 0.266510054732981])
			])
		else:
			self.x_transforms = transforms.Compose([
				# my_transforms.ToCVImage(),
				transforms.Resize((448, 448)),
				transforms.ToTensor(),
				transforms.Normalize([0.4862169586881995, 0.4998156522834164, 0.4311430419332438],
									 [0.23264268069040475, 0.22781080253662814, 0.26667253517177186])
			])
		self.y_transforms = None

		# load all dataset
		label2filepaths = CubDataset.get_dataset(self.data_name, self.data_dir)

		# load phase dataset
		dataset = self.load_dataset(label2filepaths, phase)

		self.data, self._labels, self.label2Indices, self.label2target, self.target2label = prepare_datas_by_standard_data(
			dataset)

	@classmethod
	def get_dataset(cls, data_name, data_dir):
		"""load all dataset"""
		if cls.label2filepaths is None:

			logging.debug(f"load all dataset {data_name}")
			cls.label2filepaths = defaultdict(list)
			for label in os.listdir(data_dir):
				file_dir = os.path.join(data_dir, label)

				for file in os.listdir(file_dir):
					file_path = os.path.join(file_dir, file)
					cls.label2filepaths[label].append(file_path)

		return cls.label2filepaths

	def load_dataset(self, label2filepaths, phase):
		label2filepaths_phase = self.get_label2filepaths(label2filepaths, phase)

		dataset = FolderDataset(label2filepaths_phase)
		return dataset

	def set_max_dataset_size(self):
		super().set_max_dataset_size()
		self.max_dataset_size = 5000  # number of images per class
