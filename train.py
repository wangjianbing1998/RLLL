# encoding=utf-8
'''
@File    :   fit.py
@Contact :   jianbingxiaman@gmail.com
@License :   (C)Copyright 2020-2021, John Hopcraft Lab-CV
@Desciption : 
@Modify Time      @Author    @Version
------------      -------    --------
2021/1/24 13:48   jianbingxia     1.0    
'''

import matplotlib

matplotlib.use('Agg')

import logging
import os
from typing import List, Tuple

from VARIABLES import JOINT_TRAIN
from task_datasets import PseudoData, create_task_dataset
from utils.util import TestMatrix, MatrixItem, MultiOutput, my_sum

"""General-purpose training script for multi-task learning(or LifeLong Learning) translation.

This script works for various models (with option '--model_name': e.rg., lwf, finetune, warmtune, hottune, folwf, tblwf) and
different task_datasets (with option '--task_dataset_name': e.rg., custom).

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the model, print/save the loss plot.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a lwf model:
        python train.py --model_name lwf --task_dataset_name custom --continue_train --n_epochs 100 --n_epochs_decay 80
    Train a folwf model:
        python train.py --model_name folwf --task_dataset_name custom --continue_train --n_epochs 100 --n_epochs_decay 80
    and so on.

See options/base_options.py and options/train_options.py for more training options.

"""
import time

from models import create_model, BaseModel
from options.train_options import TrainOptions
from utils.visualizer import Visualizer


def joint_fit(opt,
			  model,
			  task_index,
			  continued_task_index,  # currently trained task index
			  train_datasets,
			  val_datasets=None,
			  visualizer=None,
			  ):
	model.continued_task_index = continued_task_index

	logging.info(f"Fitting task {task_index}")
	best_matrix_item = None
	step = 1
	for epoch in range(opt.epoch_start, opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs;
		epoch_start_time = time.time()  # timer for entire epoch

		total_loss = 0
		n_batch = 0

		val_matrix_items = []
		for index in range(continued_task_index + 1):

			train_dataset = train_datasets[index]
			model.setup(task_index=index, step=2)

			for data in train_dataset:  # inner loop within one epoch
				image, target = data
				image = image.to(opt.device, non_blocking=True)
				target = target.to(opt.device, non_blocking=True)

				multi_output = [None for _ in range(nb_tasks)]
				data: 'image,MultiOutput' = PseudoData(opt, image, target, MultiOutput(multi_output), index)  #

				model.set_data(data)  # unpack _data from dataset and apply preprocessing
				model.train(index)

				total_loss += model.loss_total
				n_batch += 1

			# Validation
			val_matrix, _ = val(val_datasets[index], model, index, visualizer)
			val_matrix_items.append(val_matrix)

		val_matrix = my_sum(val_matrix_items)
		val_matrix = val_matrix / len(val_matrix_items)

		total_loss /= n_batch
		if epoch % opt.curve_freq == 0:  # visualizing training losses and save logging information to the disk
			visualizer.add_losses({'loss_total': total_loss}, epoch)

		if (epoch + 1) % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
			logging.info('saving the model at the end of epoch %d' % (epoch))
			model.save_networks(continued_task_index, step, epoch)

		if opt.save_best and (best_matrix_item is None or val_matrix > best_matrix_item):
			logging.info(f'saving the best model at the end of epoch {epoch}')
			model.save_networks(continued_task_index, step, epoch=str(epoch))
			best_matrix_item = val_matrix

		logging.info(
			f'End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t train_loss={total_loss.detach()},val:{val_matrix}, Time Taken: {time.time() - epoch_start_time} sec')
		model.update_learning_rate()  # update learning rates at the end of every epoch.


# @torchsnooper.snoop()
def fit(opt,
		model,
		task_index,
		continued_task_index,  # currently trained task index
		train_dataset,
		val_dataset=None,
		visualizer=None,
		step=1,
		):
	import torch
	model.continued_task_index = continued_task_index

	logging.info(f"Fitting task {task_index}, step {step}")
	best_matrix_item = None
	epoch_end = opt.n_epochs + opt.n_epochs_decay + 1
	epoch_end //= step
	len_batch = len(train_dataset)
	len_labeled_batch = int(len_batch * opt.labeled_ratio)
	logging.info(f'Task {task_index}:{len_labeled_batch} labeled batch in {len_batch}')

	for epoch in range(opt.epoch_start, epoch_end):  # outer loop for different epochs;
		epoch_start_time = time.time()  # timer for entire epoch

		total_losses = []

		for batch_index, data in enumerate(train_dataset):  # inner loop within one epoch
			image, target = data
			image = image.to(opt.device, non_blocking=True)
			target = target.to(opt.device, non_blocking=True)
			previous_data: 'image,SingleOutput' = PseudoData(opt, image, target=None)
			model.set_data(previous_data)

			model.test(visualizer=visualizer)  # Get model.output

			output = model.output
			if batch_index < len_labeled_batch:
				'''
				data.image=data.image
				data.target=[output,output,...,<data.target>,output,output,...]
				'''
				data: 'image,MultiOutput' = PseudoData(opt, image, target, output,
													   task_index)  # set data._target=Combination(model.output,target)

			else:
				if not opt.unlabeled:
					break

				# go to the unlabeled data
				# data: 'image,MultiOutput' = PseudoData(opt, image, None, output,
				#                                        task_index)  # set data.target=model.output
				data: 'image,MultiOutput' = PseudoData(opt, image, target, output,
													   task_index)  # set data._target=Combination(model.output,target)

			model.set_data(data)
			model.train(task_index)
			total_losses.append(model.loss_total)

		if len(total_losses):
			total_loss = sum(total_losses) / len(total_losses)
		else:
			total_loss = torch.FloatTensor([0.]).cuda()

		if epoch % opt.curve_freq == 0:  # visualizing training losses and save logging information to the disk
			visualizer.add_losses({'loss_total': total_loss}, epoch)
		# Validation
		val_matrix, val_matrix_items = val(val_dataset, model, task_index, visualizer)

		if (epoch + 1) % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
			logging.info('saving the model at the end of epoch %d' % (epoch))
			model.save_networks(continued_task_index, step, epoch)

		if opt.save_best and (best_matrix_item is None or val_matrix > best_matrix_item):
			logging.info(f'saving the best model at the task {continued_task_index}, step {step}, epoch {epoch}')
			model.save_networks(continued_task_index, step=step, epoch=str(epoch))
			best_matrix_item = val_matrix

		logging.info(
			f'Task {task_index} step {step} Epoch {epoch} / {epoch_end} \t train_loss={float(total_loss.detach())},val:{val_matrix}, Time Taken: {time.time() - epoch_start_time} sec')
		if len(total_losses):
			model.update_learning_rate()  # update learning rates at the end of every epoch.


def train(opt, model, task_index, continued_task_index, train_dataset, val_dataset=None, visualizer=None, step=1):
	fit(opt,
		model=model,
		task_index=task_index,
		continued_task_index=continued_task_index,
		train_dataset=train_dataset,
		val_dataset=val_dataset,
		visualizer=visualizer,
		step=step
		)

	if model.need_backward and task_index >= 1:
		"""backward training"""
		train(opt,
			  model=model,
			  task_index=task_index - 1,
			  continued_task_index=continued_task_index,
			  train_dataset=train_dataset,
			  val_dataset=val_dataset,
			  visualizer=visualizer,
			  )


def test(opt, test_datasets, model: BaseModel, train_index, task_index, visualizer=None):
	"""test the model on multi-task test_datasets, after training task indexed with <train_index>

	Return
	None
	the global testMatrix will be updated
	"""

	print(f'=============================After trained task {train_index - 1}=========================================')
	matrixItems = []

	for test_index, test_dataset in enumerate(test_datasets):
		matrixItem, _ = val(test_dataset, model, test_index, visualizer, )
		test_matrix[(train_index, test_index + 1)] = matrixItem

		if test_index <= task_index:
			matrixItems.append(model.get_matrix_item(task_index))

		print(f'Test in task {test_index}, matrixItem=({matrixItem})')

	if len(matrixItems):
		res = my_sum(matrixItems)
		res = res / len(matrixItems)
		print(f'Average Accuracy is ({res})')

	train_index += 1
	return train_index


def val(val_dataset: 'Single task_dataset', model: BaseModel, task_index, visualizer=None) -> Tuple[MatrixItem, List]:
	"""for validation on one task"""
	logging.info(f"Validating task {task_index}")
	start_time = time.time()  # timer for validate a task

	matrixItems = []
	for i, data in enumerate(val_dataset):  # inner loop within one epoch
		image, target = data
		# logging.debug(f'{image.shape},{target}')
		image = image.to(opt.device, non_blocking=True)
		target = target.to(opt.device, non_blocking=True)

		model.set_data(PseudoData(opt, image, target))
		model.test(visualizer)
		# Add matrixItem result
		matrixItems.append(model.get_matrix_item(task_index))

	res = my_sum(matrixItems)
	res = res / len(matrixItems)
	logging.info(f"Validation Time Taken: {time.time() - start_time} sec")
	return res, matrixItems


if __name__ == '__main__':

	train_index = 0
	opt = TrainOptions().parse()  # get training options
	print(f'Name={opt.name:*^50}')

	model = create_model(opt)  # create a model given opt.model and other options
	model.setup()  # regular setup: load and print networks; create schedulers

	visualizer = Visualizer(opt)
	visualizer.setup()  # regular setup:

	train_datasets = create_task_dataset(opt, phase="train")
	val_datasets = create_task_dataset(opt, phase="val")
	test_datasets = create_task_dataset(opt, phase="test")

	print('Datasets Builded')
	nb_tasks = train_datasets.nb_tasks

	test_matrix = TestMatrix()
	train_index = test(opt, test_datasets, model, train_index=train_index, task_index=-1, visualizer=visualizer)

	for task_index in range(nb_tasks):
		step = 1
		model.setup(task_index,
					step=step)  # regular setup: load and print networks; create schedulers before training each task

		if task_index >= opt.load_taskindex:

			if opt.model_name == JOINT_TRAIN:
				joint_fit(opt,
						  model=model,
						  task_index=task_index,
						  continued_task_index=task_index,
						  train_datasets=train_datasets,
						  val_datasets=val_datasets,
						  visualizer=visualizer)
			else:
				train_dataset = train_datasets[task_index]
				val_dataset = val_datasets[task_index]

				if step >= opt.load_step:
					train(opt,
						  model=model,
						  task_index=task_index,
						  continued_task_index=task_index,
						  train_dataset=train_dataset,
						  val_dataset=val_dataset,
						  visualizer=visualizer,
						  step=step,
						  )

				for step in range(2, model.max_step + 1):
					train_index = test(opt, test_datasets, model, train_index=train_index, task_index=task_index,
									   visualizer=visualizer)
					if step >= opt.load_step:
						logging.info(f"model {opt.model_name}, step {step}")
						model.setup(task_index, step=step)
						train(opt,
							  model=model,
							  task_index=task_index,
							  continued_task_index=task_index,
							  train_dataset=train_dataset,
							  val_dataset=train_dataset,
							  visualizer=visualizer,
							  step=step
							  )

		train_index = test(opt, test_datasets, model, train_index=train_index, task_index=task_index,
						   visualizer=visualizer)

	test_matrix.save_matrix(os.path.join(opt.result_dir, '-'.join(opt.dataset_list), f'{opt.name}.xlsx'))
