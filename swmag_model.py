####################################################################################
#
# exmining_twins_and_supermag/swmag_modeling_v0.py
#
# Performing the modeling using the Solar Wind and Ground Magnetomoeter data.
# TWINS data passes through a pre-trained autoencoder that reduces the TWINS maps
# to a reuced dimensionality. This data is then concatenated onto the model after
# both branches of the CNN hae been flattened, and before the dense layers.
# Similar model to Coughlan (2023) but with a different target variable.
#
####################################################################################


import argparse
# Importing the libraries
import datetime
import gc
import glob
import json
import math
import os
import pickle
import subprocess
import time
import wandb

import matplotlib
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import tqdm
from scipy.stats import boxcox
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from spacepy import pycdf
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchsummary import summary
from torchvision.models.feature_extraction import (create_feature_extractor,
                                                   get_graph_node_names)

from data_prep import PreparingData

import utils

pd.options.mode.chained_assignment = None

os.environ["CDF_LIB"] = "~/CDF/lib"

data_directory = '../../../../data/'
supermag_dir = '../data/supermag/feather_files/'


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')


class CRSP(nn.Module):
	'''
	Defining the CRPS loss function for model training.
	'''

	def __init__(self):
		super(CRSP, self).__init__()

	def forward(self, y_pred, y_true):

		# splitting the y_pred tensor into mean and std

		mean, std = torch.unbind(y_pred, dim=-1)
		# y_true = torch.unbind(y_true, dim=-1)

		# making the arrays the right dimensions
		mean = mean.unsqueeze(-1)
		std = std.unsqueeze(-1)
		y_true = y_true.unsqueeze(-1)

		# calculating the error
		crps = torch.mean(self.calculate_crps(self.epsilon_error(y_true, mean), std))

		return crps

	def epsilon_error(self, y, u):

		epsilon = torch.abs(y - u)

		return epsilon

	def calculate_crps(self, epsilon, sig):

		crps = torch.mul(sig, (torch.add(torch.mul(torch.div(epsilon, sig), torch.erf(torch.div(epsilon, torch.mul(np.sqrt(2), sig)))), \
								torch.sub(torch.mul(torch.sqrt(torch.div(2, np.pi)), torch.exp(torch.div(torch.mul(-1, torch.pow(epsilon, 2)), \
								(torch.mul(2, torch.pow(sig, 2)))))), torch.div(1, torch.sqrt(torch.tensor(np.pi)))))))

		# crps = sig * ((epsilon / sig) * torch.erf((epsilon / (np.sqrt(2) * sig))) + torch.sqrt(torch.tensor(2 / np.pi)) * torch.exp(-epsilon ** 2 / (2 * sig ** 2)) - 1 / torch.sqrt(torch.tensor(np.pi)))

		return crps


class SWMAG(nn.Module):
	def __init__(self):
		super(SWMAG, self).__init__()

		self.conv_block = nn.Sequential(

			nn.Conv2d(in_channels=1, out_channels=128, kernel_size=2, stride=1, padding='same'),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=1, padding='same'),
			nn.ReLU(),
		)
		self.linear_block = nn.Sequential(
			nn.Linear(256*30*7, 256),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(128, 2),
			nn.Softmax(dim=1)
			# nn.Linear(128, 1),
			# nn.Sigmoid()
		)

	def forward(self, x):

		x = self.conv_block(x)
		x = torch.reshape(x, (-1, 256*30*7))
		x = self.linear_block(x)

		# clipping to avoid values too small for backprop
		x = torch.clamp(x, min=1e-9)

		return x


class Early_Stopping():
	'''
	Class to create an early stopping condition for the model.

	'''

	def __init__(self, decreasing_loss_patience=25):
		'''
		Initializing the class.

		Args:
			decreasing_loss_patience (int): the number of epochs to wait before stopping the model if the validation loss does not decrease
			pretraining (bool): whether the model is being pre-trained. Just used for saving model names.

		'''

		# initializing the variables
		self.decreasing_loss_patience = decreasing_loss_patience
		self.loss_counter = 0
		self.training_counter = 0
		self.best_score = None
		self.early_stop = False
		self.best_epoch = None

	def __call__(self, train_loss, val_loss, model, optimizer, epoch):
		'''
		Function to call the early stopping condition.

		Args:
			train_loss (float): the training loss for the model
			val_loss (float): the validation loss for the model
			model (object): the model to be saved
			epoch (int): the current epoch

		Returns:
			bool: whether the model should stop training or not
		'''

		# using the absolute value of the loss for negatively orientied loss functions
		# val_loss = abs(val_loss)

		# initializing the best score if it is not already
		self.model = model
		self.optimizer = optimizer
		if self.best_score is None:
			self.best_train_loss = train_loss
			self.best_score = val_loss
			self.best_loss = val_loss
			self.save_checkpoint(val_loss)
			self.best_epoch = epoch

		# if the validation loss greater than the best score add one to the loss counter
		elif val_loss >= self.best_score:
			self.loss_counter += 1

			# if the loss counter is greater than the patience, stop the model training
			if self.loss_counter >= self.decreasing_loss_patience:
				gc.collect()
				print(f'Engaging Early Stopping due to lack of improvement in validation loss. Best model saved at epoch {self.best_epoch} with a training loss of {self.best_train_loss} and a validation loss of {self.best_score}')
				return True

		# if the validation loss is less than the best score, reset the loss counter and use the new validation loss as the best score
		else:
			self.best_train_loss = train_loss
			self.best_score = val_loss
			self.best_epoch = epoch

			# saving the best model as a checkpoint
			self.save_checkpoint(val_loss)
			self.loss_counter = 0
			self.training_counter = 0

			return False

	def save_checkpoint(self, val_loss):
		'''
		Function to continually save the best model.

		Args:
			val_loss (float): the validation loss for the model
		'''

		# saving the model if the validation loss is less than the best loss
		self.best_loss = val_loss
		print('Saving checkpoint!')

		torch.save({'model': self.model.state_dict(),
					'optimizer':self.optimizer.state_dict(),
					'best_epoch':self.best_epoch,
					'finished_training':False},
					f'models/{TARGET}/region_{REGION}_{VERSION}.pt')


def resume_training(model, optimizer):
	'''
	Function to resume training of a model if it was interupted without completeing.

	Args:
		model (object): the model to be trained
		optimizer (object): the optimizer to be used
		pretraining (bool): whether the model is being pre-trained

	Returns:
		object: the model to be trained
		object: the optimizer to be used
		int: the epoch to resume training from
	'''

	try:
		checkpoint = torch.load(f'models/{TARGET}/region_{REGION}_{VERSION}.pt')
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		epoch = checkpoint['best_epoch']
		finished_training = checkpoint['finished_training']
	except KeyError:
		model.load_state_dict(torch.load(f'models/{TARGET}/region_{REGION}_{VERSION}.pt'))
		optimizer = None
		epoch = 0
		finished_training = True

	return model, optimizer, epoch, finished_training


def fit_model(model, train, val, class_weights=None, val_loss_patience=25, overfit_patience=5, num_epochs=500, learning_rate=1e-3):

	'''
	_summary_: Function to train the swmag model.

	Args:
		model (object): the model to be trained
		train (torch.utils.data.DataLoader): the training data
		val (torch.utils.data.DataLoader): the validation data
		val_loss_patience (int): the number of epochs to wait before stopping the model
									if the validation loss does not decrease
		overfit_patience (int): the number of epochs to wait before stopping the model
									if the training loss is significantly lower than the
									validation loss
		num_epochs (int): the number of epochs to train the model
		pretraining (bool): whether the model is being pre-trained

	Returns:
		object: the trained model
	'''
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	# checking if the model has already been trained, loading it if it exists
	if os.path.exists(f'models/{TARGET}/region_{REGION}_{VERSION}.pt'):
		model, optimizer, current_epoch, finished_training = resume_training(model=model, optimizer=optimizer)
	else:
		finished_training = False
		current_epoch = 0

	if current_epoch is None:
		current_epoch = 0

	# checking to see if the model was already trained or was interupted during training
	if not finished_training:

		# initializing the lists to hold the training and validation loss which will be used to plot the losses as a function of epoch
		train_loss_list, val_loss_list = [], []

		# moving the model to the available device
		model.to(DEVICE)

		# defining the loss function and the optimizer
		criterion = nn.BCELoss()
		
		# criterion = nn.BCELoss()
		optimizer = optim.Adam(model.parameters(), lr=learning_rate)

		# initalizing the early stopping class
		early_stopping = Early_Stopping(decreasing_loss_patience=val_loss_patience)

		# looping through the epochs
		while current_epoch < num_epochs:

			# starting the clock for the epoch
			stime = time.time()

			# setting the model to training mode
			model.train()

			# initializing the running loss
			running_training_loss, running_val_loss = 0.0, 0.0

			# using the training set to train the model
			for X, y in train:
				# moving the data to the available device
				X = X.to(DEVICE, dtype=torch.float)
				y = y.to(DEVICE, dtype=torch.float)
		
				# forward pass
				output = model(X)

				output = output.squeeze()
				if class_weights is not None:
					if y.dim() == 1:
					# calculating the loss
						criterion.weight = y * class_weights[1] + (1-y)*class_weights[0]
					else:
						# repeating the class weights for the batch size
						ones_tensor = torch.ones_like(y)
						criterion.weight = class_weights*ones_tensor
				
				# calculating the loss
				loss = criterion(output, y)

				# backward pass
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				# emptying the cuda cache
				X = X.to('cpu')
				y = y.to('cpu')

				# adding the loss to the running training loss
				running_training_loss += loss.to('cpu').item()


			# setting the model to eval mode so the dropout layers are not used during validation and weights are not updated
			model.eval()
			# criterion = nn.BCELoss()

			# using validation set to check for overfitting
			# looping through the batches
			for X, y in val:

				# moving the data to the available device
				X = X.to(DEVICE, dtype=torch.float)
				y = y.to(DEVICE, dtype=torch.float)

				# forward pass with no gradient calculation
				with torch.no_grad():

					output = model(X)
					# output = output.view(len(output),2)
					output = output.squeeze()

					val_loss = criterion(output, y)

					# emptying the cuda cache
					X = X.to('cpu')
					y = y.to('cpu')

					# adding the loss to the running val loss
					running_val_loss += val_loss.to('cpu').item()

			# getting the average loss for the epoch
			loss = running_training_loss/len(train)
			val_loss = running_val_loss/len(val)
			# wandb.log({'train_loss':loss, 'val_loss':val_loss})
			# adding the loss to the list
			train_loss_list.append(loss)
			val_loss_list.append(val_loss)

			# checking for early stopping or the end of the training epochs
			if (early_stopping(train_loss=loss, val_loss=val_loss, model=model, optimizer=optimizer, epoch=current_epoch)) or (current_epoch == num_epochs-1):

				# saving the final model
				gc.collect()

				# clearing the cuda cache
				torch.cuda.empty_cache()
				gc.collect()

				# clearing the model so the best one can be loaded without overwhelming the gpu memory
				model = None
				model = SWMAG()

				# loading the best model version
				final = torch.load(f'models/{TARGET}/region_{REGION}_{VERSION}.pt')

				# setting the finished training flag to True
				final['finished_training'] = True

				# getting the best model state dict
				model.load_state_dict(final['model'])

				# saving the final model
				torch.save(final, f'models/{TARGET}/region_{REGION}_{VERSION}.pt')

				# breaking the loop
				break

			# getting the time for the epoch
			epoch_time = time.time() - stime

			# printing the loss for the epoch
			print(f'Epoch [{current_epoch}/{num_epochs}], Loss: {loss:.4f} Validation Loss: {val_loss:.4f}' + f' Epoch Time: {epoch_time:.2f} seconds')

			# emptying the cuda cache
			torch.cuda.empty_cache()

			# updating the epoch
			current_epoch += 1

		# transforming the lists to a dataframe to be saved
		loss_tracker = pd.DataFrame({'train_loss':train_loss_list, 'val_loss':val_loss_list})
		loss_tracker.to_feather(f'outputs/{VERSION}_loss_tracker.feather')

		gc.collect()

	else:
		# loading the model if it has already been trained.
		try:
			final = torch.load(f'models/{TARGET}/region_{REGION}_{VERSION}.pt')
			model.load_state_dict(final['model'])
		except KeyError:
			model.load_state_dict(torch.load(f'models/{TARGET}/region_{REGION}_{VERSION}.pt'))

	return model


def evaluation(model, test, test_dates, n_ensemble=100):
	'''
	Function using the trained models to make predictions with the testing data.

	Args:
		model (object): pre-trained model
		test_dict (dict): dictonary with the testing model inputs and the real data for comparison
		split (int): which split is being tested

	Returns:
		dict: test dict now containing columns in the dataframe with the model predictions for this split
	'''
	print(f'length of test dates: {len(test_dates)}')
	# creting an array to store the predictions
	
	# setting the encoder and decoder into evaluation model
	# model.eval()

	# setting the model to train mode to enable dropout layers
	model.train()

	# creating a loss value
	running_loss = 0.0

	# making sure the model is on the correct device
	model.to(DEVICE, dtype=torch.float)

	for i in tqdm.tqdm(range(n_ensemble)):
		predicted_class_1, xtest_list, ytest_list = [], [], []
		with torch.no_grad():
			for x, y in test:

				x = x.to(DEVICE, dtype=torch.float)
				y = y.to(DEVICE, dtype=torch.float)

				predicted = model(x)
				predicted = predicted.squeeze()
				
				# getting shape of tensor
				if len(predicted.shape) == 1:
					loss = F.mse_loss(predicted, y)
				else:
					loss = F.mse_loss(predicted[:,1], y[:,1])
				running_loss += loss.item()

				# making sure the predicted value is on the cpu
				if predicted.get_device() != -1:
					predicted = predicted.to('cpu')
				if x.get_device() != -1:
					x = x.to('cpu')
				if y.get_device() != -1:
					y = y.to('cpu')

				# adding the decoded result to the predicted list after removing the channel dimension
				# predicted = torch.squeeze(predicted, dim=1).numpy()

				# predicted_mean.append(predicted[:,0])
				# predicted_std.append(predicted[:,1])
				if len(predicted.shape) == 1:
					predicted_class_1.append(predicted.numpy())
					if i == 0:
						ytest_list.append(y.numpy())
				else:
					predicted_class_1.append(predicted[:,1].numpy())
					if i == 0:
						ytest_list.append(y[:,1].numpy())

				if i == 0:
					x = torch.squeeze(x, dim=1).numpy()

					xtest_list.append(x)

	

	# transforming the lists to arrays
	# predicted_mean = np.concatenate(predicted_mean, axis=0)
		predicted_class_1 = np.concatenate(predicted_class_1, axis=0)
		if i == 0:
			xtest_list = np.concatenate(xtest_list, axis=0)
			ytest_list = np.concatenate(ytest_list, axis=0)

			results_df = pd.DataFrame({'predicted_0':predicted_class_1, 'actual':ytest_list, 'dates':test_dates})
		else:
			results_df[f'predicted_{i}'] = predicted_class_1
	
	# prauc = utils.calibrating_prauc(ytest_list, predicted_class_1)
	# print(f'PRAUC: {prauc}')
	# wandb.log({'prauc':prauc})
	# # results_df = pd.DataFrame({'predicted_mean':predicted_mean, 'predicted_std':predicted_std, 'actual':ytest_list, 'dates':test_dates['Date_UTC']})
	
	# print(f'Evaluation Loss: {running_loss/len(test)}')
	# wandb.log({'test_loss':running_loss/len(test)})
	
	# print(f'results df shape: {results_df.shape}')
	# print(f'results df: {results_df.head()}')

	return results_df


def main():
	'''
	Pulls all the above functions together. Outputs a saved file with the results.

	'''
	CONFIG = {'time_history':60,
			'random_seed':42,
			'filters':128,
			'learning_rate':1e-7,
			'early_stop_patience':25,
			'batch_size':128,
			'epochs':500,
			'loss':'BCE',
			'using_weights_for_imbalance':True,
			'oversampling':OVERSAMPLING,
			'target':TARGET,
			'target_dim':2,
			'region':REGION,
			'final_activation':'softmax',
			'other_notes': 'lead: 18H+60, recovery 18H+60'
			}

	# wandb.init(project=f'extended_v0-1_{CLUSTER}', entity='mike-k-coughlan-university-of-new-hampshire', config=CONFIG, name=f'{REGION}_{TARGET}')

	if not os.path.exists(f'outputs/{TARGET}'):
		os.makedirs(f'outputs/{TARGET}')
	if not os.path.exists(f'models/{TARGET}'):
		os.makedirs(f'models/{TARGET}')
	print(VERSION)

	# loading all data and indicies
	print('Loading data...')
	PD = PreparingData(target_param=TARGET, region=REGION, cluster=CLUSTER, oversampling=OVERSAMPLING, 
						omni=False, config=CONFIG, features=['dbht', 'MAGNITUDE', 'theta', 'N', 'E', 'sin_theta', 'cos_theta'], 
						mean=True, std=True, maximum=True, median=True, window=60, forecast=30, classification=True, version=VERSION)

	train_dict, val_dict, test_dict = PD()

	print(f"Train ratio: {train_dict['targets'][:,1].sum()/len(train_dict['targets'])} - Val ratio: {val_dict['targets'][:,1].sum()/len(val_dict['targets'])} - Test ratio: {test_dict['targets'][:,1].sum()/len(test_dict['targets'])}")

	train_size = list(train_dict['storms'].shape)
	# print(train_dict['targets'].shape)
	# print(train_dict['targets'])
	if CONFIG['using_weights_for_imbalance']:
		n0, n1 = (train_size[0]-train_dict['targets'][:,1].sum()), train_dict['targets'][:,1].sum()
		print(f'n0: {n0}; n1: {n1}')
		print(f'n0: {n0}; n1: {n1}')
		print(f'train size: {train_size[0]}')
		class_weights = torch.tensor([(n1/train_size[0]), (n0/train_size[0])]).to(DEVICE)
		print(f'class weights: {class_weights}')
	else:
		class_weights = None

	# creating the dataloaders
	train = DataLoader(list(zip(torch.tensor(train_dict['storms']).unsqueeze(1), torch.tensor(train_dict['targets']))), batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
	val = DataLoader(list(zip(torch.tensor(val_dict['storms']).unsqueeze(1), torch.tensor(val_dict['targets']))), batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
	test = DataLoader(list(zip(torch.tensor(test_dict['storms']).unsqueeze(1), torch.tensor(test_dict['targets']))), batch_size=CONFIG['batch_size'], shuffle=False)

	# creating the model
	print('Creating model....')

	# setting random seed
	torch.manual_seed(CONFIG['random_seed'])
	torch.cuda.manual_seed(CONFIG['random_seed'])
	model = SWMAG()

	# printing model summary
	model.to(DEVICE)
	print(summary(model, (1, train_size[1], train_size[2])))
	# fitting the model
	print('Fitting model...')
	model = fit_model(model, train, val, class_weights=class_weights, val_loss_patience=25, num_epochs=CONFIG['epochs'], learning_rate=CONFIG['learning_rate'])

	# making predictions
	print('Making predictions...')
	results_df = evaluation(model, test, test_dict['dates'])
	print(f'results df shape: {results_df.shape}')
	print(f'results df: {results_df.tail()}')
	results_df.reset_index(drop=True, inplace=True)
	results_df.to_feather(f'outputs/{TARGET}/mc_swmag_modeling_region_{REGION}_version_{VERSION}.feather')

	# clearing the session to prevent memory leaks
	gc.collect()
	torch.cuda.empty_cache()
	# wandb.finish()


if __name__ == '__main__':

	args = argparse.ArgumentParser(description='Modeling the SWMAG data')
	args.add_argument('--target', type=str, help='The target variable to be modeled')
	args.add_argument('--region', type=str, help='The region to be modeled')
	args.add_argument('--cluster', type=str, help='The cluster containing the region to be modeled')
	args.add_argument('--version', type=str, help='The version of the model to be run')
	args.add_argument('--oversampling', type=str, help='Whether to oversample the data or not', default='False')

	args = args.parse_args()

	def str2bool(v):
		if isinstance(v, bool):
			return v
		if v.lower() in ('yes', 'true', 't', 'y', '1'):
			return True
		elif v.lower() in ('no', 'false', 'f', 'n', '0'):
			return False
		else:
			raise argparse.ArgumentTypeError('Boolean value expected.')


	# global TARGET
	# global REGION
	# global CLUSTER
	if str2bool(args.oversampling):
		VERSION = args.version+'_oversampling'
	else:
		VERSION = args.version

	TARGET = args.target
	REGION = args.region
	CLUSTER = args.cluster
	OVERSAMPLING = str2bool(args.oversampling)

	main()

	print('It ran. God job!')
