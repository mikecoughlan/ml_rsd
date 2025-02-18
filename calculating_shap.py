# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import datetime as dt
import gc
import json
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib import colors
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from spacepy import pycdf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import twins_model_v_autoencoder as twins_autoencoder 
import twins_model_v_maxpooling as twins_maxpooling
import swmag_model as swmag_modeling
from data_prep import PreparingData

import utils

pd.options.mode.chained_assignment = None

os.environ["CDF_LIB"] = "~/CDF/lib"

data_directory = '../../../../data/'
supermag_dir = '../data/supermag/feather_files/'
regions_dict = 'mike_working_dir/identifying_regions_data/identifying_regions_data/twins_era_identified_regions_min_2.pkl'
regions_stat_dict = 'mike_working_dir/identifying_regions_data/identifying_regions_data/twins_era_stats_dict_radius_regions_min_2.pkl'
modeling_dir = data_directory+'mike_working_dir/including_ion_temp_maps/'


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')


CONFIG = {'time_history':60,
			'random_seed':42,
			'filters':128,
			'learning_rate':1e-7,
			'epochs':500,
			'loss':'mse',
			'early_stop_patience':25,
			'batch_size':128}


def loading_model(auto_or_max='auto'):

	if MODEL_TYPE == 'twins':
		if auto_or_max == 'auto':
			print('Loading the twins autoencoder model....')
			autoencoder = twins_autoencoder.Autoencoder()
			saved_model = torch.load(f'models/autoencoder_pytorch_perceptual_v1-42.pt')
			autoencoder.load_state_dict(saved_model['model'])

			# getting jsut the encoder part of the model
			encoder = autoencoder.encoder

			encoder.to(DEVICE)

			model = twins_autoencoder.TWINSModel(encoder=encoder)

		elif auto_or_max == 'max':
			print('Loading the twins maxpooling model....')
			model = twins_maxpooling.TWINSModel()
	elif MODEL_TYPE == 'swmag':
		print('Loading the swmag model....')
		model = swmag_modeling.SWMAG()
	
	else:
		raise ValueError('The model type must be either twins or swmag.')
	checkpoint = torch.load(f'{modeling_dir}models/{TARGET}/region_{REGION}_{VERSION}.pt')
	# new_keys = ['conv_block.0.weight', 'conv_block.0.bias', 'conv_block.3.weight', 'conv_block.3.bias',
	# 					'linear_block.0.weight', 'linear_block.0.bias', 'linear_block.3.weight', 'linear_block.3.bias', 'linear_block.6.weight', 'linear_block.6.bias']
	# checkpoint['model'] = {new_key:value for new_key, value in zip(new_keys, checkpoint['model'].values())}
	model.load_state_dict(checkpoint['model'])
	print(model)
	# Print the shape of each weight matrix
	model.to(DEVICE)
	model.eval()



	return model



def get_shap_values(model, model_name, training_data, testing_data, background_examples=1000):
	'''
	Function that calculates the shap values for the given model and evaluation data. First checks for previously calculated shap
	values and loads them if they exist. If not, it calculates them and saves them to a pickle file.

	Args:
		model (keras object): trainined neural network model to calculate shap values for.
		background_examples (int, optional): number of background samples to use in calculating shap values. Defaults to 1000.

	Returns:
		np.array or list of np.arrays: shap values for each input feature. Will return a list of arrays if the model has multiple
										inputs. Shape will be the same as the shape of the evaluation data with an additional dimension
										for each of the model outputs.
	'''


	# checking to see if the xtrain is a list of multiple inputs. Creates background for each using same random sampling
	background = []
	if isinstance(training_data, list):
		print('Training data is a list....')
		random_indicies = np.random.choice(training_data[0].shape[0], background_examples, replace=False)
		for data in training_data:
			background.append(data[random_indicies].to(DEVICE, dtype=torch.float))

		explainer = shap.DeepExplainer(model, background)

		print('Calculating shap values....')
		delimiter = 10
		shap_values = []
		for batch in tqdm(range(0,testing_data[0].shape[0],delimiter)):
			try:
				shap_values.append(explainer.shap_values([testing_data[0][batch:(batch+delimiter)].to(DEVICE, dtype=torch.float),
														testing_data[1][batch:(batch+delimiter)].to(DEVICE, dtype=torch.float)],
														check_additivity=False))
			except IndexError:
				shap_values.append(explainer.shap_values([testing_data[0][batch:(testing_data.shape[0]-1)].to(DEVICE, dtype=torch.float),
															testing_data[1][batch:(testing_data.shape[0]-1)].to(DEVICE, dtype=torch.float)],
															check_additivity=False))

	elif isinstance(training_data, np.ndarray) or isinstance(training_data, torch.Tensor):
		print('Training data is a numpy array....')
		random_indicies = np.random.choice(training_data.shape[0], background_examples, replace=False)
		background = training_data[random_indicies].to(DEVICE, dtype=torch.float)

		explainer = shap.DeepExplainer(model, background)

		print('Calculating shap values....')
		delimiter = 10
		shap_values = []
		for batch in tqdm(range(0,testing_data.shape[0],delimiter)):
			try:
				shap_values.append(explainer.shap_values(testing_data[batch:(batch+delimiter)].to(DEVICE, dtype=torch.float), check_additivity=False))
			except IndexError:
				shap_values.append(explainer.shap_values(testing_data[batch:(testing_data.shape[0]-1)].to(DEVICE, dtype=torch.float), check_additivity=False))

	else:
		raise ValueError('The training data must be a numpy array or a list of arrays.')


	return shap_values, explainer.expected_value


def converting_shap_to_percentages(shap_values, features):

	if len(shap_values) > 1:
		all_shap_values = []
		for shap in shap_values:
			summed_shap_values = np.sum(shap, axis=1)
			summed_shap_values = summed_shap_values.reshape(summed_shap_values.shape[0], summed_shap_values.shape[1])
			shap_df = pd.DataFrame(summed_shap_values, columns=features)
			perc_df = (shap_df.div(shap_df.abs().sum(axis=1), axis=0))*100
			all_shap_values.append(perc_df)

	else:
		summed_shap_values = np.sum(shap_values, axis=1)
		summed_shap_values = summed_shap_values.reshape(summed_shap_values.shape[0], summed_shap_values.shape[1])
		shap_df = pd.DataFrame(summed_shap_values, columns=features)
		perc_df = (shap_df.div(shap_df.abs().sum(axis=1), axis=0))*100
		all_shap_values = perc_df

	return all_shap_values


def preparing_shap_values_for_plotting(df, dates):

	df = handling_gaps(df, 15, dates)

	df = df['2012-03-09 00:00:00':'2012-03-10 00:00:00']

	# Seperating the positive contributions from the negative for plotting
	pos_df = df.mask(df < 0, other=0)
	neg_df = df.mask(df > 0, other=0)

	pos_dict, neg_dict = {}, {}

	# Creating numpy arrays for each parameter
	for pos, neg in zip(pos_df, neg_df):
		pos_dict[pos] = pos_df[pos].to_numpy()
		neg_dict[neg] = neg_df[neg].to_numpy()

	return pos_dict, neg_dict, df.index


def handling_gaps(df, threshold, dates):
	'''
	Function for keeping blocks of nans in the data if there is a maximum number of data points between sucessive valid data.
	If the number of nans is too large between sucessive data points it will drop those nans.

	Args:
		df (pd.DataFrame): data to be processed

	Returns:
		pd.DataFrame: processed data
	'''
	df['Date_UTC'] = dates
	df.set_index('Date_UTC', inplace=True)
	df.index = pd.to_datetime(df.index)

	start_time = pd.to_datetime('2009-07-19')
	end_time = pd.to_datetime('2018-12-31')
	date_range = pd.date_range(start_time, end_time, freq='min')

	full_time_df = pd.DataFrame(index=date_range)

	df = full_time_df.join(df, how='left')

	# creting a column in the data frame that labels the size of the gaps
	df['gap_size'] = df[df.columns[1]].isna().groupby(df[df.columns[1]].notna().cumsum()).transform('sum')

	# setting teh gap size column to nan if the value is above the threshold, setting it to 0 otherwise
	df['gap_size'] = np.where(df['gap_size'] > threshold, np.nan, 0)

	# dropping nans from the subset of the gap size column
	df.dropna(inplace=True, subset=['gap_size'])

	# dropping the gap size column
	df.drop(columns=['gap_size'], inplace=True)

	return df


def plotting_shap_values(evaluation_dict, features, region):

	for key in evaluation_dict.keys():

		shap_percentages = converting_shap_to_percentages(evaluation_dict[key]['shap_values'], features)
		mean_pos_dict, mean_neg_dict, mean_dates = preparing_shap_values_for_plotting(shap_percentages[0], evaluation_dict[key]['Date_UTC'])
		std_pos_dict, std_neg_dict, std_dates = preparing_shap_values_for_plotting(shap_percentages[1], evaluation_dict[key]['Date_UTC'])

		colors = sns.color_palette('tab20', len(mean_pos_dict.keys()))

		# Creating the x-axis for the plot
		x = evaluation_dict[key]['Date_UTC'].values

		# Plotting
		fig = plt.figure(figsize=(20,17))

		ax1 = plt.subplot(111)
		ax1.set_title('SHAP Values for Mean Predictions')
		pos_values = [val for val in mean_pos_dict.values()]
		neg_values = [val for val in mean_neg_dict.values()]

		# Stacking the positive and negative percent contributions
		plt.stackplot(mean_dates, pos_values, labels=features, colors=colors, alpha=1)
		plt.stackplot(mean_dates, neg_values, colors=colors, alpha=1)
		ax1.margins(x=0, y=0)				# Tightning the plot margins
		plt.ylabel('Percent Contribution')

		# Placing the legend outside of the plot
		plt.legend(bbox_to_anchor=(1,1), loc='upper left')
		plt.axhline(0, color='black')

		plt.savefig(f'plots/shap/{TARGET}/{key}_twins_mean_region_{region}.png')


		# Plotting
		fig = plt.figure(figsize=(20,17))

		ax1 = plt.subplot(111)
		ax1.set_title('SHAP Values for Std Predictions')

		pos_values = [val for val in std_pos_dict.values()]
		neg_values = [val for val in std_neg_dict.values()]

		# Stacking the positive and negative percent contributions
		plt.stackplot(std_dates, pos_values, labels=features, colors=colors, alpha=1)
		plt.stackplot(std_dates, neg_values, colors=colors, alpha=1)
		ax1.margins(x=0, y=0)				# Tightning the plot margins
		plt.ylabel('Percent Contribution')

		# Placing the legend outside of the plot
		plt.legend(bbox_to_anchor=(1,1), loc='upper left')
		plt.axhline(0, color='black')

		plt.savefig(f'plots/shap/{TARGET}/{key}_twins_std_region_{region}.png')


def getting_feature_importance(evaluation_dict, features):

	feature_importances = []

	for key in evaluation_dict.keys():
		shap_percentages = converting_shap_to_percentages(evaluation_dict[key]['shap_values'], features)

		# seperating mean and std vlaues
		mean_shap_values = shap_percentages[0]
		std_shap_values = shap_percentages[1]

		# getting the mean and std of the mean and std shap values
		mean_mean_shap = mean_shap_values.abs().mean(axis=0)
		mean_std_shap = std_shap_values.abs().mean(axis=0)

		std_mean_shap = mean_shap_values.abs().std(axis=0)
		std_std_shap = std_shap_values.abs().std(axis=0)

		feature_importance_df = pd.DataFrame({'mean_mean_shap':mean_mean_shap, 'mean_std_shap':mean_std_shap,
											'std_mean_shap':std_mean_shap, 'std_std_shap':std_std_shap}, index=features)

		feature_importances.append(feature_importance_df)

	return feature_importance_df


def main():

	feature_importance_dict = {}

	# if os.path.exists(f'outputs/shap_values/{MODEL_TYPE}_region_{REGION}_{VERSION}.pkl'):
	# 	raise ValueError(f'Shap values for region {REGION} already exist. Skipping....')


	# print(f'Preparing data....')
	# # preparing the data for the model
	# train_swmag, train_twins, ytrain, val_swmag, val_twins, yval, test_swmag, test_twins, ytest, \
	# 	dates_dict, features = utils.getting_prepared_data(target_var=TARGET, cluster=CLUSTER, region=REGION, version=VERSION, config=CONFIG, oversampling=False, get_features=True)

	if not os.path.exists(f'outputs/{TARGET}'):
		os.makedirs(f'outputs/{TARGET}')
	if not os.path.exists(f'models/{TARGET}'):
		os.makedirs(f'models/{TARGET}')
	print(VERSION)

	# loading all data and indicies
	print('Loading data...')
	PD = PreparingData(target_param=TARGET, region=REGION, cluster=CLUSTER, oversampling=False, 
						omni=False, config=CONFIG, features=['dbht', 'MAGNITUDE', 'theta', 'N', 'E', 'sin_theta', 'cos_theta'], 
						mean=True, std=True, maximum=True, median=True, window=60, forecast=30, classification=True, version=VERSION)

	train_dict, val_dict, test_dict= PD()
	features = PD.get_features()

	# data_prep = PreparingData(target_param=TARGET, region=REGION, cluster=CLUSTER, oversampling=False, 
	# 					omni=False, config=CONFIG, features=['dbht', 'MAGNITUDE', 'theta', 'N', 'E', 'sin_theta', 'cos_theta'], 
	# 					mean=True, std=True, maximum=True, median=True, window=60, forecast=30, classification=True, version=VERSION,
	# 					start_time='2018-01-01', end_time='2024-12-31 23:59:00', ml_challenge=True)

	# test_dict = data_prep.preping_specific_test_storms(storm_list=['2023-01-04 09:04:00', '2023-05-06 05:11:00', '2024-05-11 02:14:00'], solar_wind_data='dscovr')


	if MODEL_TYPE == 'twins':		
		raise ValueError('The twins model is not currently supported for the shap values.')
		# training_data, testing_data = [train_swmag, train_twins], [test_swmag, test_twins]
	elif MODEL_TYPE == 'swmag':
		train_twins, val_twins, test_twins = None, None, None
		training_data, testing_data = torch.tensor(train_dict['storms']).unsqueeze(1), torch.tensor(test_dict['storms']).unsqueeze(1)
	else:
		raise ValueError('The model type must be either twins or swmag.')
		

	# if os.path.exists(f'outputs/shap_values/{MODEL_TYPE}_region_{REGION}_{VERSION}.pkl'):
		# raise ValueError(f'Shap values for region {REGION} already exist. Skipping....')
	train_size = list(training_data.shape)
	print('Loading model....')
	MODEL = loading_model(auto_or_max='max')

	print('Model summary....')
	print(summary(MODEL, (1, train_size[2], train_size[3])))

	print('Getting shap values....')
	shap_values, expected_values = get_shap_values(model=MODEL, model_name=f'{REGION}_{VERSION}', 
													training_data=training_data, 
													testing_data=testing_data, 
													background_examples=500)
	
	if MODEL_TYPE == 'swmag':
		twins_test=None

	evaluation_dict = {'shap_values':shap_values, 
						'testing_data':testing_data,
						'ytest':test_dict['targets'],
						'Date_UTC':test_dict['dates'],
						'features':features,
						'expected_values': expected_values}

	print('Dates in evaluation dict: '+str(evaluation_dict['Date_UTC'].shape))

	with open(f'{modeling_dir}outputs/shap_values/new_{MODEL_TYPE}_region_{REGION}_{VERSION}.pkl', 'wb') as f:
		pickle.dump(evaluation_dict, f)

	print('Plotting shap values....')
	# plotting_shap_values(evaluation_dict, features, region)

	print('Getting feature importance....')
	# feature_importance_dict[region]['feature_importance'] = getting_feature_importance(evaluation_dict, features)

	gc.collect()

	# with open(f'outputs/shap_values/twins_feature_importance_dict.pkl', 'wb') as f:
	# 	pickle.dump(feature_importance_dict, f)

	# keys = [key for key in evaluation_dict.keys()]
	# # plotting feature importance for each feature as a function of mean latitude
	# for feature in features:
	# 	mean_mean_0, mean_std_0, std_mean_0, std_std_0, lat = [], [], [], [], []
	# 	mean_mean_1, mean_std_1, std_mean_1, std_std_1 = [], [], [], []
	# 	for region in REGIONS:
	# 		lat.append(feature_importance_dict[region]['mean_lat'])
	# 		mean_mean_0.append(feature_importance_dict[region]['feature_importance'][0]['mean_mean_shap'][feature])
	# 		mean_std_0.append(feature_importance_dict[region]['feature_importance'][0]['mean_std_shap'][feature])
	# 		std_mean_0.append(feature_importance_dict[region]['feature_importance'][0]['std_mean_shap'][feature])
	# 		std_std_0.append(feature_importance_dict[region]['feature_importance'][0]['std_std_shap'][feature])
	# 		mean_mean_1.append(feature_importance_dict[region]['feature_importance'][1]['mean_mean_shap'][feature])
	# 		mean_std_1.append(feature_importance_dict[region]['feature_importance'][1]['mean_std_shap'][feature])
	# 		std_mean_1.append(feature_importance_dict[region]['feature_importance'][1]['std_mean_shap'][feature])
	# 		std_std_1.append(feature_importance_dict[region]['feature_importance'][1]['std_std_shap'][feature])

	# 	# defining two colors close to each other for each of the storms
	# 	colors = ['#ff0000', '#ff4d4d', '#ff8080', '#ffcccc', '#0000ff', '#4d4dff', '#8080ff', '#ccccff']

	# 	fig = plt.figure(figsize=(20,17))
	# 	ax1 = plt.subplot(211)
	# 	ax1.set_title(f'Mean SHAP Percentage Importance for {feature}')
	# 	plt.scatter(lat, mean_mean_0, label=f'$\mu$ {keys[0]}', color=colors[0])
	# 	plt.scatter(lat, mean_std_0, label=f'$\sigma$ {keys[0]}', color=colors[1])
	# 	plt.scatter(lat, mean_mean_1, label=f'$\mu$ {keys[1]}', color=colors[4])
	# 	plt.scatter(lat, mean_std_1, label=f'$\sigma$ {keys[1]}', color=colors[5])
	# 	plt.ylabel('Mean SHAP Percentage Importance')
	# 	plt.xlabel('Region Latitude')
	# 	plt.legend()

	# 	ax2 = plt.subplot(212)
	# 	ax2.set_title(f'Std SHAP Percentage Importance for {feature}')
	# 	plt.scatter(lat, std_mean_0, label=f'$\mu$ {keys[0]}', color=colors[0])
	# 	plt.scatter(lat, std_std_0, label=f'$\sigma$ {keys[0]}', color=colors[1])
	# 	plt.scatter(lat, std_mean_1, label=f'$\mu$ {keys[1]}', color=colors[4])
	# 	plt.scatter(lat, std_std_1, label=f'$\sigma$ {keys[1]}', color=colors[5])
	# 	plt.ylabel('Std SHAP Percentage Importance')
	# 	plt.xlabel('Region Latitude')
	# 	plt.legend()

	# 	plt.savefig(f'plots/shap/{TARGET}/twins_feature_importance_{feature}.png')



if __name__ == '__main__':

	args = argparse.ArgumentParser(description='Getting the SHAP values')
	args.add_argument('--target', type=str, help='The target variable to be modeled')
	args.add_argument('--region', type=str, help='The region to be modeled')
	args.add_argument('--cluster', type=str, help='The cluster containing the region to be modeled')
	args.add_argument('--version', type=str, help='The version of the model to be used for the SHAP values.')
	args.add_argument('--model_type', type=str, help='The type of model to be used for the SHAP values.', choices=['twins', 'swmag'])

	args = args.parse_args()

	# global TARGET
	# global REGION
	# global CLUSTER

	TARGET = args.target 
	REGION = args.region
	CLUSTER = args.cluster
	VERSION = args.version
	MODEL_TYPE = args.model_type
	

	main()

	print('It ran. God job!')



'''To do for this code:
	- add the preparing data function
	- remove all instances of keras/tensorflow and replace them with pytorch
	-  modify the loading model function for pytorch
	- load the models and the CRPS from the modeling code
	- put the calculations of the shap values into a function
	- write in the ability to do twins or swmag models
	- expand the to-do list when I get through these and inevitably find more things to do'''






