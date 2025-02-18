import gc
import glob
import math
import os
import pickle
from datetime import datetime
from functools import partial
from multiprocessing import Manager, Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
# import shapely
from dateutil import parser
# from geopack import geopack, t89
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Wedge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from spacepy import pycdf
from tqdm import tqdm
import torch

pd.options.mode.chained_assignment = None

os.environ["CDF_LIB"] = "~/CDF/lib"

data_dir = '../../../../data/'
supermag_dir = data_dir+'supermag/feather_files/'
regions_dict = data_dir+'mike_working_dir/identifying_regions_data/adjusted_regions.pkl'
regions_stat_dict = data_dir+'mike_working_dir/identifying_regions_data/twins_era_stats_dict_radius_regions_min_2.pkl'
working_dir = data_dir+'mike_working_dir/twins_data_modeling/'


class PreparingData():
	
	def __init__(self, cluster=None, region=None, omni=False, version='you_forgot_to_add_the_version_number', 
					config=None, oversampling=False, vars_to_keep=None, features=None, mean=False, std=False, 
					maximum=False, median=False, **kwargs):

		self.data_dir = '../../../../data/'
		self.supermag_dir = self.data_dir+'supermag/feather_files/'
		self.regions_dict = self.data_dir+'mike_working_dir/identifying_regions_data/adjusted_regions.pkl'
		self.working_dir = self.data_dir+'mike_working_dir/twins_data_modeling/'

		if cluster is None:
			raise ValueError('Must specify a cluster to analyze.')

		if region is None:
			raise ValueError('Must specify a region to analyze.')

		self.cluster = cluster
		self.region_name = region
		self.omni = omni
		self.version = version
		self.config = config
		self.oversampling = oversampling
		self.vars_to_keep = vars_to_keep
		self.features = features
		self.mean = mean
		self.std = std
		self.maximum = maximum
		self.median = median

		self.__dict__.update(kwargs)
		self.forecast = self.__dict__.get('forecast', 15)
		self.window = self.__dict__.get('window', 15)
		self.classification = self.__dict__.get('classification', False)
		self.target_param = self.__dict__.get('target_param', 'rsd')
		self.specific_test_storms = self.__dict__.get('specific_test_storms', None)
		self.start_time = self.__dict__.get('start_time', '1995-01-01')
		self.end_time = self.__dict__.get('end_time', '2018-12-31')
		self.ml_challenge = self.__dict__.get('ml_challenge', False)

		print(f'Forecast: {self.forecast}, Window: {self.window}, Target parameter: {self.target_param}')


	def loading_solarwind(self, solar_wind_data='ace'):
		'''
		Loads the solar wind data

		Returns:
			df (pd.dataframe): dataframe containing the solar wind data
		'''
		if not solar_wind_data in ['ace', 'dscovr', 'omni']:
			raise ValueError('Invalid solar wind data source. Must be "ace", "dscovr", or "omni".')
		print('Loading solar wind data....')
		if self.omni:
			self.solarwind = pd.read_csv('../data/SW/omni.csv')
			self.solarwind.set_index('Epoch', inplace=True, drop=True)
			self.solarwind.index = pd.to_datetime(self.solarwind.index, format='%Y-%m-%d %H:%M:%S')
		elif solar_wind_data == 'dscovr':
			self.solarwind = pd.read_csv(self.data_dir + 'dscovr/processed_dscovr_data.csv')
			self.solarwind.set_index('Date_UTC', inplace=True, drop=True)
			self.solarwind.index = pd.to_datetime(self.solarwind.index, format='%Y-%m-%d %H:%M:%S')
			self.solarwind['Vx'] = self.solarwind['Vx']*(-1)
		else:
			self.solarwind = pd.read_feather('../data/SW/ace_data.feather')
			self.solarwind.set_index('ACEepoch', inplace=True, drop=True)
			self.solarwind.index = pd.to_datetime(self.solarwind.index, format='%Y-%m-%d %H:%M:%S')

		return self.solarwind


	def loading_supermag(self, station):
		'''
		Loads the supermag data

		Args:
			station (string): station of interest

		Returns:
			df (pd.dataframe): dataframe containing the supermag data with a datetime index
		'''

		print(f'Loading station {station}....')
		if not self.ml_challenge:
			df = pd.read_feather(self.supermag_dir+station+'.feather')
		else:
			df = pd.read_feather(self.supermag_dir+station+'_ml_challenge.feather')

		if 'DATE_UTC' in df.columns:
			df.rename(columns={'DATE_UTC':'Date_UTC'}, inplace=True)
		if 'DBHT' or 'dbht' not in df.columns:
			df['dbht'] = np.sqrt(((df['N'].diff(1))**2)+((df['E'].diff(1))**2)) # creates the combined dB/dt column
		# limiting the analysis to the nightside
		df.set_index('Date_UTC', inplace=True, drop=True)
		df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:$S')
		df['theta'] = (np.arctan2(df['N'], df['E']) * 180 / np.pi)	# calculates the angle of B_H
		df['cos_theta'] = np.cos(df['theta'] * np.pi / 180)			# calculates the cosine of the angle of B_H
		df['sin_theta'] = np.sin(df['theta'] * np.pi / 180)			# calculates the sine of the angle of B_H

		return df


	def classification_column(self, df, param, percentile=0.99, set_threshold=None):
		'''
		Creating a new column which labels whether there will be a crossing of threshold
			by the param selected in the forecast window.

		Args:
			df (pd.dataframe): dataframe containing the param values.
			param (str): the paramaeter that is being examined for threshold crossings (dBHt for this study).
			thresh (float or list of floats): threshold or list of thresholds to define parameter crossing.
			forecast (int): how far out ahead we begin looking in minutes for threshold crossings.
								If forecast=30, will begin looking 30 minutes ahead.
			window (int): time frame in which we look for a threshold crossing starting at t=forecast.
								If forecast=30, window=30, we look for threshold crossings from t+30 to t+60

		Returns:
			pd.dataframe: df containing a bool column called crossing and a persistance colmun
		'''

		if set_threshold is None:
			# creating the shifted parameter column
			thresh = df[param].quantile(percentile)
		else:
			thresh = set_threshold
		print(f'THIS IS THE SET THRESHOLD: {thresh}')
		# print(f'Threshold: {thresh}')

		df[f'shifted_{param}'] = df[param].shift(-self.forecast)					# creates a new column that is the shifted parameter. Because time moves foreward with increasing

		if self.window > 0:																				# index, the shift time is the negative of the forecast instead of positive.
			indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=self.window)			# Yeah this is annoying, have to create a forward rolling indexer because it won't do it automatically.
			df['window_max'] = df[f'shifted_{param}'].rolling(indexer, min_periods=1).max()		# creates new column in the df labeling the maximum parameter value in the forecast:forecast+window time frame
		# df['pers_max'] = df[param].rolling(0, min_periods=1).max()						# looks backwards to find the max param value in the time history limit
		else:
			df['window_max'] = df[f'shifted_{param}']
		# df.reset_index(drop=False, inplace=True)											# resets the index

		'''This section creates a binary column for each of the thresholds. Binary will be one if the parameter
			goes above the given threshold, and zero if it does not.'''

		conditions = [(df['window_max'] < thresh), (df['window_max'] >= thresh)]			# defining the conditions
		# pers_conditions = [(df['pers_max'] < thresh), (df['pers_max'] >= thresh)]			# defining the conditions for a persistance model

		binary = [0, 1] 																	# 0 if not cross 1 if cross

		df['classification'] = np.select(conditions, binary)						# new column created using the conditions and the binary
		# df['persistance'] = np.select(pers_conditions, binary)				# creating the persistance column
		# df.drop(['pers_max', 'window_max', f'shifted_{param}'], axis=1, inplace=True)			# removes the working columns for memory purposes
		df.drop(['window_max', f'shifted_{param}'], axis=1, inplace=True)			# removes the working columns for memory purposes

		return df, thresh


	def getting_dbdt_dataframe(self):

		dbdt_df = pd.DataFrame(index=pd.date_range(start=self.start_time, end=self.end_time, freq='min'))
		for station in self.region['stations']:
			# loading the station data
			# station_df = pd.read_feather(self.supermag_dir + station + '.feather')
			station_df = self.loading_supermag(station)
			# station_df.set_index('Date_UTC', inplace=True)
			# station_df.index = pd.to_datetime(station_df.index)
			# creating the dbdt time series
			dbdt_df[station] = station_df['dbht']

		return dbdt_df


	def finding_mlt(self):
		'''finding which station has the least missing data and using that to define the mlt for the region'''

		print(f'region keys: {self.region.keys()}')
		if 'mlt_station' in self.region.keys():
			print(f'MLT station already defined for region {self.region_name}')
			return self.mlt_df[self.clusters[self.cluster]['regions'][self.region_name]['mlt_station']]

		else:
			temp_df = self.mlt_df.copy()

			# self.mlt_df['mix'] = self.mlt_df.median(axis=1)
			missing_mlt = temp_df.isnull().sum()
			station = missing_mlt.idxmin()

			print(f'Missing data for each station: {missing_mlt}')
			print(f'Station with the least missing data: {station}')

			self.clusters[self.cluster]['regions'][self.region_name]['mlt_station'] = station

			return self.mlt_df[station]


	def calculating_rsd(self):

		dbdt_df = self.getting_dbdt_dataframe()
		rsd = pd.DataFrame(index=dbdt_df.index)

		# calculating the RSD
		for col in dbdt_df.columns:
			ss = dbdt_df[col]
			temp_df = dbdt_df.drop(col,axis=1)
			ra = temp_df.mean(axis=1)
			rsd[col] = ss-ra

		max_rsd = rsd.max(axis=1)
		max_station = rsd.idxmax(axis=1)
		rsd['max_rsd'] = max_rsd
		rsd['max_station'] = max_station

		return rsd


	def combining_stations_into_regions(self):

		time_period = pd.date_range(start=pd.to_datetime(self.start_time), end=pd.to_datetime(self.end_time), freq='min')

		regional_df = pd.DataFrame(index=time_period)
		self.mlt_df = pd.DataFrame(index=time_period)

		# creating a dataframe for each feature with the twins time period as the index and storing them in a dict
		feature_dfs = {}
		if self.features is not None:
			for feature in self.features:
				feature_dfs[feature] = pd.DataFrame(index=time_period)

		for stat in self.region['stations']:
			df = self.loading_supermag(stat)
			self.mlt_df[stat] = df['MLT']
			if self.features is not None:
				for feature in self.features:
					if feature == 'rsd':
						continue
					feature_dfs[feature][f'{stat}_{feature}'] = df[feature]
		if self.features is not None:
			for feature in self.features:
				if self.mean:
					if feature == 'N' or feature == 'E':
						regional_df[f'{feature}_mean'] = feature_dfs[feature].abs().mean(axis=1)
					else:
						regional_df[f'{feature}_mean'] = feature_dfs[feature].mean(axis=1)
				if self.std:
					regional_df[f'{feature}_std'] = feature_dfs[feature].std(axis=1)
				if self.maximum:
					if feature == 'N' or feature == 'E':
						regional_df[f'{feature}_max'] = feature_dfs[feature].abs().max(axis=1)
					else:
						regional_df[f'{feature}_max'] = feature_dfs[feature].max(axis=1)
				if self.median:
					if feature == 'N' or feature == 'E':
						regional_df[f'{feature}_median'] = feature_dfs[feature].abs().median(axis=1)
					else:
						regional_df[f'{feature}_median'] = feature_dfs[feature].median(axis=1)

		indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=15)
		mlt = self.finding_mlt()
		rsd = self.calculating_rsd()

		regional_df['rsd'] = rsd['max_rsd']
		
		regional_df['rolling_rsd'] = rsd['max_rsd'].rolling(indexer, min_periods=1).max()
		regional_df['MLT'] = mlt
		regional_df['cosMLT'] = np.cos(regional_df['MLT'] * 2 * np.pi * 15 / 360)
		regional_df['sinMLT'] = np.sin(regional_df['MLT'] * 2 * np.pi * 15 / 360)

		if self.classification:

			if not self.ml_challenge:

				print(f'Target parameter: {self.target_param}')
				regional_df, threshold = self.classification_column(df=regional_df, param=self.target_param, percentile=0.99)

				if os.path.exists(self.working_dir+'threshold_dict.pkl'):
					with open(self.working_dir+'threshold_dict.pkl', 'rb') as f:
						threshold_dict = pickle.load(f)
				else:
					threshold_dict = {'rsd':{}, 'dbht_max':{}}
				threshold_dict[self.target_param][self.region_name] = threshold

				with open(self.working_dir+'threshold_dict.pkl', 'wb') as f:
					pickle.dump(threshold_dict, f)
				
				print(f'Region: {self.region_name}, Threshold: {threshold}')
			
			else:
				with open(self.working_dir+'threshold_dict.pkl', 'rb') as f:
					threshold_dict = pickle.load(f)
				
				threshold = threshold_dict[self.target_param][self.region_name]
				regional_df, threshold = self.classification_column(df=regional_df, param=self.target_param, percentile=0.99, set_threshold=threshold)
		return regional_df


	def RegionPreprocessing(self, cluster_dict='cluster_dict.pkl', **kwargs):

		with open(cluster_dict, 'rb') as f:
			self.clusters = pickle.load(f)

		self.region = self.clusters[self.cluster]['regions'][self.region_name]

		supermag_df = self.combining_stations_into_regions()

		with open(cluster_dict, 'wb') as f:
			pickle.dump(self.clusters, f)

		return supermag_df


	def loading_data(self, solar_wind_data='ace', **kwargs):

		# loading all the datasets and dictonaries

		# loading all the datasets and dictonaries
		supermag_df = self.RegionPreprocessing()	# loading the supermag data
		solarwind = self.loading_solarwind(solar_wind_data=solar_wind_data)			# loading the solar wind data
		# converting the solarwind data to log10
		solarwind['logT'] = np.log10(solarwind['T'])
		solarwind.drop(columns=['T'], inplace=True)

		# self.region_df = pd.merge(supermag_df, solarwind, left_index=True, right_index=True, how='inner')
		self.region_df = supermag_df.join(solarwind, how='left')

		return self.region_df

	def storm_extract(self, df, storm_list=None, lead=2220, recovery=2880, target_var=None):

		'''
		Pulling out storms using a defined list of datetime strings, adding a lead and recovery time to it and
		appending each storm to a list which will be later processed.

		Args:
			data (list of pd.dataframes): ACE and supermag data with the test set's already removed.
			lead (int): how much time in hours to add to the beginning of the storm.
			recovery (int): how much recovery time in hours to add to the end of the storm.

		Returns:
			list: ace and supermag dataframes for storm times
			list: np.arrays of shape (n,2) containing a one hot encoded boolean target array
		'''
		storms, y = list(), list()				# initalizing the lists
		all_storms, all_targets = pd.DataFrame(), pd.DataFrame()
		skipped = 0

		# setting the datetime index
		if 'Date_UTC' in df.columns:
			pd.to_datetime(df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
			df.reset_index(drop=True, inplace=True)
			df.set_index('Date_UTC', inplace=True, drop=True)
		else:
			print('Date_UTC not in columns. Check to make sure index is datetime not integer.')

		df.index = pd.to_datetime(df.index)

		if storm_list is None:
			storm_list = pd.read_csv('stormList.csv', header=None, names=['dates'])
		# storm_list = storm_list['dates']

		stime, etime = [], []					# will store the resulting time stamps here then append them to the storm time df

		if isinstance(storm_list, list):		
			storm_list = pd.DataFrame(storm_list, columns=['dates'])		# if the storm list is a list, convert it to a dataframe

		# will loop through the storm dates, create a datetime object for the lead and recovery time stamps and append those to different lists
		if not isinstance(storm_list['dates'][0], pd.Timestamp):
			storm_list['dates'] = pd.to_datetime(storm_list['dates'], format='%Y-%m-%d %H:%M:%S')

		storm_list['stime'] = storm_list['dates'] - pd.Timedelta(minutes=lead)
		storm_list['etime'] = storm_list['dates'] + pd.Timedelta(minutes=recovery)
		storm_list['dates'] = storm_list['dates'].dt.strftime('%Y-%m-%d %H:%M:%S')

		print(storm_list)

		# data_dict = {date: {} for date in storm_list['dates']}
		data_dict = {}

		for start, end, date in zip(storm_list['stime'], storm_list['etime'], storm_list['dates']):		# looping through the storms to remove the data from the larger df
			if start < df.index[0] or end > df.index[-1]:						# if the storm is outside the range of the data, skip it
				# data_dict[date]['storm'] = None
				# data_dict[date]['target'] = None
				# skipped += 1
				continue
			storm = df[(df.index >= start) & (df.index <= end)]

			if storm.shape[0] > 0:
				data_dict[date] = {}
				data_dict[date]['target'] = storm[target_var].values
				storm.drop(target_var, axis=1, inplace=True)
				data_dict[date]['storm'] = storm

			else:
				continue
				# data_dict[date]['storm'] = None
				# data_dict[date]['target'] = None
				# skipped += 1

		print(f'Skipped {skipped} storms.')
		return data_dict


	def split_sequences(self, sequences, targets=None, n_steps=30, include_target=True, model_type='classification', oversample=False, oversample_percentage=1):
		'''
			Takes input from the input array and creates the input and target arrays that can go into the models.

			Args:
				sequences (np.array): input features. Shape = (length of data, number of input features)
				results_y: series data of the targets for each threshold. Shape = (length of data, 1)
				n_steps (int): the time history that will define the 2nd demension of the resulting array.
				include_target (bool): true if there will be a target output. False for the testing data.

			Returns:
				np.array (n, time history, n_features): array for model input
				np.array (n, 1): target array
			'''
		X, y = list(), list()						# creating lists for storing results
		for sequence, target in zip(sequences, targets):	# looping through the sequences and targets
			if isinstance(sequence, pd.DataFrame):
				sequence = sequence.to_numpy()
			for i in range(len(sequence)-n_steps):			# going to the end of the dataframes
				end_ix = i + n_steps						# find the end of this pattern
				if end_ix > len(sequence):					# check if we are beyond the dataset
					break
				
				seq_x = sequence[i:end_ix, :]				# grabs the appropriate chunk of the data
				if include_target:
					if model_type == 'classification':
						seq_y1 = target[end_ix, :]				# gets the appropriate target
					elif model_type == 'regression':
						seq_y1 = target[end_ix]					# gets the appropriate target
					else:
						raise ValueError('Must specify a valid model type. Options are "classification" and "regression".')
					y.append(np.eye(2, dtype='uint8')[seq_y1])
					# y.append(seq_y1)
				X.append(seq_x)
				if oversample:
					if (seq_y1 == 1) or (seq_y1.ndim == 2 and seq_y1[1] == 1):
						if np.random.rand() <= oversample_percentage:
							X.append(seq_x)
							y.append(np.eye(2, dtype='uint8')[seq_y1])
							# y.append(seq_y1)

		return np.array(X), np.array(y)

	def get_dates(self, df):
		'''
		Getting the dates from the dataframes that were used to create the input arrays.

		Args:
			df (pd.DataFrame): dataframe that was used to create the input arrays

		Returns:
			pd.Series: series containing the dates
		'''
		# checking to make sure the index is a datetime object
		# if df is None:
		# 	return pd.Series()
		# else:
		# df.reset_index(drop=False, inplace=True)
		if not isinstance(df.index[0], pd.Timestamp):
			df.index = pd.to_datetime(df.index)
		
		temp_df = df.copy()
		# chopping off the first and last 30 minutes of the data
		temp_df = temp_df.iloc[self.config['time_history']:]

		return pd.Series(temp_df.index)

	def get_features(self):
		'''
		Getting the features that will be used in the model.

		Returns:
			features (list): list of features that will be used in the model.
		'''
		# loading the data
		region_df = self.loading_data()
		# getting the features
		features = region_df.columns

		return features

	def preping_specific_test_storms(self, storm_list, solar_wind_data='ace', lead=1140, recovery=1140):
		'''
		Preparing the specific test storms for the model.

		Returns:
			specific_storms (list): list of the specific test storms
			specific_targets (list): list of the specific test storm targets
			specific_dates (list): list of the specific test storm dates
		'''

		region_df = self.loading_data(solar_wind_data=solar_wind_data)
		if self.vars_to_keep is None:
			# reducing the dataframe to only the features that will be used in the model plus the target variable
			self.vars_to_keep = ['classification', 'dbht_median', 'MAGNITUDE_median', 'sin_theta_std', 'cos_theta_std', 'cosMLT', 'sinMLT',
							'BX_GSE', 'BY_GSM', 'BZ_GSM', 'Vx', 'Vy', 'Vz', 'proton_density', 'logT']
		
		region_df = region_df[self.vars_to_keep]
		# dropping the rows with nans
		region_df.dropna(inplace=True)


		test_dict, stored_test_dates = {}, pd.Series()

		storm_dict = self.storm_extract(df=region_df, storm_list=storm_list, lead=lead, recovery=recovery, target_var='classification')

		storm_dict = {key: value for key, value in storm_dict.items() if value['storm'].shape[0] > 0}

		storms, targets, dates = [value['storm'] for value in storm_dict.values()], [value['target'] for value in storm_dict.values()], \
									[key for key in storm_dict.keys()]

		for storm in storms:
			stored_test_dates = pd.concat([stored_test_dates, self.get_dates(storm)], axis=0)
		
		with open(f'{data_dir}mike_working_dir/including_ion_temp_maps/models/{self.target_param}/region_{self.region_name}_version_{self.version}_scaler.pkl', 'rb') as f:
			scaler = pickle.load(f)
		
		storms = [scaler.transform(x) for x in storms]

		storms, targets = self.split_sequences(storms, targets=targets, n_steps=self.config['time_history'],
												model_type='regression', oversample=False)

		test_dict['storms'], test_dict['targets'], test_dict['dates'] = storms, targets, stored_test_dates

		return test_dict

		# making sure the target variable has been dropped from the input data

		

	def __call__(self):
		'''
		Calling the data prep class without the TWINS data for this version of the model.

		Returns:
			X_train (np.array): training inputs for the model
			X_val (np.array): validation inputs for the model
			X_test (np.array): testing inputs for the model
			y_train (np.array): training targets for the model
			y_val (np.array): validation targets for the model
			y_test (np.array): testing targets for the model

		'''

		region_df = self.loading_data()
		
		if self.vars_to_keep is None:
			# reducing the dataframe to only the features that will be used in the model plus the target variable
			self.vars_to_keep = ['classification', 'dbht_median', 'MAGNITUDE_median', 'sin_theta_std', 'cos_theta_std', 'cosMLT', 'sinMLT',
							'BX_GSE', 'BY_GSM', 'BZ_GSM', 'Vx', 'Vy', 'Vz', 'proton_density', 'logT']
		
		region_df = region_df[self.vars_to_keep]
		print(region_df.head())
		print(region_df.isnull().sum())
		# dropping the rows with nans
		region_df.dropna(inplace=True)

		# loading the data corresponding to the twins maps if it has already been calculated
		if os.path.exists(self.working_dir+f'storm_extraction_region_{self.region_name}_version_{self.version}.pkl'):
			with open(self.working_dir+f'storm_extraction_region_{self.region_name}_version_{self.version}.pkl', 'rb') as f:
				storm_dict = pickle.load(f)

		# if not, calculating the twins maps and extracting the storms
		else:
			print('right here')
			storm_dict = self.storm_extract(df=region_df, lead=1140, recovery=1140, target_var='classification')
			with open(self.working_dir+f'storm_extraction_region_{self.region_name}_version_{self.version}.pkl', 'wb') as f:
				pickle.dump(storm_dict, f)

		# making sure the target variable has been dropped from the input data

		# splitting the data on a day to day basis to reduce data leakage
		specific_test_storms = ['2012-03-09 08:13:00', '2012-03-12 16:55:00', '2013-03-17 20:28:00', '2017-09-08 01:08:00']

		train_dict, val_dict, test_dict = {},{},{}

		specific_storms, specific_targets, specific_dates = [storm_dict[key]['storm'] for key in specific_test_storms],\
															[storm_dict[key]['target'] for key in specific_test_storms], \
																[key for key in specific_test_storms]
		storm_dict = {key: value for key, value in storm_dict.items() if key not in specific_test_storms}
		storm_dict = {key: value for key, value in storm_dict.items() if value['storm'].shape[0] > 0}
		storms, targets, dates = [value['storm'] for value in storm_dict.values()], [value['target'] for value in storm_dict.values()], \
									[key for key in storm_dict.keys()]
		for storm in storms:
			if storm.shape[0] == 0:
				print('Empty storm')
				print(storm)

		train_storms, test_storms, train_targets, test_targets, train_dates, test_dates = train_test_split(storms, targets, dates, 
																							test_size=0.3, shuffle=True, 
																							random_state=self.config['random_seed'])
		test_storms, val_storms, test_targets, val_targets, test_dates, val_dates, = train_test_split(test_storms, test_targets, test_dates, 
																						test_size=0.5, shuffle=True, 
																						random_state=self.config['random_seed'])

		test_storms.extend(specific_storms), test_targets.extend(specific_targets), test_dates.extend(specific_dates)

		stored_train_dates, stored_val_dates, stored_test_dates = pd.Series(), pd.Series(), pd.Series()
		for storm in train_storms:
			stored_train_dates = pd.concat([stored_train_dates, self.get_dates(storm)], axis=0)
		for storm in val_storms:
			stored_val_dates = pd.concat([stored_val_dates, self.get_dates(storm)], axis=0)
		for storm in test_storms:
			stored_test_dates = pd.concat([stored_test_dates, self.get_dates(storm)], axis=0)

		for storm in train_storms:
			if storm.shape[0] == 0:
				print('Empty storm')
				print(storm)

		print(train_storms[0])
		print(f"Dimensions of train_storms: {len(train_storms)} - {train_storms[0].shape} - {train_targets[0].shape}")
		scaling_array = pd.concat(train_storms, axis=0)
		# scaling_array.dropna(inplace=True)
		print(f'Scaling array shape: {scaling_array.shape}')
		scaler = StandardScaler()
		scaler.fit(scaling_array)

		with open(f'models/{self.target_param}/region_{self.region_name}_version_{self.version}_scaler.pkl', 'wb') as f:
			pickle.dump(scaler, f)

		train_storms = [scaler.transform(x) for x in train_storms]
		print('Finished training storms')
		val_storms = [scaler.transform(x) for x in val_storms]
		print('Finished validation storms')
		test_storms = [scaler.transform(x) for x in test_storms]

		# splitting the sequences for input to the CNN
		train_storms, train_targets = self.split_sequences(train_storms, targets=train_targets, n_steps=self.config['time_history'],
											model_type='regression', oversample=self.oversampling)

		val_storms, val_targets = self.split_sequences(val_storms, targets=val_targets, n_steps=self.config['time_history'],
										model_type='regression', oversample=self.oversampling)

		test_storms, test_targets = self.split_sequences(test_storms, targets=test_targets, n_steps=self.config['time_history'],
											model_type='regression', oversample=False)
		# print(f'Train_targets: {train_targets[:10, :]}')

		print(f"Demensions after splitting: Train storms: {train_storms.shape} Train targets: {train_targets.shape} Val storms: {val_storms.shape} Val targets: {val_targets.shape} Test storms: {test_storms.shape} Test targets: {test_targets.shape}")
		
		
		print('Finished testing storms')
		print(f'Train storms: {train_storms[0].shape} Val storms: {val_storms[0].shape} Test storms: {test_storms[0].shape}')
		# saving the scaler
		

		# trainX, trainy, trainD, valX, valy, valD, testX, testy, testD = [],[],[],[],[],[],[],[],[]
		# for storm, target, date in zip(train_storms, train_targets, stored_train_dates):
		# 	# if not np.isnan(storm).any():
		# 	trainX.append(storm)
		# 	trainy.append(target)
		# 	trainD.append(date)
		# for storm, target, date in zip(val_storms, val_targets, stored_val_dates):
		# 	# if not np.isnan(storm).any():
		# 	valX.append(storm)
		# 	valy.append(target)
		# 	valD.append(date)
		# for storm, target, date in zip(test_storms, test_targets, stored_test_dates):
		# 	testX.append(storm)
		# 	testy.append(target)
		# 	testD.append(date)

		# train_dict['storms'], train_dict['targets'], train_dict['dates'] = np.array(trainX), np.array(trainy), np.array(trainD)
		# val_dict['storms'], val_dict['targets'], val_dict['dates'] = np.array(valX), np.array(valy), np.array(valD)
		# test_dict['storms'], test_dict['targets'], test_dict['dates'] = np.array(testX), np.array(testy), np.array(testD)

		train_dict['storms'], train_dict['targets'], train_dict['dates'] = train_storms, train_targets, stored_train_dates
		val_dict['storms'], val_dict['targets'], val_dict['dates'] = val_storms, val_targets, stored_val_dates
		test_dict['storms'], test_dict['targets'], test_dict['dates'] = test_storms, test_targets, stored_test_dates

		with open(f'models/{self.target_param}/region_{self.region_name}_version_{self.version}_data.pkl', 'wb') as f:
			pickle.dump({'train':train_dict, 'val':val_dict, 'test':test_dict}, f)

		# checking the storms for nans and removing those that have them and the conjugate targets

		return train_dict, val_dict, test_dict



		

		