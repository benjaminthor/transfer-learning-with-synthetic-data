import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
# import seaborn as sns
import random
import os
import re
import sys
import time
import datetime
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig,OutputType
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from sktime.datasets import load_from_ucr_tsv_to_dataframe
from sktime.datasets import load_from_tsfile
import neptune.new as neptune
import datetime
import uuid
import pickle
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, matthews_corrcoef, cohen_kappa_score, f1_score, precision_score, recall_score
import yaml
import chardet

class TimeSeriesDataset(Dataset):    
	def __init__(self, X, y, transform=None, trarget_transform=None):
		self.X = X 
		self.y = y
		self.transform = transform
		self.target_transform = trarget_transform
		
	def __len__(self):
		return len(self.y)
	
	def __getitem__(self,idx):
		X = self.X[idx]
		y = self.y[idx]
		if self.transform:
			X = self.transform(X)
		if self.target_transform:
			y = self.target_transform(y)
		return torch.tensor(X), torch.tensor(y)
	

class LSTM_Classifier(nn.Module):
	def __init__(self, input_dim=31, hidden_dim=256, num_layers=1, output_dim=5, dropout=0):
		'''
		input_dim = number of features at each time step 
		hidden_dim = number of features produced by each LSTM cell (in each layer)
		num_layers = number of LSTM layers
		output_dim = number of classes (number of activities)
		'''
		super().__init__()
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
							num_layers=num_layers, batch_first=True, dropout=dropout)
		self.fc = nn.Linear(hidden_dim, output_dim)
		self.softmax = nn.Softmax(dim=1)
		
		
	def forward(self, X):
		_, (h_n, c_n) = self.lstm(X)  # (h_0, c_0) default to zeros
		out = self.fc(h_n[-1,:,:])
		out = self.softmax(out)
		return out
	

class GRU_Classifier(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, num_classes):
		super(GRU_Classifier, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_size, num_classes)
		
	def forward(self, x):
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=x.device)
		out, _ = self.gru(x, h0.detach())
		out = self.fc(out[:, -1, :])
		return out
	
def correct_sizes(sizes):
	corrected_sizes = [s if s % 2 != 0 else s - 1 for s in sizes]
	return corrected_sizes


def pass_through(X):
	return X

class Flatten(nn.Module):
	def __init__(self, out_features):
		super(Flatten, self).__init__()
		self.output_dim = out_features

	def forward(self, x):
		return x.view(-1, self.output_dim)
	
class Reshape(nn.Module):
	def __init__(self, out_shape):
		super(Reshape, self).__init__()
		self.out_shape = out_shape

	def forward(self, x):
		return x.view(-1, *self.out_shape)

class Inception(nn.Module):
	def __init__(self, in_channels,
				 n_filters, 
				kernel_sizes=[9, 19, 39], 
				bottleneck_channels=32, 
				activation=nn.ReLU(), 
				return_indices=False):
		"""
		: param in_channels				Number of input channels (input features)
		: param n_filters				Number of filters per convolution layer => out_channels = 4*n_filters
		: param kernel_sizes			List of kernel sizes for each convolution.
										Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
										This is nessesery because of padding size.
										For correction of kernel_sizes use function "correct_sizes". 
		: param bottleneck_channels		Number of output channels in bottleneck. 
										Bottleneck wont be used if number of in_channels is equal to 1.
		: param activation				Activation function for output tensor (nn.ReLU()). 
		: param return_indices			Indices are needed only if we want to create decoder with InceptionTranspose with MaxUnpool1d. 
		"""
		super(Inception, self).__init__()
		self.return_indices=return_indices
		if in_channels > 1:
			self.bottleneck = nn.Conv1d(
								in_channels=in_channels, 
								out_channels=bottleneck_channels, 
								kernel_size=1, 
								stride=1, 
								bias=False
								)
		else:
			self.bottleneck = pass_through
			bottleneck_channels = 1

		self.conv_from_bottleneck_1 = nn.Conv1d(
										in_channels=bottleneck_channels, 
										out_channels=n_filters, 
										kernel_size=kernel_sizes[0], 
										stride=1, 
										padding=kernel_sizes[0]//2, 
										bias=False
										)
		self.conv_from_bottleneck_2 = nn.Conv1d(
										in_channels=bottleneck_channels, 
										out_channels=n_filters, 
										kernel_size=kernel_sizes[1], 
										stride=1, 
										padding=kernel_sizes[1]//2, 
										bias=False
										)
		self.conv_from_bottleneck_3 = nn.Conv1d(
										in_channels=bottleneck_channels, 
										out_channels=n_filters, 
										kernel_size=kernel_sizes[2], 
										stride=1, 
										padding=kernel_sizes[2]//2, 
										bias=False
										)
		self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, return_indices=return_indices)
		self.conv_from_maxpool = nn.Conv1d(
									in_channels=in_channels, 
									out_channels=n_filters, 
									kernel_size=1, 
									stride=1,
									padding=0, 
									bias=False
									)
		self.batch_norm = nn.BatchNorm1d(num_features=4*n_filters)
		self.activation = activation

	def forward(self, X):
		# step 1
		Z_bottleneck = self.bottleneck(X)
		if self.return_indices:
			Z_maxpool, indices = self.max_pool(X)
		else:
			Z_maxpool = self.max_pool(X)
		# step 2
		Z1 = self.conv_from_bottleneck_1(Z_bottleneck)
		Z2 = self.conv_from_bottleneck_2(Z_bottleneck)
		Z3 = self.conv_from_bottleneck_3(Z_bottleneck)
		Z4 = self.conv_from_maxpool(Z_maxpool)
		# step 3 
		Z = torch.cat([Z1, Z2, Z3, Z4], axis=1)
		Z = self.activation(self.batch_norm(Z))
		if self.return_indices:
			return Z, indices
		else:
			return Z


class InceptionBlock(nn.Module):
	def __init__(self, in_channels, n_filters=32, kernel_sizes=[9,19,39], bottleneck_channels=32, use_residual=True, activation=nn.ReLU(), return_indices=False):
		super(InceptionBlock, self).__init__()
		self.use_residual = use_residual
		self.return_indices = return_indices
		self.activation = activation
		self.inception_1 = Inception(
							in_channels=in_channels,
							n_filters=n_filters,
							kernel_sizes=kernel_sizes,
							bottleneck_channels=bottleneck_channels,
							activation=activation,
							return_indices=return_indices
							)
		self.inception_2 = Inception(
							in_channels=4*n_filters,
							n_filters=n_filters,
							kernel_sizes=kernel_sizes,
							bottleneck_channels=bottleneck_channels,
							activation=activation,
							return_indices=return_indices
							)
		self.inception_3 = Inception(
							in_channels=4*n_filters,
							n_filters=n_filters,
							kernel_sizes=kernel_sizes,
							bottleneck_channels=bottleneck_channels,
							activation=activation,
							return_indices=return_indices
							)	
		if self.use_residual:
			self.residual = nn.Sequential(
								nn.Conv1d(
									in_channels=in_channels, 
									out_channels=4*n_filters, 
									kernel_size=1,
									stride=1,
									padding=0
									),
								nn.BatchNorm1d(
									num_features=4*n_filters
									)
								)

	def forward(self, X):
		if self.return_indices:
			Z, i1 = self.inception_1(X)
			Z, i2 = self.inception_2(Z)
			Z, i3 = self.inception_3(Z)
		else:
			Z = self.inception_1(X)
			Z = self.inception_2(Z)
			Z = self.inception_3(Z)
		if self.use_residual:
			Z = Z + self.residual(X)
			Z = self.activation(Z)
		if self.return_indices:
			return Z,[i1, i2, i3]
		else:
			return Z



class InceptionTranspose(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_sizes=[9, 19, 39], bottleneck_channels=32, activation=nn.ReLU()):
		"""
		: param in_channels				Number of input channels (input features)
		: param n_filters				Number of filters per convolution layer => out_channels = 4*n_filters
		: param kernel_sizes			List of kernel sizes for each convolution.
										Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
										This is nessesery because of padding size.
										For correction of kernel_sizes use function "correct_sizes". 
		: param bottleneck_channels		Number of output channels in bottleneck. 
										Bottleneck wont be used if nuber of in_channels is equal to 1.
		: param activation				Activation function for output tensor (nn.ReLU()). 
		"""
		super(InceptionTranspose, self).__init__()
		self.activation = activation
		self.conv_to_bottleneck_1 = nn.ConvTranspose1d(
										in_channels=in_channels, 
										out_channels=bottleneck_channels, 
										kernel_size=kernel_sizes[0], 
										stride=1, 
										padding=kernel_sizes[0]//2, 
										bias=False
										)
		self.conv_to_bottleneck_2 = nn.ConvTranspose1d(
										in_channels=in_channels, 
										out_channels=bottleneck_channels, 
										kernel_size=kernel_sizes[1], 
										stride=1, 
										padding=kernel_sizes[1]//2, 
										bias=False
										)
		self.conv_to_bottleneck_3 = nn.ConvTranspose1d(
										in_channels=in_channels, 
										out_channels=bottleneck_channels, 
										kernel_size=kernel_sizes[2], 
										stride=1, 
										padding=kernel_sizes[2]//2, 
										bias=False
										)
		self.conv_to_maxpool = nn.Conv1d(
									in_channels=in_channels, 
									out_channels=out_channels, 
									kernel_size=1, 
									stride=1,
									padding=0, 
									bias=False
									)
		self.max_unpool = nn.MaxUnpool1d(kernel_size=3, stride=1, padding=1)
		self.bottleneck = nn.Conv1d(
								in_channels=3*bottleneck_channels, 
								out_channels=out_channels, 
								kernel_size=1, 
								stride=1, 
								bias=False
								)
		self.batch_norm = nn.BatchNorm1d(num_features=out_channels)

		def forward(self, X, indices):
			Z1 = self.conv_to_bottleneck_1(X)
			Z2 = self.conv_to_bottleneck_2(X)
			Z3 = self.conv_to_bottleneck_3(X)
			Z4 = self.conv_to_maxpool(X)

			Z = torch.cat([Z1, Z2, Z3], axis=1)
			MUP = self.max_unpool(Z4, indices)
			BN = self.bottleneck(Z)
			# another possibility insted of sum BN and MUP is adding 2nd bottleneck transposed convolution
			
			return self.activation(self.batch_norm(BN + MUP))


class InceptionTransposeBlock(nn.Module):
	def __init__(self, in_channels, out_channels=32, kernel_sizes=[9,19,39], bottleneck_channels=32, use_residual=True, activation=nn.ReLU()):
		super(InceptionTransposeBlock, self).__init__()
		self.use_residual = use_residual
		self.activation = activation
		self.inception_1 = InceptionTranspose(
							in_channels=in_channels,
							out_channels=in_channels,
							kernel_sizes=kernel_sizes,
							bottleneck_channels=bottleneck_channels,
							activation=activation
							)
		self.inception_2 = InceptionTranspose(
							in_channels=in_channels,
							out_channels=in_channels,
							kernel_sizes=kernel_sizes,
							bottleneck_channels=bottleneck_channels,
							activation=activation
							)
		self.inception_3 = InceptionTranspose(
							in_channels=in_channels,
							out_channels=out_channels,
							kernel_sizes=kernel_sizes,
							bottleneck_channels=bottleneck_channels,
							activation=activation
							)	
		if self.use_residual:
			self.residual = nn.Sequential(
								nn.ConvTranspose1d(
									in_channels=in_channels, 
									out_channels=out_channels, 
									kernel_size=1,
									stride=1,
									padding=0
									),
								nn.BatchNorm1d(
									num_features=out_channels
									)
								)

	def forward(self, X, indices):
		assert len(indices)==3
		Z = self.inception_1(X, indices[2])
		Z = self.inception_2(Z, indices[1])
		Z = self.inception_3(Z, indices[0])
		if self.use_residual:
			Z = Z + self.residual(X)
			Z = self.activation(Z)
		return Z
	

# Functions and Utilities
def preprocess_dgan(df:pd.DataFrame,sequence_length:int):
	df = df.copy(deep=True)
	data = []
	for row in df.iterrows():
		for col in df.columns:
			data.append([row[1][col]])
	data = np.array(data)
	data = data.reshape((df.shape[0], sequence_length, df.shape[1]))
	return data

def create_dgan_param(config_data):
	DGAN_param = {'epochs': config_data['datageneration']['epochs'],
				'attribute_noise_dim': config_data['datageneration']['attribute_noise_dim'],
				'feature_noise_dim': config_data['datageneration']['feature_noise_dim'], 
				'attribute_num_layers': config_data['datageneration']['attribute_num_layers'], 
				'attribute_num_units': config_data['datageneration']['attribute_num_units'], 
				'feature_num_layers':  config_data['datageneration']['feature_num_layers'], 
				'feature_num_units':config_data['datageneration']['feature_num_units'], 
				'use_attribute_discriminator': config_data['datageneration']['use_attribute_discriminator'], 
				'normalization': config_data['datageneration']['normalization'], 
				'apply_feature_scaling': config_data['datageneration']['apply_feature_scaling'], 
				'apply_example_scaling': config_data['datageneration']['apply_example_scaling'], 
				'binary_encoder_cutoff': config_data['datageneration']['binary_encoder_cutoff'], 
				'forget_bias': config_data['datageneration']['forget_bias'], 
				'gradient_penalty_coef':config_data['datageneration']['gradient_penalty_coef'], 
				'attribute_gradient_penalty_coef':config_data['datageneration']['attribute_gradient_penalty_coef'], 
				'attribute_loss_coef': config_data['datageneration']['attribute_loss_coef'], 
				'generator_learning_rate':  config_data['datageneration']['generator_learning_rate'], 
				'generator_beta1':  config_data['datageneration']['generator_beta1'], 
				'discriminator_learning_rate': config_data['datageneration']['discriminator_learning_rate'], 
				'discriminator_beta1': config_data['datageneration']['discriminator_beta1'], 
				'attribute_discriminator_learning_rate':  config_data['datageneration']['attribute_discriminator_learning_rate'], 
				'attribute_discriminator_beta1': config_data['datageneration']['attribute_discriminator_beta1'], 
				'batch_size':  config_data['datageneration']['batch_size'], 
				'discriminator_rounds': config_data['datageneration']['discriminator_rounds'], 
				'generator_rounds': config_data['datageneration']['generator_rounds'],
				'mixed_precision_training': config_data['datageneration']['mixed_precision_training']
				}
	return DGAN_param

def define_criterion(config_data):
	if config_data['pretraining']['criterion'].lower() == "crossentropy":
		criterion = nn.CrossEntropyLoss()

	elif config_data['pretraining']['criterion'].lower() == "bce":
		criterion = nn.BCELoss()
	
	return criterion

def create_experiment_param(config_data):
	Experiment_param ={'experiment state':'data generation',
					'Experiment_id': f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{uuid.uuid4().hex}', # experiment unique ID 
					'Dataset name':config_data['experiment_params']['dataset_name'], # Dataset name most be as generated dataset dir name. it will be used while saving the generated data  
					'usage of original data': config_data['datageneration']['percentage_of_original_data'],   # how mach of the original data set is used to for generating synthetic
					'generate_n_sample':config_data['datageneration']['generate_n_sample'], # control the number of samples to generate per class
					'model type':config_data['datageneration']['generate_n_sample']
					}
	return Experiment_param

def create_experiment_param_new(config_data):
	Experiment_param ={'experiment state':'pretraining_finetuning',
					'Experiment_id': f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{uuid.uuid4().hex}', # experiment unique ID 
					'Dataset name':config_data['experiment_params']['dataset_name'], # Dataset name most be as generated dataset dir name. it will be used while saving the generated data  
					'usage of original data': config_data['datageneration']['percentage_of_original_data'],   # how mach of the original data set is used to for generating synthetic
					'model type':config_data['pretraining']['model_type'],
					'index':config_data['experiment_params']['experiment_index']

					}
	return Experiment_param

# Split the data into train and test sets
def split_dataset_by_label(X, y):
	splits = {}
	unique_labels = np.unique(y)
	for label in unique_labels:
		splits[label] = {'X': np.array(X[y == label]), 'y': np.array(y[y == label])}
		print(f"Now splitting data for label : {label}")
	return splits

# def get_divisor(num:int):
#     divisors = []
#     for i in range(1, num + 1):
#         if num % i == 0:
#             divisors.append(i)
#     return divisors

def train_dgan(data:np.ndarray, DGAN_param:dict, Experiment_param:dict):
	run = neptune.init_run(
	project="astarteam/FinalProject",
	api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhMDI5YzIxMy00NjE1LTQ2MDUtOTk3NS1jNDJhMjIzZDE0NDMifQ==",
	)  # your credentialscredentials
	DGAN_param['sample_len'] = data.shape[1]     # random.choice(get_divisor(data.shape[1])[-3:])
	DGAN_param['max_sequence_len'] = data.shape[1]
	DGAN_param['batch_size'] = min(1000, data.shape[0])
	run['Experiment_param'] = Experiment_param
	run["DGAN_param"] = DGAN_param

	model = DGAN(DGANConfig(
		max_sequence_len=DGAN_param['max_sequence_len'],
		sample_len=DGAN_param['sample_len'],
		batch_size=DGAN_param['batch_size'] ,
		apply_feature_scaling=DGAN_param['apply_feature_scaling'],
		apply_example_scaling=DGAN_param['apply_example_scaling'],
		use_attribute_discriminator=DGAN_param['use_attribute_discriminator'],
		generator_learning_rate=DGAN_param['generator_learning_rate'],
		discriminator_learning_rate=DGAN_param['discriminator_learning_rate'],
		epochs=DGAN_param['epochs'],
		gradient_penalty_coef = DGAN_param['gradient_penalty_coef'],
	))

	model.train_numpy(
		data,
		feature_types=[OutputType.CONTINUOUS] * data.shape[2],
	)
	# run.stop()
	return model

def save_model(model, Experiment_param:dict, label:str):
	# define directory path
	directory_path =f'''dataset/{Experiment_param['Dataset name']}/{Experiment_param['Experiment_id']}/synthetic_models/'''
	# create directory if it doesn't exist
	if not os.path.exists(directory_path):
		os.makedirs(directory_path)
	# define file path
	file_path = os.path.join(directory_path, f'model_{label}.pt')
	# save model
	model.save(file_path)
	

def train_generator_per_label(splitted_data:pd.DataFrame, DGAN_param:dict, Experiment_param:dict):
	models = {}
	for label in splitted_data.keys():
		print(f"Training generator for label {label}")
		X = splitted_data[label]['X']
		DGAN_param['label'] = label
		model = train_dgan(X, DGAN_param, Experiment_param)
		save_model(model, Experiment_param, label)
		models[label] = model
	return models

def generate_data_per_label(models, Experiment_param,config_data):
	generated_data = {}
	synthetic = config_data['datageneration']['percentage_of_original_data']
	for label in models.keys():
		print(f"Generating data for label {label}")
		generated_data[label] = models[label].generate_numpy(Experiment_param['generate_n_sample'])[1]
	concatenated_data = {'X':np.concatenate([generated_data[label] for label in generated_data.keys()]),
				'y':np.concatenate([np.array([label]*Experiment_param['generate_n_sample']) for label in generated_data.keys()])}
	# directory_path =f'''dataset/{Experiment_param['Dataset name']}/{Experiment_param['Experiment_id']}/data/'''
	directory_path =f'''DGAN_data/{Experiment_param['Dataset name']}/{synthetic}/'''

	# create directory if it doesn't exist
	if not os.path.exists(directory_path):
		os.makedirs(directory_path)
	file_path = os.path.join(directory_path, f'generated_data.npy')
	np.save(file_path, concatenated_data)
	return generated_data,concatenated_data

def create_data_loaders(X,y,n_splits:int = 1, validation_size:float=0.2):
	ssf = StratifiedShuffleSplit(n_splits=n_splits, test_size=validation_size)
	train_ind, test_ind = next(ssf.split(X,y))
	train_dataloader = DataLoader(TimeSeriesDataset(X[train_ind],y[train_ind]),batch_size=20,shuffle=True)
	validation_dataloader = DataLoader(TimeSeriesDataset(X[test_ind],y[test_ind]),batch_size=20,shuffle=True)
	return train_dataloader, validation_dataloader

def map_label_int(y):    
	label_to_int = {label: i for i, label in enumerate(np.unique(y))}
	int_to_label = {i: label for label, i in label_to_int.items()}
	y_int = np.array([label_to_int[label] for label in y])
	return label_to_int, int_to_label, y_int

# def train_loop(data_loader, model, device, loss_fn, optimizer, print_every_n=200):
# 	model.train()
# 	size = len(data_loader.dataset)
# 	num_batches = len(data_loader)
# 	train_loss=0
# 	tp=0
# 	for batch,(X,y) in enumerate(data_loader):
# 		X = X.to(device)
# 		y = y.type(torch.LongTensor)
# 		y = y.to(device)
# 		pred = model(X.float())
# 		# print(f'Preds : {pred.argmax(1)}')
# 		# print(f'GT : {y}')
# 		loss = loss_fn(pred,y)
# 		train_loss += loss
# 		# print("Shape of y:", y.shape)
# 		# print("Shape of pred:", pred.shape)
# 		# print("Device of y:", y.device)
# 		# print("Device of pred:", pred.device)
# 		# print(f"Pred type",type(pred))
# 		# print("y Type:",type(y))
# 		# print("Max value in y:", y.max().item())
# 		# print("Min value in y:", y.min().item())


# 		tp += (y==pred.argmax(1)).type(torch.float).sum().item()
# 		optimizer.zero_grad()
# 		loss.backward()
# 		optimizer.step()
# 		loss, current = loss.item(), batch*len(X)
# 		if batch%print_every_n==0:
# 			print(f'loss={loss:.3f}, {current} / {size}')

# 	train_loss /= num_batches
# 	train_acc = tp/size    
# 	print(f'train accuracy = {train_acc}, val_loss = {train_loss:2f}')
# 	return train_loss,train_acc
def train_loop(data_loader, model, device, loss_fn, optimizer, print_every_n=200):
    model.train()
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    train_loss=0;acc=0;train_f1_score=0; train_precision=0; train_recall=0; train_specificity_score=0; train_fpr_score=0
    mcc, cohen_kappa= 0,0
    for batch,(X,y) in enumerate(data_loader):
        X = X.to(device)
        y = y.type(torch.LongTensor)
        y = y.to(device)
        pred = model(X.float())
        loss = loss_fn(pred,y)
        train_loss += loss
        pred= pred.argmax(1)
        acc += (y==pred).type(torch.float).sum().item()
        y = y.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        train_f1_score+= f1_score(y,pred,average='micro')
        train_precision+= precision_score(y,pred,average='micro')
        train_recall+= recall_score(y,pred,average='micro')
        mcc += matthews_corrcoef(y, pred)
        cohen_kappa += cohen_kappa_score(y, pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss, current = loss.item(), batch*len(X)
        if batch%print_every_n==0:
            print(f'loss={loss:.3f}, {current} / {size}')

    train_loss /= num_batches
    train_acc = acc/size
    train_f1_score /= num_batches
    train_precision /= num_batches
    train_recall /= num_batches
    mcc /= num_batches
    cohen_kappa /= num_batches

    print(f'train accuracy = {train_acc}, val_loss = {train_loss:2f}')
    return train_loss, train_acc, train_f1_score, train_precision, train_recall, mcc, cohen_kappa

# def validation_loop(data_loader,model,device,loss_fn):
# 	model.eval()
# 	size=len(data_loader.dataset)
# 	num_batches = len(data_loader)
# 	val_loss=0
# 	tp=0
# 	with torch.no_grad():
# 		for X,y in data_loader:
# 			X = X.to(device)
# 			y = y.type(torch.LongTensor)
# 			y = y.to(device)
# 			pred = model(X.float())
# 			val_loss += loss_fn(pred,y).item()
# 			tp += (y==pred.argmax(1)).type(torch.float).sum().item()
		
# 	val_loss /= num_batches
# 	val_acc = tp/size
# 	print(f'validation accuracy = {val_acc}, val_loss = {val_loss:2f}')
# 	return val_loss,val_acc
def validation_loop(data_loader,model,device,loss_fn):
      model.eval()
      size=len(data_loader.dataset)
      num_batches = len(data_loader)
      val_loss=0; acc=0; val_f1_score=0; val_precision=0; val_recall=0
      mcc_val, cohen_kappa_val = 0,0
      with torch.no_grad():
            for X,y in data_loader:
                  X = X.to(device)
                  y = y.type(torch.LongTensor)
                  y = y.to(device)
                  pred = model(X.float())
                  val_loss += loss_fn(pred,y).item()
                  pred= pred.argmax(1)
                  acc += (y==pred).type(torch.float).sum().item()
                  y = y.cpu().detach().numpy()
                  pred = pred.cpu().detach().numpy()
                  val_f1_score+= f1_score(y,pred,average='micro')
                  val_precision+= precision_score(y,pred,average='micro')
                  val_recall+= recall_score(y,pred,average='micro')
                  mcc_val += matthews_corrcoef(y, pred)
                  cohen_kappa_val += cohen_kappa_score(y, pred)

      val_loss /=num_batches
      val_acc = acc/size
      val_f1_score /= num_batches
      val_precision /= num_batches
      val_recall /= num_batches
      mcc_val /= num_batches
      cohen_kappa_val /= num_batches

      print(f'validation accuracy = {val_acc}, val_loss = {val_loss:2f}')
      return val_loss,val_acc, val_f1_score, val_precision, val_recall, mcc_val, cohen_kappa_val

def pretrain_and_finetune(synthetic_dataloader,train_dataloader,validation_dataloader,model,device,criterion,optimizer,run_param,experiment_param):
	print(f'''Pretraining : {run_param['epochs']} epochs''')
	run = neptune.init_run(
	project="astarteam/FinalProject",
	api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhMDI5YzIxMy00NjE1LTQ2MDUtOTk3NS1jNDJhMjIzZDE0NDMifQ==")  # your credentialscredentials
	directory_path = f'''dataset/{experiment_param['Dataset name']}/{experiment_param['Experiment_id']}/{experiment_param['experiment state']}/'''

	if not os.path.exists(directory_path):
		os.makedirs(directory_path)

	run["parameters"] = run_param
	run['Experiment_param'] = experiment_param
	best_loss = np.inf
	for epoch in range(run_param['epochs']):
	# Train
		train_loss, train_acc, train_f1_score, train_precision, train_recall, mcc_train, cohen_kappa_train = train_loop(data_loader = synthetic_dataloader, model=model, device = device, loss_fn = criterion, optimizer = optimizer)
		run["pretrain/accuracy"].log(train_acc)
		run["pretrain/loss"].log(train_loss)
		run["pretrain/f1_score"].log(train_f1_score)
		run["pretrain/precision_score"].log(train_precision)
		run["pretrain/recall_score"].log(train_recall)
		run["pretrain/matthews_corrcoef"].log(mcc_train)
		run["pretrain/cohen_kappa"].log(cohen_kappa_train)
		if train_acc >= 0.99:
			print(f"Accuracy has reached 0.99, stopping pretraining at epoch :{epoch}.")
			break


	print("Finished Pre-training")

	# Save pretrained model
	torch.save(model.state_dict(), os.path.join(directory_path, f'pretrained_model.pt'))

	# Load pretrained model
	model.load_state_dict(torch.load(os.path.join(directory_path, f'pretrained_model.pt')))
	print("Pretrained model is loaded succesfully")

	# Fine-tune
	print(f'''Finetuning : {run_param['epochs']} epochs''')
	for epoch in range(run_param['epochs']):
	# Train
		train_loss, train_acc, train_f1_score, train_precision, train_recall, mcc_train, cohen_kappa_train = train_loop(train_dataloader, model, device, criterion, optimizer)
		run["finetune_training/accuracy"].log(train_acc)
		run["finetune_training/loss"].log(train_loss)
		run["finetune_training/f1_score"].log(train_f1_score)
		run["finetune_training/precision_score"].log(train_precision)
		run["finetune_training/recall_score"].log(train_recall)
		run["finetune_training/matthews_corrcoef"].log(mcc_train)
		run["finetune_training/cohen_kappa"].log(cohen_kappa_train)
	# Validation
		val_loss,val_acc, val_f1_score, val_precision, val_recall, mcc_val, cohen_kappa_val = validation_loop(validation_dataloader, model, device, criterion)
		run["finetune_validation/accuracy"].log(val_acc)
		run["finetune_validation/loss"].log(val_loss)
		run["finetune_validation/f1_score"].log(val_f1_score)
		run["finetune_validation/precision_score"].log(val_precision)
		run["finetune_validation/recall_score"].log(val_recall)
		run["finetune_validation/matthews_corrcoef"].log(mcc_val)
		run["finetune_validation/cohen_kappa"].log(cohen_kappa_val)

		if val_loss < best_loss:
			# save the model
			model_path = f'''{directory_path}model_{experiment_param["model type"]}_.pt'''
			torch.save(model.state_dict(), model_path)
			best_loss = val_loss
			early_stopping_counter = 0
		else:
			early_stopping_counter += 1
			# if the early stopping counter has reached the patience, stop training
			if early_stopping_counter == run_param['patience']:
				break
		time.sleep(2)
	print("Finished Training and validation, now uploading to Neptune.")
	run.stop()
	return model_path

def train_and_log(train_dataloader,validation_dataloader,model,device,criterion,optimizer,save_model, run_param,Experiment_param):
	run = neptune.init_run(
	project="astarteam/FinalProject",
	api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhMDI5YzIxMy00NjE1LTQ2MDUtOTk3NS1jNDJhMjIzZDE0NDMifQ==")  # your credentialscredentials
	directory_path = f'''dataset/{Experiment_param['Dataset name']}/{Experiment_param['Experiment_id']}/{Experiment_param['experiment state']}/'''
	if not os.path.exists(directory_path):
		os.makedirs(directory_path)
	run["parameters"] = run_param
	run['Experiment_param'] = Experiment_param
	best_loss = np.inf
	# define the number of epochs and early stopping patience
	for epoch in range(run_param['epochs']):
		#Train
		train_loss, train_acc, train_f1_score, train_precision, train_recall, mcc_train, cohen_kappa_train = train_loop(train_dataloader, model, device, criterion, optimizer)
		run["train/accuracy"].log(train_acc)
		run["train/loss"].log(train_loss)
		run["train/f1_score"].log(train_f1_score)
		run["train/precision_score"].log(train_precision)
		run["train/recall_score"].log(train_recall)
		run["train/matthews_corrcoef"].log(mcc_train)
		run["train/cohen_kappa"].log(cohen_kappa_train)

		#Evaluate
		val_loss,val_acc, val_f1_score, val_precision, val_recall, mcc_val, cohen_kappa_val = validation_loop(validation_dataloader, model, device, criterion)
		run["validation/accuracy"].log(val_acc)
		run["validation/loss"].log(val_loss)
		run["validation/f1_score"].log(val_f1_score)
		run["validation/precision_score"].log(val_precision)
		run["validation/recall_score"].log(val_recall)
		run["validation/matthews_corrcoef"].log(mcc_val)
		run["validation/cohen_kappa"].log(cohen_kappa_val)

		if val_loss < best_loss:
			if save_model:
				# save the model
				model_path = f'''{directory_path}model_{Experiment_param["model type"]}_.pt'''
				torch.save(model.state_dict(), model_path)
			best_loss = val_loss
			early_stopping_counter = 0
		else:
			early_stopping_counter += 1
		# if the early stopping counter has reached the patience, stop training
		if early_stopping_counter == run_param['patience']:
			break
	time.sleep(2)
	print("Finished Training and validation, now uploading to Neptune.")
	run.stop()
	# return model_path

# def train_and_log(train_dataloader,validation_dataloader,model,device,criterion,optimizer,save_model, run_param,Experiment_param):
# 	run = neptune.init_run(
# 	project="astarteam/FinalProject",
# 	api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhMDI5YzIxMy00NjE1LTQ2MDUtOTk3NS1jNDJhMjIzZDE0NDMifQ==")  # your credentialscredentials
# 	directory_path = f'''dataset/{Experiment_param['Dataset name']}/{Experiment_param['Experiment_id']}/{Experiment_param['experiment state']}/'''
# 	if not os.path.exists(directory_path):
# 		os.makedirs(directory_path)
# 	run["parameters"] = run_param
# 	run['Experiment_param'] = Experiment_param
# 	best_loss = np.inf
# 	# define the number of epochs and early stopping patience
# 	for epoch in range(run_param['epochs']):
# 		# Train
# 		train_loss, train_acc = train_loop(train_dataloader, model, device, criterion, optimizer)
# 		run["train/accuracy"].log(train_acc)
# 		run["train/loss"].log(train_loss)
# 		#Evaluate
# 		val_loss, val_acc = validation_loop(validation_dataloader, model, device, criterion)
# 		run["validation/accuracy"].log(val_acc)
# 		run["validation/loss"].log(val_loss)
		
		
# 		if val_loss < best_loss:
# 			if save_model:
# 				# save the model
# 				model_path = f'''{directory_path}model_{Experiment_param["model type"]}_.pt'''
# 				torch.save(model.state_dict(), model_path)
# 			best_loss = val_loss
# 			early_stopping_counter = 0
# 		else:
# 			early_stopping_counter += 1
# 		# if the early stopping counter has reached the patience, stop training
# 		if early_stopping_counter == run_param['patience']:
# 			break
# 	time.sleep(2)
# 	print("Finished Training and validation, now uploading to Neptune.")
# 	run.stop()
# 	return model_path

def detect_file_encoding(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
    return result['encoding']

def extract_metadata(path):
    # Replace 'file_path' with the path to your file
    file_path = path

    encoding = detect_file_encoding(file_path)

    with open(file_path, 'r', encoding=encoding) as file:
        file_content = file.read()

    # Extract the seriesLength value
    series_length_match = re.search(r"@seriesLength (\d+)", file_content)
    if series_length_match:
        series_length = int(series_length_match.group(1))
        print("Series Length:", series_length)
    else:
        print("Series Length not found")

    # Extract the dimensions value
    dimensions_match = re.search(r"@dimensions (\d+)", file_content)
    if dimensions_match:
        dimensions = int(dimensions_match.group(1))
        print("Dimensions:", dimensions)
    else:
        print("Dimensions not found")

    # Extract the unique labels from the @classLabel line
    class_label_match = re.search(r"@classLabel true (.+)", file_content)
    if class_label_match:
        class_labels_str = class_label_match.group(1)
        class_labels = [label for label in class_labels_str.split()]
        print("Unique labels:", class_labels)
    else:
        print("Class labels not found")

    return series_length,dimensions,len(class_labels)


def read_yaml_config(CONFIG):
    with open(CONFIG, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            print(exc)
            return None
	

def create_model_based_on_config(model_str,config_data):
	if model_str == "GRU":   
		model = GRU_Classifier(input_size=config_data['experiment_params']['num_features'],
								hidden_size=config_data['pretraining']['hidden_size'],
								num_layers=config_data['pretraining']['num_layers_stacked'], 
								num_classes=config_data['experiment_params']['num_classes'])
	elif model_str == "LSTM":
		model = LSTM_Classifier(input_dim=config_data['experiment_params']['num_features'],
								hidden_dim=config_data['pretraining']['hidden_size'],
								num_layers=config_data['pretraining']['num_layers_stacked'], 
								output_dim=config_data['experiment_params']['num_classes'],
								dropout=config_data['pretraining']['dropout'])
	elif model_str == "inceptionTime":
		model = nn.Sequential(

						Reshape((config_data['experiment_params']['num_features'],
								config_data['experiment_params']['sequence_length'])),

						InceptionBlock(
							in_channels=config_data['experiment_params']['num_features'], 
							n_filters=32, 
							kernel_sizes=[5, 11, 23],
							bottleneck_channels=32,
							use_residual=True,
							activation=nn.ReLU()
						),
						InceptionBlock(
							in_channels=32*4, 
							n_filters=32, 
							kernel_sizes=[5, 11, 23],
							bottleneck_channels=32,
							use_residual=True,
							activation=nn.ReLU()
						),
						nn.AdaptiveAvgPool1d(output_size=1),
						Flatten(out_features=32*4*1),
						nn.Linear(in_features=4*32*1, out_features=config_data['experiment_params']['num_classes'])
			)
	elif model_str == "TCN":
		pass
	elif model_str == "Transformer":
		pass
	elif model_str == "CNN":
		pass
	elif model_str == "RNN":
		pass
	elif model_str == "NN":
		pass
	return model



def get_latest_model_path(parent_directory):
	# Get the list of all directories
	all_dirs = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]

	# Get the last modified directory
	last_modified_dir = max(all_dirs, key=lambda d: os.path.getmtime(os.path.join(parent_directory, d)))

	# Create the path for the 'pretraining' subdirectory
	pretraining_dir = os.path.join(parent_directory, last_modified_dir, 'pretraining')

	# Find all the .pt files in the 'pretraining' subdirectory
	model_files = glob.glob(os.path.join(pretraining_dir, '*.pt'))

	# Get the creation time of the first file
	last_file = model_files[0]
	last_created_file = last_file
	last_created_time = os.path.getctime(os.path.join(pretraining_dir, last_created_file))

	for file in model_files:
		file_path = os.path.join(pretraining_dir, file)
		file_creation_time = os.path.getctime(file_path)

		if file_creation_time > last_created_time:
			last_created_file = file
			last_created_time = file_creation_time

	# Return the first .pt file found (if any)
	if last_created_file:
		print(f"The last modified file in the directory is: {last_created_file}")
		return last_created_file
	else:
		print("The directory is empty.")
		return None


	
	# last_modified_file = max(files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
	# return last_modified_file

