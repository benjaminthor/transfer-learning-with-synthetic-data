import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
# import seaborn as sns
import random
import os
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
from sktime.datasets import load_from_ucr_tsv_to_dataframe, load_from_tsfile
import neptune.new as neptune
import datetime
import uuid
import pickle
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import yaml
from utilities_helper import *

def read_yaml_config(CONFIG):
    with open(CONFIG, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            print(exc)
            return None
        
if __name__ == '__main__':
    print("Generating data for a single dataset !")
    ROOT = os.chdir(os.path.dirname(os.path.abspath(__file__)))
    CONFIG = os.path.join(os.getcwd(),"config.yaml")
    config_data = read_yaml_config(CONFIG)
    parent_directory =f'''{os.getcwd()}\dataset\{config_data['experiment_params']['dataset_name']}'''
    # Read The data
    X_train,y_train = load_from_tsfile(f"{config_data['experiment_params']['root_path']}\Datasets\{config_data['experiment_params']['dataset_name']}\{config_data['experiment_params']['dataset_name']}_TRAIN.ts")
    #Take X ratio out of the training data to be used to train the generator
    _, X_train_gen, _, y_train_gen = train_test_split(X_train, y_train, test_size=config_data['datageneration']['percentage_of_original_data'],shuffle=True)

    #relevant preprocessing
    X_train = preprocess_dgan(X_train,config_data['datageneration']['max_sequence_len'])
    X_train_gen = preprocess_dgan(X_train_gen,config_data['datageneration']['max_sequence_len'])

    # Print the shapes of all the data that was preprocessed
    print(f"X_train data shape {X_train.shape}")
    print(f"X_train_Generated data shape {X_train_gen.shape}")

    # Map the labels to integers
    _,__, y_train = map_label_int(y_train)

    # Setup for finetuning
    print('Creating dataloaders for the original data...')
    train_dataloader = DataLoader(TimeSeriesDataset(X_train,y_train),batch_size=config_data['finetuning']['batch_size'],shuffle=config_data['finetuning']['shuffle'])
        
    # Split training data by labels
    split_data = split_dataset_by_label(X_train_gen,y_train_gen)

    #create experiment parameters
    Experiment_param = create_experiment_param(config_data)
    
    # Create the parameters for the DGAN
    DGAN_param = create_dgan_param(config_data)
    
    # Train the generators
    models = train_generator_per_label(split_data, DGAN_param, Experiment_param)
    generated_data,concatenated_data = generate_data_per_label(models, Experiment_param,config_data)