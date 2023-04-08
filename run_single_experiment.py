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
from sktime.datasets import load_from_ucr_tsv_to_dataframe
from sktime.datasets import load_from_tsfile
import neptune.new as neptune
import datetime
import uuid
import pickle
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import yaml
from utilities_helper import *

def read_yaml_config(file_path):
    with open(file_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            print(exc)
            return None
if __name__ == '__main__':

    config_path = r'FinalProject\FinalProject\config.yaml'
    config_data = read_yaml_config(config_path)

    # Read The data
    X_train,y_train = load_from_tsfile(f"{config_data['experiment_params']['root_path']}\Datasets\{config_data['experiment_params']['dataset_name']}\{config_data['experiment_params']['dataset_name']}_TRAIN.ts")
    X_test,y_test = load_from_tsfile(f"{config_data['experiment_params']['root_path']}\Datasets\{config_data['experiment_params']['dataset_name']}\{config_data['experiment_params']['dataset_name']}_TEST.ts")

    # Split  the data based on config ratio
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1-config_data['preprocessing']['split_ratio'],shuffle=True)

    #Take X ratio out of the training data to be used to train the generator
    X_train, X_train_gen, y_train, y_train_gen = train_test_split(X_train, y_train, test_size=config_data['datageneration']['percentage_of_original_data'],shuffle=True)

    #relevant preprocessing
    X_train = preprocess_dgan(X_train,config_data['datageneration']['max_sequence_len'])
    X_val = preprocess_dgan(X_val,config_data['datageneration']['max_sequence_len'])
    X_test = preprocess_dgan(X_test,config_data['datageneration']['max_sequence_len'])
    X_train_gen = preprocess_dgan(X_train_gen,config_data['datageneration']['max_sequence_len'])
    # Print the shapes of all the data that was preprocessed
    print(f"X_train data shape {X_train.shape}")
    print(f"X_val data shape {X_val.shape}")
    print(f"X_test data shape {X_test.shape}")
    print(f"X_train_Generated data shape {X_train_gen.shape}")


    # Split training data by labels
    split_data = split_dataset_by_label(X_train_gen,y_train_gen)
    print(f"Split data shape {split_data['walking']['X'].shape}")
    # add params to train generator so we can track the dataset generating process  
    Experiment_param ={'experiment state':'data generation',
                    'Experiment_id': f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{uuid.uuid4().hex}', # experiment unique ID 
                    'Dataset name':config_data['experiment_params']['dataset_name'], # Dataset name most be as generated dataset dir name. it will be used while saving the generated data  
                    'usage of original data': config_data['datageneration']['percentage_of_original_data'],   # how mach of the original data set is used to for generating synthetic
                    'generate_n_sample':config_data['datageneration']['generate_n_sample'], # control the number of samples to generate per class
                    'model type':config_data['datageneration']['generate_n_sample']
                    }


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

    # Train the generators
    models = train_generator_per_label(split_data, DGAN_param, Experiment_param)
    generated_data,concatenated_data = generate_data_per_label(models, Experiment_param)
    label_to_int, int_to_label, concatenated_data['y'] = map_label_int(concatenated_data['y'])
    #TODO: Need to check if this is done correctly
    train_dataloader, validation_dataloader = create_data_loaders(concatenated_data['X'],concatenated_data['y'],n_splits=1, validation_size=0.2)

    # Pretraining parameters from config
    output_dim = config_data['experiment_params']['num_classes']
    lr = config_data['pretraining']['learning_rate']
    best_acc = 0
    best_loss = np.inf
    if config_data['pretraining']['model_type'] == "GRU":   
        model = GRU_Classifier(input_size=config_data['experiment_params']['num_features'],
                                hidden_size=config_data['pretraining']['hidden_size'],
                                num_layers=config_data['pretraining']['num_layers_stacked'], 
                                num_classes=config_data['experiment_params']['num_classes'])
    elif config_data['pretraining']['model_type'] == "LSTM":
        model = LSTM_Classifier(input_dim=config_data['experiment_params']['num_features'],
                                hidden_dim=config_data['pretraining']['hidden_size'],
                                num_layers=config_data['pretraining']['num_layers_stacked'], 
                                output_dim=config_data['experiment_params']['num_classes'],
                                dropout=config_data['pretraining']['dropout'])
    elif config_data['pretraining']['model_type'] == "InceptionTime":
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
    elif config_data['pretraining']['model_type'] == "TCN":
        pass
    elif config_data['pretraining']['model_type'] == "Transformer":
        pass
    elif config_data['pretraining']['model_type'] == "CNN":
        pass
    elif config_data['pretraining']['model_type'] == "RNN":
        pass
    elif config_data['pretraining']['model_type'] == "NN":
        pass

    model.to(device)

    if config_data['pretraining']['criterion'].lower() == "crossentropy":
        criterion = nn.CrossEntropyLoss()
    elif config_data['pretraining']['criterion'].lower() == "bce":
        criterion = nn.BCELoss()


    if config_data['pretraining']['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif config_data['pretraining']['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif config_data['pretraining']['optimizer'].lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    if config_data['pretraining']['save_each_epoch'] == True:
        save_each_epoch = True
    else:
        save_each_epoch = False

    run_param = {"epochs":config_data['pretraining']['epochs'],
                "patience":config_data['pretraining']['patience'],
                "batch_size": config_data['pretraining']['batch_size'],
                "learning_rate": lr, 
                "criterion":config_data['pretraining']['criterion'],
                "optimizer": config_data['pretraining']['optimizer']}

    Experiment_param['model type'] = config_data['pretraining']['model_type']
    Experiment_param['experiment state'] = 'pretraining'
    #Pretrain the model on synthetic data
    train_and_log(train_dataloader,validation_dataloader,model,device,criterion,optimizer,save_each_epoch,run_param,Experiment_param)
    directory =f'''{config_data['experiment_params']['root_path']}\dataset\{config_data['experiment_params']['dataset_name']}'''
    last_modified_file = get_last_modified_file(directory)

    if last_modified_file:
        print(f"The last modified file in the directory is: {last_modified_file}")
    else:
        print("The directory is empty.")
    #Now finetuning
    print('Now finetuning...')

    # Load the pre-trained model
    print('Loading pretrained model')
    model.load_state_dict(torch.load(last_modified_file))
    model.to(device)


    print('Creating dataloaders for the original data...')
    train_dataloader = DataLoader(TimeSeriesDataset(X_train,y_train),batch_size=config_data['finetuning']['batch_size'],shuffle=config_data['finetuning']['shuffle'])
    validation_dataloader = DataLoader(TimeSeriesDataset(X_val,y_val),batch_size=config_data['finetuning']['batch_size'],shuffle=config_data['finetuning']['shuffle'])
    test_dataloader = DataLoader(TimeSeriesDataset(X_test,y_test),batch_size=config_data['finetuning']['batch_size'],shuffle=config_data['finetuning']['shuffle'])

    print('Finetuning - Reading config data from YAML...')
    lr = config_data['finetuning']['learning_rate']
    best_acc = 0
    best_loss = np.inf
    epochs =  config_data['finetuning']['epochs']

    if config_data['finetuning']['criterion'].lower() == "crossentropy":
        criterion = nn.CrossEntropyLoss()
    elif config_data['finetuning']['criterion'].lower() == "bce":
        criterion = nn.BCELoss()


    if config_data['finetuning']['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif config_data['finetuning']['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif config_data['finetuning']['optimizer'].lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    Experiment_param['model type'] = config_data['finetuning']['model_type']
    Experiment_param['experiment state'] = 'fine-tuning'
    run_param = {"epochs":config_data['finetuning']['epochs'],
                "patience":config_data['finetuning']['patience'],
                "batch_size": config_data['finetuning']['batch_size'],
                "learning_rate": lr, 
                "criterion":config_data['finetuning']['criterion'],
                "optimizer": config_data['finetuning']['optimizer']}

    print("Finetuning the model...")
    train_and_log(train_dataloader,validation_dataloader,model,device,criterion,optimizer,save_each_epoch)


