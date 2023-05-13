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
    print("Running Single Experiment !")
    ROOT = os.chdir(os.path.dirname(os.path.abspath(__file__)))
    CONFIG = os.path.join(os.getcwd(),"config.yaml")
    config_data = read_yaml_config(CONFIG)
    parent_directory =f'''{os.getcwd()}\dataset\{config_data['experiment_params']['dataset_name']}'''


    # Read The data
    X_train,y_train = load_from_tsfile(f"{config_data['experiment_params']['root_path']}\Datasets\{config_data['experiment_params']['dataset_name']}\{config_data['experiment_params']['dataset_name']}_TRAIN.ts")
    X_test,y_test = load_from_tsfile(f"{config_data['experiment_params']['root_path']}\Datasets\{config_data['experiment_params']['dataset_name']}\{config_data['experiment_params']['dataset_name']}_TEST.ts")

    # Split  the data based on config ratio
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1-config_data['preprocessing']['split_ratio'],shuffle=True)

    #Take X ratio out of the training data to be used to train the generator
    _, X_train_gen, _, y_train_gen = train_test_split(X_train, y_train, test_size=config_data['datageneration']['percentage_of_original_data'],shuffle=True)

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

    # Map the labels to integers
    _,__, y_train = map_label_int(y_train)
    _,__,y_val = map_label_int(y_val)
    _,__,y_test = map_label_int(y_test)
    # Setup for finetuning
    print('Creating dataloaders for the original data...')
    train_dataloader = DataLoader(TimeSeriesDataset(X_train,y_train),batch_size=config_data['finetuning']['batch_size'],shuffle=config_data['finetuning']['shuffle'])
    validation_dataloader = DataLoader(TimeSeriesDataset(X_val,y_val),batch_size=config_data['finetuning']['batch_size'],shuffle=config_data['finetuning']['shuffle'])
    test_dataloader = DataLoader(TimeSeriesDataset(X_test,y_test),batch_size=config_data['finetuning']['batch_size'],shuffle=config_data['finetuning']['shuffle'])
    
    #Now finetuning
    print('Now finetuning...')
    if config_data['finetuning']['criterion'].lower() == "crossentropy":
        criterion = nn.CrossEntropyLoss()
    elif config_data['finetuning']['criterion'].lower() == "bce":
        criterion = nn.BCELoss()
    lr = config_data['finetuning']['learning_rate']
    epochs =  config_data['finetuning']['epochs']

    # Run Parameters
    run_param = {"epochs":config_data['pretraining']['epochs'],
            "patience":config_data['pretraining']['patience'],
            "batch_size": config_data['pretraining']['batch_size'],
            "learning_rate": lr, 
            "criterion":config_data['pretraining']['criterion'],
            "optimizer": config_data['pretraining']['optimizer']}
    
    # Split training data by labels
    split_data = split_dataset_by_label(X_train_gen,y_train_gen)

    #create experiment parameters
    Experiment_param = create_experiment_param(config_data)
    
    # Create the parameters for the DGAN
    DGAN_param = create_dgan_param(config_data)
    
    # Train the generators
    models = train_generator_per_label(split_data, DGAN_param, Experiment_param)
    generated_data,concatenated_data = generate_data_per_label(models, Experiment_param,config_data)
    
    label_to_int, int_to_label, concatenated_data['y'] = map_label_int(concatenated_data['y'])

    #
    train_dataloader, validation_dataloader = create_data_loaders(concatenated_data['X'],concatenated_data['y'],n_splits=1, validation_size=0.2)

    # Pretraining parameters from config
    output_dim = config_data['experiment_params']['num_classes']
    lr = config_data['pretraining']['learning_rate']
    best_acc = 0
    best_loss = np.inf

    criterion = define_criterion(config_data)
    

    if config_data['pretraining']['save_each_epoch'] == True:
        save_each_epoch = True

    else:
        save_each_epoch = False

    multiple_experiments  = config_data['pretraining']['multiple_experiments']

    if multiple_experiments == True:
        #Models Creation
        visit_information = {}
        models = {}
        for model in config_data['pretraining']['models_list']:
            models[model] = create_model_based_on_config(model,config_data)

        # Pretraining models
        for model_type in models.keys():
            Experiment_param['model type'] = model_type
            Experiment_param['experiment state'] = 'pretraining'
            model = models[model_type]
            model.to(device)
            if config_data['pretraining']['optimizer'].lower() == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            elif config_data['pretraining']['optimizer'].lower() == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            elif config_data['pretraining']['optimizer'].lower() == 'rmsprop':
                optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            model_path = train_and_log(train_dataloader,validation_dataloader,model,device,criterion,optimizer,save_each_epoch,run_param,Experiment_param)
            visit_information[model_type] = {'model_path':model_path,'last_modified_file':get_latest_model_path(parent_directory)}

        # Finetuning models
        for info in visit_information:
            last_modified_file = visit_information[info]['model_path']

            print('Loading pretrained model')
            model = models[info]
            model.load_state_dict(torch.load(last_modified_file))
            model.to(device)

            print('Finetuning - Reading config data from YAML...')
            best_acc = 0
            best_loss = np.inf

            if config_data['finetuning']['optimizer'].lower() == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            elif config_data['finetuning']['optimizer'].lower() == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            elif config_data['finetuning']['optimizer'].lower() == 'rmsprop':
                optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

            Experiment_param['model type'] = info
            Experiment_param['experiment state'] = 'fine-tuning'
            run_param = {"epochs":config_data['finetuning']['epochs'],
                        "patience":config_data['finetuning']['patience'],
                        "batch_size": config_data['finetuning']['batch_size'],
                        "learning_rate": lr, 
                        "criterion":config_data['finetuning']['criterion'],
                        "optimizer": config_data['finetuning']['optimizer']}
            
            print("Finetuning the model...")
            train_and_log(train_dataloader,validation_dataloader,model,device,criterion,optimizer,save_each_epoch,run_param,Experiment_param)
      
    else:
        # Create the model
        model = create_model_based_on_config(config_data['pretraining']['model_type'],config_data)
        model.to(device)
        # Pretraining
        if config_data['pretraining']['optimizer'].lower() == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif config_data['pretraining']['optimizer'].lower() == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        elif config_data['pretraining']['optimizer'].lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

        Experiment_param['model type'] = config_data['pretraining']['model_type']
        Experiment_param['experiment state'] = 'pretraining'
        model_path = train_and_log(train_dataloader,validation_dataloader,model,device,criterion,optimizer,save_each_epoch,run_param,Experiment_param)
        last_modified_file = get_latest_model_path(parent_directory)

        # Finetuning
        # Load the pre-trained model
        print('Loading pretrained model')
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        print('Finetuning - Reading config data from YAML...')
        best_acc = 0
        best_loss = np.inf

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
        train_and_log(train_dataloader,validation_dataloader,model,device,criterion,optimizer,save_each_epoch,run_param,Experiment_param)


    # Generate config directory and save a copy of config into it
    config_dir = f'''{os.getcwd()}\dataset\{config_data['experiment_params']['dataset_name']}\{Experiment_param['Experiment_id']}\config'''
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    with open(f'''{config_dir}\config.yaml''', 'w') as outfile:
        yaml.dump(config_data, outfile, default_flow_style=False)
            
