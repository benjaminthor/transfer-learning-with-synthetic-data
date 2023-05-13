import re
import pandas as pd
import numpy as np
import os
from sktime.datasets import load_from_ucr_tsv_to_dataframe, load_from_tsfile
import yaml
import sys
import subprocess


def extract_metadata(path):
    # Replace 'file_path' with the path to your file
    file_path = path

    # Read the file
    with open(file_path, 'r') as file:
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
        class_labels = [float(label) for label in class_labels_str.split()]
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
        

if __name__ == 'main':
    CONFIG = os.path.join(os.getcwd(),"config.yaml")
    config_data = read_yaml_config(CONFIG)
    # Get the full path to the Python interpreter
    python_executable = sys.executable
    script_path = r'C:\Users\nati\Desktop\Implementations\FinalProject\FinalProject\generate_synthetic_data.py'

    data = pd.read_csv(r"C:\Users\nati\Downloads\all_experiments_param (2).csv")
    # For each of the benchmark config go over all possible combinations of experiments config, generate the appropriate yaml file and run the experiment
    for index, row in data.iterrows():
        index = index
        # Get the benchmark config
        pretraining_epochs = row['epochs']
        learning_rate = row['learning_rate']
        dataset_name = row['dataset_name']
        model_type = row['model_name']
        train_path = f'''{os.path.dirname(os.getcwd())}\Datasets\{dataset_name}\{dataset_name}_TRAIN.ts'''
        test_path = f'''{os.path.dirname(os.getcwd())}\Datasets\{dataset_name}\{dataset_name}_TEST.ts'''
        X,_ = load_from_tsfile(train_path)
        X_test,_ = load_from_tsfile(test_path)
        series_length,features_num,num_classes = extract_metadata(train_path)
        train_samples = X.shape[0]
        test_samples = X_test.shape[0]
        if model_type == 'LSTM':
            hidden_dim = row['hidden_dim']
            num_layers = row['num_layers']
        else:
            hidden_dim = 'Null'
            num_layers = 'Null'
        synthetic_num_samples = int((row['synthetic_num_samples']*train_samples)/num_classes)
        batchsize = int(np.floor(row['BM_batch_size_ratio']*train_samples))
        percentage_of_original_data = row['dgan_original_data_ratio']

        epochs = row['epochs']
        #Print all of the above
        print(f'''pretraining_epochs: {pretraining_epochs}
        learning_rate: {learning_rate}
        dataset_name: {dataset_name}
        model_type: {model_type}
        train_path: {train_path}
        test_path: {test_path}
        series_length: {series_length}
        features_num: {features_num}
        num_classes: {num_classes}
        train_samples: {train_samples}
        hidden_dim: {hidden_dim}
        num_layers: {num_layers}
        synthetic_num_samples: {synthetic_num_samples}
        batchsize: {batchsize}
        percentage_of_original_data: {percentage_of_original_data}
        epochs: {epochs}''')

        finetuning_original_data_ratio = row['finetuning_original_data_ratio']
        # Edit yaml file
        config_data['datageneration']['generate_n_sample'] = synthetic_num_samples
        config_data['datageneration']['percentage_of_original_data'] = percentage_of_original_data
        config_data['datageneration']['max_sequence_len'] = series_length
        config_data['datageneration']['sample_length'] = series_length
        # Experiment Params
        config_data['experiment_params']['dataset_name'] = dataset_name
        config_data['experiment_params']['num_classes'] = num_classes
        config_data['experiment_params']['num_features'] = features_num
        config_data['experiment_params']['sequence_length'] = series_length
        # Finetuning
        config_data['finetuning']['batch_size'] = batchsize
        config_data['finetuning']['epochs'] = epochs
        config_data['finetuning']['learning_rate'] = learning_rate
        config_data['finetuning']['model_type'] = model_type
        # Pretraining
        config_data['pretraining']['hidden_size'] = hidden_dim
        config_data['pretraining']['num_layers_layers_stacked'] = num_layers
        config_data['pretraining']['batch_size'] = batchsize
        config_data['pretraining']['epochs'] = pretraining_epochs
        config_data['pretraining']['learning_rate'] = learning_rate
        config_data['pretraining']['model_type'] = model_type
        # Save the yaml file
        with open(CONFIG, 'w') as file:
            documents = yaml.dump(config_data, file)
        # Run the experiment
        # subprocess.call([python_executable, script_path])
        
        break
        