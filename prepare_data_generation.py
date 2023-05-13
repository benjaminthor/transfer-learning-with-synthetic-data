import re
import pandas as pd
import numpy as np
import os
from sktime.datasets import load_from_ucr_tsv_to_dataframe, load_from_tsfile
import yaml
import sys
import subprocess
import chardet

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
        

if __name__ == '__main__':
    ROOT = os.chdir(os.path.dirname(os.path.abspath(__file__)))
    CONFIG = os.path.join(os.getcwd(),"config.yaml")
    config_data = read_yaml_config(CONFIG)
    # Get the full path to the Python interpreter
    python_executable = sys.executable
    script_path = r'C:\Users\nati\Desktop\Implementations\FinalProject\FinalProject\generate_synthetic_data.py'

    data = pd.read_csv(r"C:\Users\nati\Desktop\Implementations\FinalProject\FinalProject\synthetic_data_generation.csv")
    data = data[data['synthetic_num_samples'] == 4]
    data = data[~data['dataset_name'].isin(['ArticularyWordRecognition','Cricket','BasicMotions','AtrialFibrillation','DuckDuckGeese','Epilepsy','EigenWorms','ERing',
                                            'EthanolConcentration','FaceDetection','FingerMovements',])]
    data = data.reset_index(drop=True)

    # For each of the benchmark config go over all possible combinations of experiments config, generate the appropriate yaml file and run the experiment
    for index, row in data.iterrows():
        index = index
        # Get the benchmark config
        epochs = 20
        dataset_name = row['dataset_name']
        train_path = f'''{os.path.dirname(os.getcwd())}\Datasets\{dataset_name}\{dataset_name}_TRAIN.ts'''
        test_path = f'''{os.path.dirname(os.getcwd())}\Datasets\{dataset_name}\{dataset_name}_TEST.ts'''
        X,_ = load_from_tsfile(train_path)
        X_test,_ = load_from_tsfile(test_path)
        series_length,features_num,num_classes = extract_metadata(train_path)
        train_samples = X.shape[0]
        test_samples = X_test.shape[0]

        synthetic_num_samples = int((row['synthetic_num_samples']*train_samples)/num_classes)
        percentage_of_original_data = row['dgan_original_data_ratio']
        #Print all of the above
  
        # Edit yaml file
        config_data['datageneration']['generate_n_sample'] = synthetic_num_samples
        config_data['datageneration']['percentage_of_original_data'] = percentage_of_original_data
        config_data['datageneration']['max_sequence_len'] = series_length
        config_data['datageneration']['sample_length'] = series_length
        config_data['datageneration']['epochs'] = epochs
        # Experiment Params
        config_data['experiment_params']['dataset_name'] = dataset_name
        config_data['experiment_params']['num_classes'] = num_classes
        config_data['experiment_params']['num_features'] = features_num
        config_data['experiment_params']['sequence_length'] = series_length

        # Save the yaml file
        with open(CONFIG, 'w') as file:
            documents = yaml.dump(config_data, file)
        # Run the experiment
        subprocess.call([python_executable, script_path])
        
        
        