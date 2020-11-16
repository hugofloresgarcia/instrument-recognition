import os
import json
import glob

import pandas as pd 

def load_metadata(path_to_metadata):
    assert os.path.exists(path_to_metadata), f"{path_to_metadata} does not exist"
    metadata = pd.read_csv(path_to_metadata).to_dict('records')
    return metadata

def save_metadata(metadata, path_to_metadata): 
    pd.DataFrame(metadata).to_csv(path_to_metadata, index=False)
    return

def get_path_to_metadata(path_to_dataset):
    return os.path.join(path_to_dataset, 'metadata.csv')

def save_dict_json(d, save_path):
    """ save a dictionary using json
    """
    with open(save_path, 'w') as f:
        json.dump(d, f)
    return 

def load_dict_json(path_to_json):
    with open(path_to_json, 'r') as f:
        d = json.load(f)
    return d

def glob_metadata_entries(path_to_dataset):
    """ reads all json files recursively and loads them into
    a list of dicts
    """ 
    pattern = os.path.join(path_to_dataset, '**/*.json')
    filepaths = glob.glob(pattern, recursive=True)

    metadata = [load_dict_json(path) for path in filepaths]
    
    return metadata
