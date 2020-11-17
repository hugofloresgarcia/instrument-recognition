import os
import json
import glob

import pandas as pd 
import tqdm.contrib.concurrent

def load_metadata_csv(path_to_metadata):
    assert os.path.exists(path_to_metadata), f"{path_to_metadata} does not exist"
    metadata = pd.read_csv(path_to_metadata).to_dict('records')
    return metadata

def save_metadata_csv(metadata, path_to_metadata): 
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

def remove_dead_entries(metadata):
    new_m = []
    for e in metadata:
        if isinstance(e['path_to_audio'], str):
            new_m.append(e)
    return new_m

def glob_metadata_entries(path_to_dataset):
    """ reads all json files recursively and loads them into
    a list of dicts
    """ 
    pattern = os.path.join(path_to_dataset, '**/*.json')
    filepaths = glob.glob(pattern, recursive=True)

    metadata = tqdm.contrib.concurrent.process_map(load_dict_json, filepaths, max_workers=20, chunksize=20)

    # metadata = [load_dict_json(path) for path in filepath_iterator]

    return metadata

def load_dataset_metadata(path_to_dataset):
    path_to_csv = get_path_to_metadata(path_to_dataset)
    
    if not os.path.exists(path_to_csv):
        metadata = glob_metadata_entries(path_to_dataset)
        save_metadata_csv(metadata, path_to_csv)
    else:
        metadata = load_metadata_csv(path_to_csv)

    metadata = remove_dead_entries(metadata)

    return metadata