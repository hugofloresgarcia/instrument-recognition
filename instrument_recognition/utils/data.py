import os
import json
import glob
import yaml
from pathlib import Path

import pandas as pd 
import tqdm.contrib.concurrent
from sklearn.model_selection import train_test_split

def _add_file_format_to_filename(path: str, file_format: str):
    if '.' not in file_format:
        file_format = f'.{file_format}'

    if Path(path).suffix != file_format:
        path = str(Path(path).with_suffix(file_format))
    return path

# TODO: implement me
def jams_to_matrix(jams_data, num_time_steps):
    """returns a one hot matrix  shape (sequence, num_classes)
    """
    raise NotImplementedError

def get_abspath(path):
    return os.path.abspath(os.path.expanduser(path))

def get_classlist(metadata):
    """ iterate through metadata and get the set
        of all labels
    """
    for e in metadata:
        e['label'] = str(e['label'].strip('[]\''))
    classes = list(set([e['label'] for e in metadata]))
    classes.sort()
    return classes

def load_metadata_csv(path_to_metadata):
    assert os.path.exists(path_to_metadata), f"{path_to_metadata} does not exist"
    metadata = pd.read_csv(path_to_metadata).to_dict('records')
    return metadata

def save_metadata_csv(metadata, path_to_metadata): 
    pd.DataFrame(metadata).to_csv(path_to_metadata, index=False)
    return

def get_path_to_metadata(path_to_dataset):
    return os.path.join(path_to_dataset, 'metadata.csv')

def save_json(d, save_path):
    """ save a dictionary using json
    """
    os.makedirs(Path(save_path).parent, exist_ok=True)
    save_path = _add_file_format_to_filename(save_path, 'json')
    with open(save_path, 'w') as f:
        json.dump(d, f)

def save_yaml(d, save_path):
    os.makedirs(Path(save_path).parent, exist_ok=True)
    save_path = _add_file_format_to_filename(save_path, 'yaml')
    with open(save_path, 'w') as f:
        yaml.dump(d, f)

def load_json(path_to_json):
    with open(path_to_json, 'r') as f:
        d = json.load(f)
    return d

def load_yaml(path_to_yaml):
    with open(path_to_yaml, 'r') as f:
        d = yaml.load(f, allow_pickle=True)
    return d

def glob_metadata_entries(path_to_dataset, pattern='**/*.yaml'):
    """ reads all yaml files recursively and loads them into
    a list of dicts
    """ 
    pattern = os.path.join(path_to_dataset, pattern)
    filepaths = glob.glob(pattern, recursive=True)

    metadata = tqdm.contrib.concurrent.process_map(load_yaml, filepaths, max_workers=20, chunksize=20)

    # metadata = [load_json(path) for path in filepath_iterator]

    return metadata

def load_dataset_metadata(path_to_dataset):
    path_to_csv = get_path_to_metadata(path_to_dataset)
    
    if not os.path.exists(path_to_csv):
        metadata = glob_metadata_entries(path_to_dataset)
        metadata = remove_dead_entries(metadata)
        save_metadata_csv(metadata, path_to_csv)
    else:
        metadata = load_metadata_csv(path_to_csv)

    return metadata

def train_test_split_by_entry_key(metadata, key='track_id', 
                                 train_size=0.8, test_size=0.2, seed=42):
    all_keys = list(set([e[key] for e in metadata]))

    train_keys, test_keys = train_test_split(all_keys, test_size=test_size, 
                              train_size=train_size, random_state=seed)

    train_metadata = [e for e in metadata if e[key] in train_keys]
    test_metadata = [e for e in metadata if e[key] in test_keys]

    print(train_keys)
    print(test_keys)
    return train_metadata, test_metadata

