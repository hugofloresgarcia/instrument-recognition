import random
import os
import glob
import shutil

import medleydb as mdb
import instrument_recognition.utils as utils
from instrument_recognition.datasets import BaseDataset, BaseDataModule

unwanted_classes = ['Main System', 'claps', 'fx/processed sound', 'tuba', 'piccolo', 'cymbal', 'glockenspiel', 'tambourine', 'timpani', 'snare drum', 
                    'clarinet section', 'flute section', 'tenor saxophone', 'trumpet section']

remap_class_dict = {'violin section': 'violin', 'viola section': 'viola'}

def remap_classes(metadata, remap_dict):
    for i, entry in enumerate(metadata):
        if entry['label'] in remap_dict.keys():
            entry['label'] = remap_dict[entry['label']]
    return metadata

def split_mdb_metadata(path_to_data, path_to_output, test_size=0.3, random_seed=20):
    # define split
    splits = mdb.utils.artist_conditional_split(test_size=test_size, num_splits=1, 
                                                random_state=random_seed)[0]
    
    # load metadata
    metadata = utils.data.load_dataset_metadata(path_to_data)

    # split metadata into train and test according to splits
    train_metadata = [e for e in metadata if e['track_id'] in splits['train']]
    test_metadata = [e for e in metadata if e['track_id'] in splits['test']]

    # define the base dirs for our new metadata
    base_train_path = os.path.abspath(os.path.join(path_to_output, 'train'))+'/'
    base_test_path = (os.path.join(path_to_output, 'test'))+'/'

    # get classlist for train and test
    train_classes = utils.data.get_classlist(train_metadata)
    test_classes = utils.data.get_classlist(test_metadata)

    # find the intersection between the two classlists
    filtered_classes = list(set(train_classes) & set(test_classes))
    print(filtered_classes)

    # delete any entry that isnt in the filtered classes
    train_metadata = [e for e in train_metadata if e['label'] in filtered_classes]
    test_metadata = [e for e in test_metadata if e['label'] in filtered_classes]

    # delete any entry that contains unwanted classes
    train_metadata = [e for e in train_metadata if e['label'] not in unwanted_classes]
    test_metadata = [e for e in test_metadata if e['label'] not in unwanted_classes]
    
    # remap sections
    train_metadata = remap_classes(train_metadata, remap_class_dict)
    test_metadata = remap_classes(test_metadata, remap_class_dict)

    # save new metadata to csv in out output path
    os.makedirs(os.path.join(base_train_path), exist_ok=True)
    os.makedirs(os.path.join(base_test_path), exist_ok=True)
    utils.data.save_metadata_csv(train_metadata, os.path.join(base_train_path, 'metadata.csv'))
    utils.data.save_metadata_csv(test_metadata, os.path.join(base_test_path, 'metadata.csv'))

    # checking to see if this actually works
    t = utils.data.load_metadata_csv(os.path.join(base_train_path, 'metadata.csv'))
    for etest, e in zip(t, train_metadata):
        # print('making sure everythings ok')
        try:
            assert etest == e
        except AssertionError:
            print(f'comparison failed: \n\t {etest} \n\t {e}')

    print('done! :)')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_data', type=str, required=True)
    parser.add_argument('--path_to_output', type=str, required=True)

    parser.add_argument('--test_size', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=20)

    args = parser.parse_args()

    split_mdb_metadata(args.path_to_data, args.path_to_output, args.test_size, args.seed)