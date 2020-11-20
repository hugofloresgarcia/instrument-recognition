import random
import os
import glob
import shutil

import medleydb as mdb
import instrument_recognition.utils as utils
from instrument_recognition.datasets.base_dataset import BaseDataset, BaseDataModule

class MDBDataset(BaseDataset):

    def __init__(self, path_to_data, embeddings=True, train=True):
        super().__init__(path_to_data, embeddings)

        # train_meta, test_meta = utils.data.train_test_split_by_entry_key(
        #     self.metadata, key='track_id', train_size=0.8, test_size=0.2, seed=42)

        self.splits = mdb.utils.artist_conditional_split(test_size=0.15, num_splits=1, 
                                                random_state=random_seed)[0]

def split_mdb_metadata(path_to_data, path_to_output, test_size=0.3, random_seed=20):
    split_track_ids = mdb.utils.artist_conditional_split(test_size=test_size, num_splits=1, 
                                                random_state=random_seed)[0]
    
    train_track_ids = split_track_ids['train']
    test_track_ids = split_track_ids['test']

    base_train_path = os.path.abspath(os.path.join(path_to_output, 'train'))+'/'
    base_test_path = (os.path.join(path_to_output, 'test'))+'/'

    

    # get the corresponding trackid for every path
    print('loading track ids for all files')
    metadata_paths = glob.iglob(
        os.path.join(path_to_data, '**/*.yaml'), recursive=True)
    track_id_dict = {}
    for p in metadata_paths:
        p = os.abspath(p)
        print(f'loading {p}')
        track_id_dict[p] = utils.data.load_dict_yaml(p)['track_id']

    not_in_split = []
    for path, track_id in track_id_dict.items():
        if track_id in train_track_ids:
            new_path = path.replace(path_to_data, base_train_path)
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            print(f'copying {path} to {new_path}')
            shutil.copy(path, new_path)
        elif track_id in test_track_ids:
            new_path = path.replace(path_to_data, base_test_path)
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            print(f'copying {path} to {new_path}')
            shutil.copy(path, new_path)
        else:
            not_in_split.append(track_id)

    print('done! :)')
    print(f'the following files are missing: {not_in_split}')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_data', type=str, required=True)
    parser.add_argument('--path_to_output', type=str, required=True)

    parser.add_argument('--test_size', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=20)

    args = parser.parse_args()

    split_mdb_metadata(args.path_to_data, args.path_to_output, args.test_size, args.seed)