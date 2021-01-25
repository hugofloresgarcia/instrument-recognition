import os
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import librosa
import pytorch_lightning as pl
import torchopenl3
import tqdm

import instrument_recognition as ir
from instrument_recognition import utils
import audio_utils as au

remap_class_dict = {'violin section': 'violin', 'viola section': 'viola'}

def debatch(batch):
    for k,v in batch.items():
        if isinstance(v, list):
            batch[k] = v[0]
    return batch

def remap_classes(records, remap_dict):
    for i, entry in enumerate(records):
        if 'events' not in entry:
            raise ValueError(f'entry is missing "events" key: {entry}')
        for event in entry['events']:
            if event['label'] in remap_dict.keys():
                event['label'] = remap_dict[event['label']]
    return records

class Dataset(torch.utils.data.Dataset):

    def __init__(self, name: str, partition: str, embedding_name: str = None, 
                 class_subset: list = None, unwanted_classes: list = None):
        """reads an audio dataset.

        the dataset doesn't care how your folder structure is organized, just finds 
        as many metadata (yaml or json) files as it can, and reads each metadata file as a separate data sample. 
        """
        # define a root path for our dataset
        self.sr = ir.SAMPLE_RATE
        self.root_dir = ir.DATA_DIR / name / partition

        # if an embedding name was provided, check that it exists in the cache
        self.embedding_name = embedding_name
        if embedding_name is not None:
            self.cache_dir = ir.CACHE_DIR / embedding_name / name / partition
            if not self.cache_dir.exists():
                raise FileNotFoundError((f'{self.cache_dir} not found. run preprocess.py with the correct input arguments'))
                
        self.y_cache = {}

        print(f'loading metadata from {self.root_dir}...')
        assert self.root_dir.is_dir(), f'{self.root_dir} does not exist or is a file'
        records = utils.data.glob_all_metadata_entries(self.root_dir, pattern='**/*.json')
        records = remap_classes(records, remap_class_dict)
        self.setup_dataset(records)

        # use regular unwanted classes? 
        if unwanted_classes is None:
            pass

        # filter out unwanted classes
        if unwanted_classes is not None:
            self.filter_unwanted_classes(unwanted_classes)

        # get only a class subset if that is desired
        if 'mdb-synthetic' in name:
            pass
            # class_subset = ['male singer', 'female singer', 'violin', 'clean electric guitar', 
            #                 'distorted electric guitar', 'piano', 'clarinet', 'trumpet']

        if class_subset is not None:
            self.filter_records_by_class_subset(class_subset)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        # get the desired record
        entry = self.records[index]

        # get input
        X = self.get_X(entry)

        # get labels
        y = self.get_y(entry)

        # create output entry
        data = dict(X=X, y=y)

        # create a copy of the metadata entry
        # and add the goodies (X and y)
        item = dict(entry)
        item.update(data)
        item['record_index'] = index

        if 'effect_params' in item:
            del item['effect_params']
        if 'path_to_effect_params' in item:
            del item['path_to_effect_params']

        return item

    def setup_dataset(self, records):
        self.records = records
        if len(self.records) == 0:
            raise ValueError(f'records were empty.')
        print(f'found {len(self.records)} entries')
        self.classlist = utils.data.get_classlist(self.records)
        print(self.get_class_frequencies())
        self.class_weights = self.get_class_weights()

    def filter_records_by_class_subset(self, class_subset):
        subset = utils.data.filter_records_by_class_subset(self.records, class_subset)
        self.setup_dataset(subset)

    def filter_unwanted_classes(self, unwanted_classes):
        subset = utils.data.filter_unwanted_classes(self.records, unwanted_classes)
        self.setup_dataset(subset)    
    
    def get_X(self, entry):
        """ load audio if we don't have an embedding name, else pull embedding from cache
        """
        if self.embedding_name is None:
            audio = au.io.load_audio_file(entry['path_to_audio'], self.sr)

            required_audio_len = int(entry['duration'] * self.sr)

            # truncate any extra samples from scaper
            if audio.shape[-1] > required_audio_len:
                audio = audio[:, 0:required_audio_len]
            
            # zero pad to meet required length if need be
            if audio.shape[-1] < required_audio_len:
                audio = au.zero_pad(audio, required_len=required_audio_len)

            # NOTE: this reshapes the audio into 1 second chunks
            audio = audio.reshape(int(entry['duration']), 1, self.sr)

            return torch.from_numpy(audio)
        else:
            path_to_embedding = self.cache_dir.parent / Path(Path(entry['path_to_audio']).relative_to(self.root_dir.parent)).with_suffix('.npy')
            assert path_to_embedding.exists(), f"{path_to_embedding} does not exist :("
            X = np.load(path_to_embedding)
            X = np.abs(X)
            return torch.from_numpy(X).float()

    def get_y(self, entry):
        resolution=1.0
        key = entry['path_to_audio']
        if key in self.y_cache:
            y = self.y_cache[key]
        else:
            y = utils.data.get_one_hot_matrix(entry, self.classlist, resolution=resolution)
    
        return torch.from_numpy(y) 

    def preprocess_all(self):
        pbar = tqdm.tqdm(self.records)
        for entry in pbar:
            pbar.set_description(f'preprocessing {Path(entry["path_to_audio"]).name}')
            self.get_X(entry)

    def load_numpy(self, entry):
        assert 'path_to_npy' in entry, f'didnt find a path to npy in {entry}'
        # load audio and set to correct format
        path_key = 'path_to_npy'
        X = np.load(entry[path_key])
        X = torch.from_numpy(X).float()
        return X

    def get_class_frequencies(self):
        """
        return a tuple with unique class names and their number of items
        """
        return utils.data.get_class_frequencies(self.records)

    def get_class_weights(self):
        class_weights = np.array([1/count
                                 for label, count in self.get_class_frequencies().items()])
        class_weights = class_weights / max(class_weights)
        return class_weights

    def get_onehot(self, record, resolution=1.0):
        return utils.data.get_one_hot_matrix(record, self.classlist, resolution)

class DataModule(pl.LightningDataModule):

    def __init__(self, dataset_name: str, batch_size: int = 64, num_workers: int = 18, 
                 use_augmented: bool = True, embedding_name: str = None, **dataset_kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_augmented = use_augmented
        self.embedding_name = embedding_name

        self.collate_fn = CollateBatches()

        self.name = dataset_name
        self.dataset_kwargs = dataset_kwargs

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = parent_parser
        parser.add_argument('--dataset_name', type=str, required=True, 
            help='name of dataset. must be a dirname in /data/')
        parser.add_argument('--batch_size', type=int, default=64, 
            help='batch size')
        parser.add_argument('--num_workers', type=int, default=18, 
            help='number of cpus for loading data')
        parser.add_argument('--use_augmented', type=ir.utils.parser_types.str2bool, default=True, 
            help='if available, uses the augmented training set.')
        parser.add_argument('--embedding_name', type=str, 
            help='if specified, loads embeddings from cache instead of raw audio. The provided ' + 
                  'embedding_name should be a directory in /cache/')
        return parser

    def load_dataset(self):
        # cool! now, lets create the dataset objects
        train_partition = 'train-augmented' if Path(ir.DATA_DIR/self.name/'train-augmented').exists() and self.use_augmented else 'train'
        self.train_data = Dataset(name=self.name, partition=train_partition, embedding_name=self.embedding_name, **self.dataset_kwargs)
        self.dataset = self.train_data

        self.test_data = Dataset(name=self.name, partition='test', embedding_name=self.embedding_name, **self.dataset_kwargs)

        if Path(ir.DATA_DIR/self.name/'validation').exists():
            self.val_data = Dataset(name=self.name, partition='validation', embedding_name=self.embedding_name, **self.dataset_kwargs)
        else:
            self.val_data = self.test_data

        print('train entries:', len(self.train_data))
        print('val entries:', len(self.val_data))

    def setup(self):
        self.load_dataset()
    
    @property
    def classlist(self):
        return self.dataset.classlist

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size,
            collate_fn=self.collate_fn, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size,
            collate_fn=self.collate_fn, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size,
            collate_fn=self.collate_fn, shuffle=False, num_workers=self.num_workers)

class CollateBatches:
    """ callable class to collate batches
    """

    def __call__(self, batch):
        """
        collate a batch of dataset samples
        """
        X = [e['X'] for e in batch]
        y = [e['y'] for e in batch]

        X = torch.stack(X)
        y = torch.stack(y)

        # add the rest of all keys
        data = {key: [entry[key] for entry in batch] for key in batch[0]}

        data['X'] = X
        data['y'] = y
        return data
