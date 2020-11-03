import os
import json
import glob
import sys

from philharmonia_dataset import PhilharmoniaSet, train_test_split
import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import torchaudio
from torch.utils.data import DataLoader

from .utils import audio_utils

#------------
#OPENMIC
#------------

class OpenMicDataModule(pl.LightningDataModule):   
    
    def __init__(self, batch_size=1, num_workers=2):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self):
        # for backwards compatibility w/ some of the older models
        self.train_data = OpenMicDataset(train=True)
        self.dataset = self.train_data # for backwards compatibility

        self.test_data = OpenMicDataset(train=False)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, 
            collate_fn=dont_collate,
            shuffle=False, num_workers=self.num_workers)

    def val_dataloader(self):  
        return DataLoader(self.test_data, batch_size=self.batch_size, 
            collate_fn=dont_collate,
            shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, 
            collate_fn=dont_collate,
             shuffle=False, num_workers=self.num_workers)

class OpenMicDataset(torch.utils.data.Dataset):

    def __init__(self, 
                path_to_data='../data/openmic-2018', 
                train=True):
        print('initing openmic dataset')
        self.path_to_data = path_to_data

        # save our faster version of the csv if our version doesn't exist
        tag = 'train' if train else 'validation'
        path_to_fast_data = os.path.join(path_to_data, f'{tag}_set.csv')

        if os.path.exists(path_to_fast_data):
            print(f'found {path_to_fast_data}')
            self.metadata = pd.read_csv(path_to_fast_data).to_dict('records')
            for idx, e in enumerate(self.metadata):
                labels = e['labels'].strip('()\'\'\""\n').replace(' ', '').replace("'", '').split(',')
                labels = [l for l in labels if not l == '']
                self.metadata[idx]['labels'] = labels
        else:
            print('reading metadata')
            self.metadata = pd.read_csv(
                            os.path.join(path_to_data, 'openmic-2018-metadata.csv')).to_dict('records')

            print('reading partitions')
            split_train = pd.read_csv(
                os.path.join(self.path_to_data, 'partitions/split01_train.csv'), 
                            header=None, squeeze=True)
            split_test = pd.read_csv(
                os.path.join(self.path_to_data, 'partitions/split01_test.csv'), 
                            header=None, squeeze=True)

            print('reading labels')
            self.aggregated_labels = pd.read_csv(
                os.path.join(self.path_to_data, 'openmic-2018-aggregated-labels.csv')
            )

            self.train_split = set(split_train)
            self.test_split = set(split_test)

            print('filtering data')
            if train:
                self.metadata = [e for e in self.metadata if e['sample_key'] in self.train_split]
            else:
                self.metadata = [e for e in self.metadata if e['sample_key'] in self.test_split]

            # add labels to metadata (there's probably a faster way to do this but who cares)
            # I care because this is incredibly slow
            # couldn't figure it out so we're just saving it onow
            print('adding labels to data')
            for idx, e in enumerate(self.metadata):
                labels = self.aggregated_labels.query(f'sample_key == "{e["sample_key"]}"')['instrument']
                # labels = [d['instrument'] for d in self.aggregated_labels if d['sample_key'] == e['sample_key']]
                self.metadata[idx]['labels'] = tuple(labels)

            # save data metadata in a more succint format
            pd.DataFrame(self.metadata).to_csv(path_to_fast_data)     

        print('loading class map')
        self.class_map = json.load(
            open(os.path.join(self.path_to_data, 'class-map.json')))

        self.classes = list(sorted(self.class_map.keys(), key=lambda x: self.class_map[x]))
        print('done initing dataset!')

        self.freq_dict = None

        #['accordion', 'banjo', 'bass', 'cello', 'clarinet', 
        # 'cymbals', 'drums', 'flute', 'guitar', 'mallet_percussion', 
        # 'mandolin', 'organ', 'piano', 'saxophone', 'synthesizer',
        #  'trombone', 'trumpet', 'ukulele', 'violin', 'voice']

    def get_class_frequencies(self):
        """
        returns a list of tuples with shape (class name, number of occurrences in the dataset)
        only computes once. 
        """
        if self.freq_dict is None:
            self.freq_dict = {class_name: 0 for class_name in self.classes}
            for idx, e in enumerate(self.metadata):
                for label in e['labels']:
                    self.freq_dict[label] +=1
        else:
            pass
        return list(sorted(self.freq_dict.items(), key= lambda x: x[0]))

    def retrieve_entry(self, entry):
        sample_key = entry['sample_key']
        parent_dir = sample_key[0:3]
        filename = sample_key + '.ogg'

        path_to_audio = os.path.join(self.path_to_data, 'audio', parent_dir, filename)

        audio, sr = torchaudio.load(path_to_audio)

        data = {
            'path_to_audio': path_to_audio,
            'labels': entry['labels'], 
            'onehot': self.get_onehot(entry['labels']),
            'audio': audio, 
            'sr': sr,
        }
        return data

    def __getitem__(self, index):
        def retrieve(index):
            return self.retrieve_entry(self.metadata[index])

        if isinstance(index, int):
            return retrieve(index)
        elif isinstance(index, slice):
            result = []
            start, stop, step = index.indices(len(self))
            for idx in range(start, stop, step):
                result.append(retrieve(idx))
            return result
        else:
            raise TypeError("index is neither an int or a slice")

    def __len__(self):
        return len(self.metadata)
    
    def get_onehot(self, labels):
        return np.array([1 if l in labels else 0 for l in self.classes])

