import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import librosa
import pytorch_lightning as pl

import instrument_recognition.utils as utils

def load_datamodule(path_to_data, batch_size, num_workers, use_embeddings):

    datamodule = BaseDataModule(
        path_to_data=path_to_data,
        batch_size=batch_size, 
        num_workers=num_workers,
        use_embeddings=use_embeddings)
    datamodule.setup()
    
    return datamodule

class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, path_to_data: str, use_embeddings: bool=True, use_augmented: bool = False,  
                class_subset: list=None, unwanted_classes: list=None):
        """reads an audio dataset.

        the dataset doesn't care how your folder structure is organized, just finds 
        as many .yaml files as it can, and reads each yaml file as a separate data sample. 
        each data sample must have the following entries:
            path_to_audio: path to wav file
            sr: sample rate
            label: class label 
            chunk_size: length of audio (in seconds)
            effect_params: dictionary with effects applied (if any)

        Args:
            path_to_data ([type]): path to the dataset
            use_embeddings (bool, optional): [description]. Defaults to True.
            class_subset ([type], optional): [description]. Defaults to None.
        """
        self.path_to_data = path_to_data
        metadata = utils.data.load_dataset_metadata(path_to_data)
        self.setup_dataset(metadata)
        # self._fix_augmented_path_bug()

        self.use_embeddings = use_embeddings
        self.use_augmented = use_augmented

        if unwanted_classes is None:
            from instrument_recognition.scripts.split_mdb import unwanted_classes as uc
            unwanted_classes = uc

        if unwanted_classes is not None:
            self.filter_unwanted_classes(unwanted_classes)

        if class_subset is not None:
            self.filter_metadata_by_class_subset(class_subset)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        entry = self.metadata[index]

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
        item['metadata_index'] = index

        if 'effect_params' in entry:
            entry['effect_params'] = {}

        entry['is_augmented'] = True if self.use_augmented else False

        return item
    
    def _fix_augmented_path_bug(self):
        for entry in self.metadata:
            for key in entry:
                if '-augmented' in key:
                    entry[key] = self.path_to_data + f'/{entry["label"]}/' + entry[key]

    def setup_dataset(self, metadata):
        self.metadata = metadata
        # print(f'found {len(self.metadata)} entries')
        self.classes = utils.data.get_classlist(self.metadata)
        # print(self.get_class_frequencies())
        self.class_weights = self.get_class_weights()

    def filter_metadata_by_class_subset(self, class_subset):
        subset = [e for e in self.metadata if e['label'] in class_subset]
        self.setup_dataset(subset)

    def filter_unwanted_classes(self, unwanted_classes):
        subset = [e for e in self.metadata if e['label'] not in unwanted_classes]
        self.setup_dataset(subset)    
    
    def get_X(self, entry):
        X = self.load_embedding(entry) if self.use_embeddings else self.load_audio(entry)
        return X 

    def get_y(self, entry):
        label = entry['label']
        label_vector = torch.from_numpy(self.get_onehot(label))
        y = torch.argmax(label_vector)
        return y

    def load_audio(self, entry):
        assert 'path_to_audio' in entry, f'didnt find a path to audio in {entry}'
        # load audio and set to correct format
        path_key = 'path_to_audio-augmented' if self.use_augmented else 'path_to_audio'
        X, _ = librosa.load(entry[path_key], sr=entry['sr'], mono=True)
        X = torch.from_numpy(X).float()

        len_X = int(entry['sr'] * entry['chunk_size'])
        X = utils.audio.zero_pad(X, len_X)
        X = X[:len_X]
        X = X.unsqueeze(0)
        return X

    def load_embedding(self, entry):
        assert 'path_to_embedding' in entry, f'didnt find a path to embedding in {entry}'
        # load audio and set to correct format
        path_key = 'path_to_embedding-augmented' if self.use_augmented else 'path_to_embedding'
        X = np.load(entry[path_key])
        X = torch.from_numpy(X).float()
        return X

    def get_class_frequencies(self):
        """
        return a tuple with unique class names and their number of items
        """
        classes = []
        for c in self.classes:
            subset = [e for e in self.metadata if e['label'] == c]
            info = (c, len(subset))
            classes.append(info)

        return tuple(classes)

    def get_class_weights(self):
        class_weights = np.array([1/pair[1]
                                 for pair in self.get_class_frequencies()])
        class_weights = class_weights / max(class_weights)
        return class_weights

    def get_onehot(self, label):
        return np.array([1 if l == label else 0 for l in self.classes])

class CollateBatches:
    """ callable class to collate batches
    """

    def __call__(self, batch):
        """
        collate a batch of dataset samples
        """
        # print(batch)
        X = [e['X'] for e in batch]
        y = [e['y'] for e in batch]

        X = torch.stack(X)
        y = torch.stack(y).view(-1)

        # add the rest of all keys
        data = {key: [entry[key] for entry in batch] for key in batch[0]}

        data['X'] = X
        data['y'] = y
        return data

class BaseDataModule(pl.LightningDataModule):

    def __init__(self, path_to_data, batch_size=1, num_workers=2, 
                **dataset_kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.collate_fn = CollateBatches()

        self.path_to_data = path_to_data
        self.dataset_kwargs = dataset_kwargs

    def load_dataset(self):
        path_to_data = os.path.abspath(self.path_to_data)
        train_path = os.path.join(path_to_data, 'train')
        val_path = os.path.join(path_to_data, 'validation')
        test_path = os.path.join(path_to_data, 'test')

        assert os.path.exists(val_path) or os.path.exists(test_path), \
            f'couldnt find {val_path} or {test_path}. at least one of these needs to exist'

        # if the validation path, use test, and viceversa
        if not os.path.exists(val_path):
            val_path = test_path
        
        if not os.path.exists(test_path):
            test_path = val_path
        
        # cool! now, lets create the dataset objects
        self.train_data = BaseDataset(train_path, **self.dataset_kwargs)
        self.dataset = self.train_data
        self.val_data = BaseDataset(val_path, **self.dataset_kwargs)
        self.test_data = BaseDataset(test_path, **self.dataset_kwargs)

        print('train entries:', len(self.train_data))
        print('val entries:', len(self.val_data))

    def setup(self):
        self.load_dataset()
    
    def get_classes(self):
        return self.dataset.classes

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size,
            collate_fn=self.collate_fn,
             shuffle=False, num_workers=self.num_workers)
