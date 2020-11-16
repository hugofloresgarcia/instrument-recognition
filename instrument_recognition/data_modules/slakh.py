import os
import warnings
import json
import yaml
import random

import numpy as np
import torch
import librosa
import sox
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
import pypianoroll
from tqdm import tqdm

from instrument_recognition.data_modules import data_utils
from instrument_recognition.utils import audio_utils

# rack up the workers
from multiprocessing import Pool, Queue

def _process_slakh_track(args):
    """
    doing this so I can process these in parallel
    """
    tracks, path_to_dataset, instruments, chunk_len, hop_len, sr, transform_audio = args
    slakh_metadata = []
    track_path = os.path.join(path_to_dataset, tracks)
    if not os.path.isdir(track_path):
        return []
    if not os.path.exists(os.path.join(track_path, 'metadata.yaml')):
        return []
    with open(os.path.join(track_path, 'metadata.yaml')) as f:
        metadata = yaml.full_load(f)

    for stem in metadata['stems']:
        path_to_audio = os.path.join(track_path, 'stems', stem + '.flac')
        path_to_midi = os.path.join(track_path, 'MIDI', stem + '.mid')

        # if any of these paths don't exist, append to our broken stem list
        if not os.path.exists(path_to_audio):
            print(f'broken stem: {path_to_audio}')
            continue

        instrument = metadata['stems'][stem]['inst_class']

        if instruments is not None:
            if instrument not in instruments:
                continue

        audio, sr = librosa.load(path_to_audio, mono=True, sr=sr)
        # print(audio.shape)
        tfm = sox.Transformer()
        tfm.silence(min_silence_duration=0.3, buffer_around_silence=False)
        audio = tfm.build_array(input_array=audio, sample_rate_in=sr)
        audio_len = len(audio)

        n_chunks = int(np.ceil(audio_len/(chunk_len*sr)))
        audio = audio_utils.zero_pad(audio, n_chunks * sr)

        # format save path
        # send them bois to CHONK
        audio_npy_path = path_to_audio.replace('slakh2100_flac', 
                                f'slakh_sr-{sr}')
        audio_npy_path = audio_npy_path.replace('.flac', f'.npy')

        # make subdirs if needed
        os.makedirs(os.path.dirname(audio_npy_path), exist_ok=True)
        print(f'saving {audio_npy_path}')

        if not os.path.exists(audio_npy_path):
            if transform_audio:
                audio = torch.from_numpy(audio)
                audio = random_transform(audio.unsqueeze(0).unsqueeze(0), 
                    sr, ['overdrive', 'reverb', 'pitch', 'stretch']).squeeze(0).squeeze(0)
                audio = audio.numpy()

            np.save(audio_npy_path, audio)
        else:
            print(f'already found: {audio_npy_path}')

        for start_time in np.arange(0, n_chunks, hop_len):
            start_time = np.around(start_time, 4)
            # # zero pad 
            # audio_chunk = get_audio_chunk(audio, sr, start_time, chunk_len)
            # audio_chunk_path = path_to_audio.replace('slakh2100_flac', 
            #                     f'slakh_chunklen-{chunk_len}_sr-{sr}_hop-{hop_len}')
            # audio_chunk_path = audio_chunk_path.replace('.wav', f'/{start_time}.npy')
            # audio_chunk_path = audio_chunk_path.replace('.mp3', f'/{start_time}.npy')
            # audio_chunk_path = audio_chunk_path.replace('.flac', f'/{start_time}.npy')

            # os.makedirs(os.path.dirname(audio_chunk_path), exist_ok=True)
            # print(f'saving {audio_chunk_path}')

            # if not os.path.exists(audio_chunk_path):
            #     if transform_audio:
            #         audio_chunk = torch.from_numpy(audio_chunk)
            #         audio_chunk = random_transform(audio_chunk.unsqueeze(0).unsqueeze(0), 
            #             sr, ['overdrive', 'reverb', 'pitch', 'stretch']).squeeze(0).squeeze(0)
            #         audio_chunk = audio_chunk.numpy()

            #     np.save(audio_chunk_path, audio_chunk)
            # else:
            #     print(f'already found: {audio_chunk_path}')


            entry = dict(
                path_to_audio=audio_npy_path, 
                instrument=instrument, 
                duration=chunk_len, 
                start_time=start_time, 
                sr=sr)

            slakh_metadata.append(entry)

    return slakh_metadata

def get_audio_chunk(audio, sr, start_time, chunk_len):
    """ given MONO audio 1d array, get a chunk of that 
    array determined by start_time (in seconds), chunk_len (in seconds)
    pad with zeros if necessary
    """
    assert audio.ndim == 1, 'audio needs to be 1 dimensional'
    duration = audio.shape[-1] / sr

    start_idx = int(start_time * sr)
    end_time = start_time + chunk_len 
    end_time = min([end_time, duration])
    end_idx = int(end_time * sr)
    chunked_audio = audio[start_idx:end_idx]

    if not len(audio) / sr == chunk_len * sr:
        chunked_audio = audio_utils.zero_pad(chunked_audio, sr * chunk_len)
    return chunked_audio

def generate_slakh_npz(path_to_dataset, instruments=None, chunk_len=1, hop_len=1, sr=48000, transform_audio=False):
    args = []
    for tracks in os.listdir(path_to_dataset):
        args.append((tracks, path_to_dataset, instruments, chunk_len, hop_len, sr, transform_audio))

    pool = Pool()
    metadata = pool.map(_process_slakh_track, args)
    metadata = [entry for submetadata in metadata for entry in submetadata]

    pool.close()
    pool.join()

    return metadata

class SlakhDataset(Dataset):

    def __init__(self, 
                path_to_dataset='/home/hugo/CHONK/data/slakh2100_flac',
                subset='train', classes=None, chunk_len=1, hop_len=1, 
                sr=48000, transform=None): # audio length in seconds
        subsets = ('train', 'validation', 'test')
        assert subset in subsets, f'subset provided not in {subsets}'
        
        if classes is not None:
            raise NotImplementedError('class subsets not implemented yet:(')

        self.path_to_data = os.path.abspath(os.path.join(path_to_dataset, subset))
        assert os.path.exists(
            self.path_to_data), f'cant find path {self.path_to_data}'

        self.chunk_len = chunk_len
        self.hop_len = hop_len
        self.sr = sr
        self.transform = transform

        # transform_audio = True if subset == 'train' else False
        self.has_npy = False

        path_to_npz_metadata = os.path.join(self.path_to_data, 
            f'slakh-metadata-chunk_len-{self.chunk_len}-sr-{self.sr}-hop-{self.hop_len}-npy.csv')
        if os.path.exists(path_to_npz_metadata):
            self.has_npy = True
            self.metadata = pd.read_csv(path_to_npz_metadata).to_dict('records')
        else:  
            self.metadata = generate_slakh_npz(self.path_to_data,  instruments=classes,
                                            chunk_len=self.chunk_len, hop_len=self.hop_len, sr=self.sr, 
                                            transform_audio=False)
            pd.DataFrame(self.metadata).to_csv(path_to_npz_metadata, index=False)

        # get sorted list of classes
        self.remap_classes_to_mdb()
        self.classes = self.get_classlist(self.metadata)
        
        self.class_weights = np.array([1/pair[1] for pair in self.get_class_frequencies()])
        self.class_weights = self.class_weights / max(self.class_weights)
        print(self.get_class_frequencies())
        [print(f'{c}-{w}') for c, w in zip(self.classes, self.class_weights)]

    def get_str_class_repr(self):
        str_class_repr = ''
        for c in self.classes: 
            str_class_repr += f'{c},\n'
        return str_class_repr  
    
    def check_metadata(self):
        missing_files = []
        for entry in self.metadata:
            if not os.path.exists(entry['path_to_audio']):
                logging.warn(f'{entry["path_to_audio"]} is missing.')
                missing_files.append(entry['path_to_audio'])

        assert len(missing_files) == 0, 'some files were missing in the dataset.\
             delete metadata file and download again, or delete missing entries from metadata'

    def get_class_frequencies(self):
        """
        return a tuple with unique class names and their number of items
        """
        classes = []
        for c in self.classes:
            subset = [e for e in self.metadata if e['instrument'] == c]
            info = (c, len(subset))
            classes.append(info)

        return tuple(classes)

    def __getitem__(self, index):
        entry = self.metadata[index]
        # load audio using numpy
        audio = np.load(entry['path_to_audio'],mmap_mode='r', allow_pickle=False)
        # start_sample = entry['start_time'] * entry['sr']
        # audio_len = entry['duration'] * entry['sr']
        # audiomm = np.memmap(entry['path_to_audio'], np.float32, 'c', offset=start_sample, shape=(audio_len))
        # print(audiomm.shape)
        # print(id(audiomm))
        start_sample = entry['start_time'] * entry['sr']
        end_sample = start_sample + entry['duration'] * entry['sr']
        audio = audio[start_sample:end_sample]
        audio = torch.from_numpy(audio).float()
        # audio = torch.zeros(48000)
        # del audiomm
        # print(audio.shape)
        # print(id(audio))
        # exit()
        audio = audio.unsqueeze(0)

        
        # get labels
        instrument = entry['instrument']
        labels = torch.from_numpy(self.get_onehot(instrument))
        label = torch.argmax(labels)

        # create output entry
        data = dict(
            X=audio, 
            y=label, 
            path_to_audio=entry['path_to_audio'])

        item = dict(entry) # copy entry so we don't update the metadata
        item.update(data) # add all that other shiz just incase

        return item

    def __len__(self):
        return len(self.metadata)
    
    def get_onehot(self, labels):
        return np.array([1 if l in labels else 0 for l in self.classes])

    def remap_classes(self, metadata, class_dict):
        """ remap instruments according to a dictionary provided
        that is, if 
            entry['instrument'] == 'electric guitar'
            
            and

            class_dict['electric guitar'] = 'guitar'

            then entry['instrument'] will be changed to 'guitar'
        """
        classes = self.get_classlist(metadata)
        for idx, c in enumerate(classes):
            if c in class_dict.keys():
                classes[idx] = class_dict[c]

        classes = list(set(classes))
        classes.sort()

        for entry in metadata:
            entry['instrument'] = entry['instrument'].strip('[]\'')
            if entry['instrument'] in class_dict.keys():
                entry['instrument'] = class_dict[entry['instrument']]

    def get_classlist(self, metadata):
        for e in metadata:
            e['instrument'] = str(e['instrument'].strip('[]\''))
        classes = list(set([e['instrument'] for e in metadata]))
        classes.sort()

        return classes

    def remap_classes_to_mdb(self):
        mapping = {
            'Strings (continued)': 'Strings', 
            'Synth Lead': 'Synth', 
            'Synth Pad': 'Synth', 
        }
        self.remap_classes(self.metadata, mapping)

class SlakhDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=1, num_workers=2, 
                sr=48000):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.sr = sr
        self.collate_fn = CollateAudio(self.sr)
        
    def load_dataset(self):
        path = os.path.expanduser('~/CHONK/data/slakh2100_flac')
        self.train_data = SlakhDataset(
            path_to_dataset=path, 
            classes=None, 
            subset='train',
            
            chunk_len=1)
        self.dataset=self.train_data
        self.val_data = SlakhDataset(
            path_to_dataset=path, 
            classes=None, 
            subset='validation',
        
            chunk_len=1)
        self.test_data = SlakhDataset(
            path_to_dataset=path, 
            classes=None, 
            subset='test',
            chunk_len=1)

        # return dataset

    def setup(self):
        self.load_dataset() 

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

class CollateAudio:
    """ callable class to collate and resample audio batches
    """

    def __init__(self, sample_rate):
        self.sr = sample_rate

    def __call__(self, batch):
        """
        collate a batch of dataset samples and resample to a 
        uniform sample rate for proper batch_processing
        """
        audio = []
        labels = []
        for e in batch:
            a = e['X']
            l = e['y']

            # resample audio to 48k (bc openl3)
            a = audio_utils.resample(a, int(e['sr']), 48000)
            a = audio_utils.zero_pad(a.squeeze(0), 48000).unsqueeze(0)
            l = torch.tensor([l])
            audio.append(a)
            labels.append(l)

        audio = torch.stack(audio)
        labels = torch.stack(labels).view(-1)

        # add the rest of all keys
        data = {key: [entry[key] for entry in batch] for key in batch[0]}

        data['X'] = audio
        data['y'] = labels
        return data

if __name__ == "__main__":
    dataset = SlakhDataset(subset='train')    
    dataset = SlakhDataset(subset='validation')
    dataset = SlakhDataset(subset='test')
