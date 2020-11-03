import glob
import sys
import os
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchaudio
import librosa
import pytorch_lightning as pl
import pandas as pd
import soundfile as sf
import sox

from philharmonia_dataset import  train_test_split, download_dataset
from instrument_recognition.data_modules import data_utils
from instrument_recognition.data_modules.data_utils import collate_audio
from instrument_recognition.utils import audio_utils

#------------
# Philharmonia
#------------


def get_audio_chunk(audio, sr, start_time, chunk_len):
    """ 
    """
    duration = audio.shape[-1] / sr

    start_idx = start_time * sr
    end_time = start_time + chunk_len 
    end_time = min([end_time, duration])
    end_idx = int(end_time * sr)
    chunked_audio = audio[start_idx:end_idx]

    if not len(audio) / sr == chunk_len * sr:
        chunked_audio = audio_utils.zero_pad(chunked_audio, sr * chunk_len)
    return chunked_audio


def generate_npz_dataset(metadata, sr=48000, chunk_len=1):
    
    new_metadata = []
    for entry in metadata: 
        path_to_audio = os.path.abspath('../'+entry['path_to_audio'])
        audio, sr = librosa.load(path_to_audio, mono=True, sr=sr)
        # print(audio.shape)
        tfm = sox.Transformer()
        tfm.silence(min_silence_duration=0.2, buffer_around_silence=True)
        # transform to sox
        # audio = audio.cpu().detach().numpy()
        audio = tfm.build_array(input_array=audio, sample_rate_in=sr)

# # split clips on silence
#         split_audio, intervals = audio_utils.split_on_silence(audio, top_db=45)
        split_audio = [audio]

        for clip in split_audio:
            n_chunks = int(np.ceil(len(clip)/(chunk_len*sr)))
            # print(split_audio)
            # exit()
            for start_time in range(n_chunks):
                # zero pad 
                audio_chunk = get_audio_chunk(clip, sr, start_time, chunk_len)
                audio_chunk_path = path_to_audio.replace('philharmonia', 
                                    f'philharmonia_chunklen-{chunk_len}_sr-{sr}')
                audio_chunk_path = audio_chunk_path.replace('.wav', f'_{start_time}')
                audio_chunk_path = audio_chunk_path.replace('.mp3', f'_{start_time}')

                os.makedirs(os.path.dirname(audio_chunk_path), exist_ok=True)
        
                print(f'saving {audio_chunk_path}')
                np.save(audio_chunk_path, audio_chunk)

                instrument =  entry['instrument']
                duration = chunk_len
            
                entry = dict(
                    path_to_audio=audio_chunk_path, 
                    instrument=instrument, 
                    # activations=activations,
                    duration=chunk_len, 
                    start_time=start_time)
                
                new_metadata.append(entry)
    return new_metadata


class CollateAudio:
    """ callable class to collate and resample audio batches
    """

    def __call__(self, batch):
        """
        collate a batch of dataset samples and resample to a 
        uniform sample rate for proper batch_processing
        """
        # print(batch)
        audio = [e['X'] for e in batch]
        labels = [e['y'] for e in batch]

        audio = torch.stack(audio)
        labels = torch.stack(labels).view(-1)

        # add the rest of all keys
        data = {key: [entry[key] for entry in batch] for key in batch[0]}

        data['X'] = audio
        data['y'] = labels
        return data


# TODO: I NEED TO REFACTOR ALL MY DATA MODULES SO THEY ACTUALLY WORK
class PhilharmoniaSet(Dataset):
    def __init__(self, 
                dataset_path: str = './data/philharmonia', 
                classes: tuple = None,
                download = True, 
                load_audio: bool = True):
        """
        create a PhilharmoniaSet object.
        params:
            path_to_csv (str): path to metadata.csv created upon downloading the dataset
            classes (tuple[str]): tuple with classnames to include in the dataset
            load_audio (bool): whether to load audio or pass the path to audio instead when retrieving an item
        """
        super().__init__()
        self.load_audio = load_audio
        self.sr = 48000
        self.chunk_len = 1

        # download if requested
        if download:
            download_dataset(dataset_path)

        path_to_csv = os.path.join(dataset_path ,'all-samples', 'metadata.csv')
        self.dataset_path = dataset_path

        assert os.path.exists(path_to_csv), f"couldn't find metadata:{path_to_csv}"
        # generate a list of dicts from our dataframe
        self.metadata = pd.read_csv(path_to_csv).to_dict('records')

        path_to_npy_metadata = os.path.join(dataset_path+f'_chunklen-{self.chunk_len}_sr-{self.sr}' ,'all-samples', 'metadata.csv')
        if not os.path.exists(path_to_npy_metadata):
            self.metadata = generate_npz_dataset(self.metadata, self.sr, self.chunk_len)
            pd.DataFrame(self.metadata).to_csv(path_to_npy_metadata)
        else:
            self.metadata = pd.read_csv(path_to_npy_metadata).to_dict('records')

        # remove all the classes not specified, unless it was left as None
        if classes == 'no_percussion':
            self.classes = list("saxophone,flute,guitar,contrabassoon,bass-clarinet,trombone,cello,oboe,bassoon,banjo,mandolin,tuba,viola,french-horn,english-horn,violin,double-bass,trumpet,clarinet".split(','))
            self.metadata = [e for e in self.metadata if e['instrument'] in self.classes]
        elif classes is not None: # if it's literally anything else lol (filter out unneeded metadata)
            self.metadata = [e for e in self.metadata if e['instrument'] in classes]
            self.classes = list(set([e['instrument'] for e in self.metadata]))
        else:
            self.classes = list(set([e['instrument'] for e in self.metadata]))

        self.classes.sort()
        self.class_weights = np.array([1/pair[1] for pair in self.get_class_frequencies()])
        self.class_weights = self.class_weights / max(self.class_weights)
        print(self.get_class_frequencies())
        [print(f'{c}-{w}') for c, w in zip(self.classes, self.class_weights)]


        self.load_chunked = False
        self.chunk_len = 1
        self.transform = None

        self.audio_cache = {}
        
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

    def _retrieve_entry(self, entry):
        path_to_audio = '../' + entry['path_to_audio']
        # path_to_audio = path_to_audio.replace('./data/philharmonia', self.dataset_path)
        
        filename = path_to_audio.split('/')[-1]

        assert os.path.exists(path_to_audio), f"couldn't find {path_to_audio}"

        instrument = entry['instrument']
        pitch = entry['pitch']

        data = {
            'filename': filename,
            'one_hot': self.get_onehot(instrument),
            'y': torch.tensor(np.argmax(self.get_onehot(instrument))), 
        }

        # add all the keys from the entryas well
        data.update(entry)

        if self.load_audio:
            if path_to_audio not in self.audio_cache:
                # import our audio using torchaudio
                tic = time.time()
                audio, sr = torchaudio.load(path_to_audio)
                audio = audio_utils.resample(audio, sr, self.sr)
                audio = audio.numpy()
                audio = audio_utils.downmix(audio, keepdim=False)
                audio = audio_utils.zero_pad(audio, self.chunk_len * self.sr)
                audio = torch.from_numpy(audio).unsqueeze(0)
                toc = time.time() - tic
                if toc > 7.5:
                    print(f"LOADING {path_to_audio} took {toc}")
                self.audio_cache[path_to_audio] = audio
            
            else:
                audio = self.audio_cache[path_to_audio]
                sr = self.sr

            if self.load_chunked:
                # calculate end time
                start_time = data['start_time'] * self.sr
                end_time = (data['start_time'] + data['audio_len']) * self.sr
                audio = audio[:, start_time:end_time]
                audio = audio_utils.zero_pad(audio, self.chunk_len * self.sr)
            data['X'] = audio
            data['sr'] = sr
                
        return data

    def __getitem__(self, idx):
        entry = self.metadata[idx]

        # if entry['path_to_audio'] not in self.audio_cache:
            
        #     # load audio, trim, and zero pad if necessary
        #     tic = time.time()
        #     #NOTE: sf.read has an option to load only a segment of the audio and optionally pads
        #     # i could use it for loading chunks instead of loading the whole thing at a time/
        #     audio, sr = sf.read(entry['path_to_audio'], dtype='float32')
        #     audio = audio_utils.resample(audio, sr, self.sr)
        #     audio = audio_utils.downmix(audio, keepdim=False)
        #     audio = audio_utils.zero_pad(audio, self.chunk_len * self.sr)
        #     toc = time.time() - tic
        #     # print(f'sf took {toc}')

        #     # tic = time.time()
        #     # audio = librosa.load(entry['path_to_audio'], 
        #     #                     sr=self.sr, 
        #     #                     mono=True)
        #     # toc = time.time() - tic
        #     # print(f'librosa took {toc}'
        #     self.audio_cache[entry['path_to_audio']] = audio
        # else:
        #     audio = self.audio_cache[entry['path_to_audio']]

        # load audio using numpy
        audio = np.load(entry['path_to_audio']+'.npy', allow_pickle=True)

        # print(audio)
        audio = torch.from_numpy(audio).float()


        # add channel dimensions
        audio = audio.unsqueeze(0)
    
        # apply transform
        audio = self.transform(audio) if self.transform is not None else audio
        
        # get labels
        instrument = entry['instrument']
        labels = torch.from_numpy(self.get_onehot(instrument))
        label = torch.argmax(labels)

        # create output entry
        data = dict(
            X=audio, 
            y=label, 
            path_to_audio=entry['path_to_audio'])
        return data

    # def __getitem__(self, index):
    #     def retrieve(index):
    #         return self._retrieve_entry(self.metadata[index])

    #     if isinstance(index, int):
    #         return retrieve(index)
    #     elif isinstance(index, slice):
    #         result = []
    #         start, stop, step = index.indices(len(self))
    #         for idx in range(start, stop, step):
    #             result.append(retrieve(idx))
    #         return result
    #     else:
    #         raise TypeError("index is neither an int or a slice")

    def __len__(self):
        return len(self.metadata)

    def get_onehot(self, label):
        assert label in self.classes, "couldn't find label in class list"
        return np.array([1 if label == l else 0 for l in self.classes])

    def get_example(self, class_name):
        """
        get a random example belonging to class_name from the dataset
        for demo purposes
        """
        subset = [e for e in self.metadata if e['instrument'] == class_name]
        # get a random index
        idx = torch.randint(0, high=len(subset), size=(1,)).item()

        entry = subset[idx]
        return self.retrieve_entry(entry)


class PhilharmoniaDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=1, num_workers=2, load_audio=True):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.load_audio = load_audio
    
    def load_dataset(self):
        dataset = PhilharmoniaSet(
            dataset_path='/home/hugo/data/philharmonia', 
            classes='no_percussion', 
            download=True,
            load_audio=self.load_audio
        )

        # if not os.path.exists('/home/hugo/lab/mono_music_sed/instrument_recognition/data_modules/cache/philharmonia-1.csv'):
        #     dataset.metadata = data_utils.expand_metadata_per_timestep(dataset.metadata, 1)
        #     # cache new dataset metadata
        #     data_utils.cache_metadata(dataset.metadata, name='philharmonia-1.csv')
        # else:
        #     dataset.metadata = pd.read_csv('/home/hugo/lab/mono_music_sed/instrument_recognition/data_modules/cache/philharmonia-1.csv').to_dict('records')
        
        # dataset.load_chunked = True
        return dataset

    def setup(self):
        
        self.dataset = self.load_dataset()
        self.splits = np.array([0.7, 0.3])

        lengths = (len(self.dataset)*self.splits).astype(int)
        # oof. hackiest thing ever. BUT I can't think of anything else
        # 
        while sum(lengths) < len(self.dataset):
            lengths[-1] +=1 

        # self.train_data, self.val_data, self.test_data = torch.utils.data.random_split(
        #                 dataset=self.dataset, 
        #                 lengths=lengths)
        self.train_data, self.val_data = torch.utils.data.random_split(
                        dataset=self.dataset, 
                        lengths=lengths)
        # self.train_loader, self.val_loader = train_test_split(self.dataset,
        #                                                     batch_size=self.batch_size,
        #                                                     num_workers=self.num_workers,
        #                                                     val_split=0.3,
        #                                                     collate_fn=collate_audio, 
        #                                                     random_seed=42)


    def train_dataloader(self):
        print('fetching train dataloader..')
        return DataLoader(self.train_data, batch_size=self.batch_size, 
            collate_fn=CollateAudio(), 
            shuffle=True, num_workers=self.num_workers)
        # return self.train_loader

    def val_dataloader(self):  
        return DataLoader(self.val_data, batch_size=self.batch_size, 
            collate_fn=CollateAudio(), 
            shuffle=False, num_workers=self.num_workers)
        # return self.val_loader

    def test_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, 
            collate_fn=CollateAudio(),
             shuffle=False, num_workers=self.num_workers)


if __name__ == "__main__":
    dataset = PhilharmoniaSet(
            dataset_path='/home/hugo/data/philharmonia', 
            classes='no_percussion', 
            download=True,
            load_audio=False
        )
    l = ''
    for c in dataset.classes:
        # l += f'"{c}",'
        l += f'{c}\n'

    print(l)
