import os
import warnings
import json
import yaml

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

def read_piano_roll(path_to_midi):
    mtrack = pypianoroll.Multitrack(path_to_midi)
    # if there's more than 1 track in the multitrack, 
    # raise a warning (its a stem so it should only have)
    # one track
    if len(mtrack.tracks) > 1:
        print(f'midi {path_to_midi} has more than one track')
    try:
        midi_track = mtrack.tracks[0]
        piano_roll = midi_track.pianoroll
    except:
        print(f'exception occured while loading {path_to_midi}')
        print(str(mtrack))
        piano_roll = mtrack.get_merged_pianoroll()
        
    return piano_roll

def is_silent(piano_roll):
    """
    figure out if piano roll if full of zeros
    """
    if np.all(piano_roll==0):
        return True
    else:
        return False

def is_silent_entry(data):
    path_to_audio = data['path_to_audio']
    audio, sr = torchaudio.load(path_to_audio)
    track_length_seconds = audio.shape[-1] / sr

    instrument = data['instrument']

    piano_roll = read_piano_roll(data['path_to_midi'])

    start_ratio =  data['start_time'] / track_length_seconds
    end_ratio = (data['start_time'] + data['audio_len']) / track_length_seconds

    start_step = int(piano_roll.shape[0] * start_ratio)
    end_step  = int(piano_roll.shape[0] * end_ratio)

    piano_roll = piano_roll[start_step:end_step, :]
    
    if is_silent(piano_roll):
        return True
    else:
        return False

def get_slakh_metadata(path_to_dataset, instruments=None, load_piano_roll=True):
    slakh_metadata = []
    broken_stems = []
    for tracks in os.listdir(path_to_dataset):
        track_path = os.path.join(path_to_dataset, tracks)
        if not os.path.isdir(track_path):
            continue
        with open(os.path.join(track_path, 'metadata.yaml')) as f:
            metadata = yaml.full_load(f)

        for stem in metadata['stems']:
            path_to_audio = os.path.join(track_path, 'stems', stem + '.flac')
            path_to_midi = os.path.join(track_path, 'MIDI', stem + '.mid')

            # if any of these paths don't exist, append to our broken stem list
            cont=False
            if not os.path.exists(path_to_audio):
                broken_stems.append(path_to_audio)
                cont = True
            if not os.path.exists(path_to_midi):
                broken_stems.append(path_to_midi)
                cont = True

            if cont: continue

            instrument = metadata['stems'][stem]['inst_class']

            if instruments is not None:
                if instrument not in instruments:
                    continue

            slakh_metadata.append(dict(path_to_audio=path_to_audio, 
                                path_to_midi=path_to_midi, 
                                instrument=instrument))
    # report any broken stems
    if not broken_stems == []:
        print(f'the following files were not found: {broken_stems}')
        print(f'total files: {len(broken_stems)}')

    return slakh_metadata

def generate_slakh_npz(path_to_dataset, instruments=None, chunk_len=1, sr=48000):
    slakh_metadata = []
    broken_stems = []
    for tracks in os.listdir(path_to_dataset):
        track_path = os.path.join(path_to_dataset, tracks)
        if not os.path.isdir(track_path):
            continue
        with open(os.path.join(track_path, 'metadata.yaml')) as f:
            metadata = yaml.full_load(f)

        for stem in metadata['stems']:
            path_to_audio = os.path.join(track_path, 'stems', stem + '.flac')
            path_to_midi = os.path.join(track_path, 'MIDI', stem + '.mid')

            # if any of these paths don't exist, append to our broken stem list
            cont=False
            if not os.path.exists(path_to_audio):
                broken_stems.append(path_to_audio)
                cont = True
            if not os.path.exists(path_to_midi):
                broken_stems.append(path_to_midi)
                cont = True

            if cont: continue

            instrument = metadata['stems'][stem]['inst_class']

            if instruments is not None:
                if instrument not in instruments:
                    continue

            audio, sr = librosa.load(path_to_audio, mono=True, sr=sr)
            # print(audio.shape)
            tfm = sox.Transformer()
            tfm.silence(min_silence_duration=0.3, buffer_around_silence=False)
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
                    audio_chunk_path = path_to_audio.replace('slakh-2100-flac', 
                                        f'slakh_chunklen-{chunk_len}_sr-{sr}')
                    audio_chunk_path = audio_chunk_path.replace('.wav', f'_{start_time}')
                    audio_chunk_path = audio_chunk_path.replace('.mp3', f'_{start_time}')
                    audio_chunk_path = audio_chunk_path.replace('.flac', f'_{start_time}')

                    os.makedirs(os.path.dirname(audio_chunk_path), exist_ok=True)
            
                    print(f'saving {audio_chunk_path}')
                    np.save(audio_chunk_path, audio_chunk)

                    instrument = instrument
                    duration = chunk_len
                
                    entry = dict(
                        path_to_audio=audio_chunk_path, 
                        instrument=instrument, 
                        # activations=activations,
                        duration=chunk_len, 
                        start_time=start_time)
    # report any broken stems
    if not broken_stems == []:
        print(f'the following files were not found: {broken_stems}')
        print(f'total files: {len(broken_stems)}')

    return slakh_metadata

def find_silence(metadata, keep=False):
    """
    remove silent regions in metadata (from midi)
    if keep, adds a 'silence' label instead. 
    """
    print('inspecting metadata for silence labels')

    new_metadata = []
    pbar = tqdm(enumerate(metadata), total=len(metadata))
    silence_count = 0
    track_lengths = {}
    for idx, data in pbar:
        path_to_audio = data['path_to_audio']
        if path_to_audio not in track_lengths:
            audio, sr = torchaudio.load(path_to_audio)
            track_length_seconds = audio.shape[-1] / sr
            track_lengths[path_to_audio] = track_length_seconds
        else:
            track_length_seconds = track_lengths[path_to_audio]

        instrument = data['instrument']

        piano_roll = read_piano_roll(data['path_to_midi'])

        start_ratio =  data['start_time'] / track_length_seconds
        end_ratio = (data['start_time'] + data['audio_len']) / track_length_seconds

        start_step = int(piano_roll.shape[0] * start_ratio)
        end_step  = int(piano_roll.shape[0] * end_ratio)

        piano_roll = piano_roll[start_step:end_step, :]
        
        if is_silent(piano_roll):
            silence_count +=1
            if keep:
                # if the piano roll is all zeros, change class to silence
                instrument = 'silence' 
                data['instrument'] = instrument
                
            else:
                continue

        new_metadata.append(data)
        pbar.set_description(f'found {silence_count} empty audio clips')

    return new_metadata

class SlakhDataset(Dataset):

    def __init__(self, 
                path_to_dataset='/home/hugo/data/slakh2100_flac',
                subset='train', 
                classes=None, # WARNING: class subsets not implemented yet
                audio_length=1, 
                generate_npz=True): # audio length in seconds per sample
        subsets = ('train', 'validation', 'test')
        assert subset in subsets, f'subset provided not in {subsets}'
        
        if classes is not None:
            raise NotImplementedError('class subsets not implemented yet:(')

        self.path_to_data = os.path.abspath(os.path.join(path_to_dataset, subset))
        assert os.path.exists(
            self.path_to_data), f'cant find path {self.path_to_data}'

        self.audio_length = audio_length
        self.chunk_len = audio_length
        self.sr = 48000

        # # path_to_metadata = os.path.join(self.path_to_data, f'SlakhDataset_metadata_len_{audio_length}.json')
        # path_to_metadata = os.path.join(self.path_to_data, f'SlakhDataset_metadata.json')

        # if not os.path.exists(path_to_metadata):
        #     self.metadata = get_slakh_metadata(path_to_dataset=self.path_to_data, 
        #                                     instruments=None)

        #     pd.DataFrame(self.metadata).to_csv(path_to_metadata, index=False)
        # else:
        #     self.metadata = pd.read_csv(path_to_metadata).to_dict('records')
        #     self.load_chunked=True
        #     # self.metadata = find_silence(self.metadata)
        #     # pd.DataFrame(self.metadata).to_csv(os.path.join(self.path_to_data, f'SlakhDataset_metadata_len_{audio_length}.json'), index=False)
        


        self.generate_npz = generate_slakh_npz(self.path_to_data, instruments=classes)
        self.has_npy = False
        if self.generate_npz:
            path_to_npz_metadata = os.path.join(self.path_to_data, f'slakh-metadata-chunk_len-{self.audio_length}-sr-{self.sr}-npy.csv')
            if os.path.exists(path_to_npz_metadata):
                self.has_npy = True
                self.metadata = pd.read_csv(path_to_npz_metadata).to_dict('records')
            else:  
                self.metadata = generate_
                pd.DataFrame(self.metadata).to_csv(path_to_npz_metadata, index=False)

        self.classes = list(set([e['instrument'] for e in self.metadata]))
        self.classes.sort()
        self.class_weights = np.array([1/pair[1] for pair in self.get_class_frequencies()])
        self.class_weights = self.class_weights / max(self.class_weights)
        print(self.get_class_frequencies())
        [print(f'{c}-{w}') for c, w in zip(self.classes, self.class_weights)]

                
    def check_metadata(self):
        missing_files = []
        for entry in self.metadata:
            if not os.path.exists(entry['path_to_audio']):
                logging.warn(f'{entry["path_to_audio"]} is missing.')
                missing_files.append(entry['path_to_audio'])

        assert len(missing_files) == 0, 'some files were missing in the dataset.\
             delete metadata file and download again, or delete missing entries from metadata'

    def save_npz_dataset(self, metadata, sr=48000, audio_len=1):
        from instrument_recognition.models.timefreq import Melspectrogram
        spectrogram = Melspectrogram(sr=sr).cuda()
        dataloader = DataLoader(self, batch_size=1, shuffle=False, collate_fn=CollateAudio(sr), num_workers=12)
        prog_bar = tqdm(enumerate(dataloader), total=len(metadata))
        print('computing spectrograms')

        error = False
        # try:
        for idx, full_entry in prog_bar:
            entry = metadata[idx]
            # no dataloader funny business
            # print(f"{entry['path_to_audio']}-{full_entry['path_to_audio']}")
            assert entry['path_to_audio'] == full_entry['path_to_audio'][0], f"{entry['path_to_audio']}-{full_entry['path_to_audio'][0]}"
            audio = full_entry['X']
            audio_chunk = audio.detach().view(-1).numpy()

            audio_chunk_path = entry['path_to_audio'].replace('slakh2100_flac', 
                        f'slakh2100_chunklen-{self.chunk_len}_sr-{sr}')
            audio_chunk_path = audio_chunk_path.replace('.flac', f'_{entry["start_time"]}')
            
            os.makedirs(os.path.dirname(audio_chunk_path), exist_ok=True)
            print(f'saving {audio_chunk_path}')
            np.save(audio_chunk_path, audio_chunk)

            entry['path_to_audio'] = audio_chunk_path
            print(full_entry['instrument'][0])
            print(entry['instrument'])
            assert entry['instrument'] == full_entry['instrument'][0] or full_entry['instrument'][0] == 'silence'

            if not full_entry['instrument'][0] == 'silence':
                metadata[idx] = entry
        # except Exception as e:
        #     print(f'an exception occured: {e}. trying to save whats left..')
        #     error = True
        return metadata, error

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

    # def _retrieve_entry(self, entry):
    #     path_to_audio = entry['path_to_audio']
    #     filename = path_to_audio.split('/')[-1]

    #     assert os.path.exists(path_to_audio), f"couldn't find {path_to_audio}"

    #     instrument = entry['instrument']

    #     data = {}
    #     # add all the keys from the entry as well
    #     data.update(entry)

    #     # if the spectrogram is already there, no need to worry abt loading audio
    #     if not self.has_npy:
    #         # import our audio using torchaudio
    #         audio, sr = torchaudio.load(path_to_audio)
    #         track_length_seconds = audio.shape[-1] / sr
    #         if 'start_time' in data:
    #             # calculate end time
    #             start_time = data['start_time'] * sr
    #             end_time = (data['start_time'] + data['audio_len']) * sr
    #             audio = audio[:, start_time:end_time]
            
    #         data['X'] = audio
    #         data['sr'] = sr
    #     else:
    #         data['X'] = data['audio']

    #     one_hot = self.get_onehot(instrument)
    #     data['one_hot'] = one_hot
    #     data['y'] =  np.argmax(one_hot)
    #     data['instrument'] = instrument

    #     return data

    def _retrieve_entry(self, entry):
        path_to_audio = entry['path_to_audio']
        filename = path_to_audio.split('/')[-1]

        assert os.path.exists(path_to_audio), f"couldn't find {path_to_audio}"

        instrument = entry['instrument']

        data = {}
        # add all the keys from the entryas well
        data.update(entry)
        # import our audio using torchaudio
        audio, sr = librosa.load(path_to_audio, sr)
        track_length_seconds = audio.shape[-1] / sr
        if self.load_chunked:
            # calculate end time
            start_time = data['start_time'] * sr
            end_time = (data['start_time'] + data['audio_len']) * sr
            audio = audio[:, start_time:end_time]

            # FOR MIDI
            # change label if audio is all silence
            # shape of piano roll is (time, pitch)
            piano_roll = read_piano_roll(data['path_to_midi'])

            start_ratio =  data['start_time'] / track_length_seconds
            end_ratio = (data['start_time'] + data['audio_len']) / track_length_seconds

            start_step = int(piano_roll.shape[0] * start_ratio)
            end_step  = int(piano_roll.shape[0] * end_ratio)

            piano_roll = piano_roll[start_step:end_step, :]
            
            data['piano_roll'] = piano_roll
            # if the piano roll is all zeros, change class to silence
            if np.all(piano_roll==0):
                instrument = 'silence'

        data['X'] = audio
        data['sr'] = sr
        
        one_hot = self.get_onehot(instrument)
        data['one_hot'] = one_hot
        data['y'] =  np.argmax(one_hot)
        data['instrument'] = instrument

        return data

    def __getitem__(self, index):
        def retrieve(index):
            return self._retrieve_entry(self.metadata[index])

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

class SlakhDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=1, num_workers=2, 
                sr=48000):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.sr = sr
        self.collate_fn = CollateAudio(self.sr)
        
    def load_dataset(self):
        path = os.path.expanduser('~/data/slakh2100_flac')
        self.train_data = SlakhDataset(
            path_to_dataset=path, 
            classes=None, 
            subset='train',
            
            audio_length=1)
        self.dataset=self.train_data
        self.val_data = SlakhDataset(
            path_to_dataset=path, 
            classes=None, 
            subset='validation',
        
            audio_length=1)
        self.test_data = SlakhDataset(
            path_to_dataset=path, 
            classes=None, 
            subset='test',
            audio_length=1)

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