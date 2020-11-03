import pandas as pd
import medleydb as mdb
import pytorch_lightning as pl
import numpy as np
import torch
from tqdm import tqdm
import time
from torch.utils.data import DataLoader
import torchaudio
import librosa
import sox
import soundfile as sf
import os
import json
from instrument_recognition.utils import audio_utils

class_dict = {
    'Main System': 'other',
    # 'accordion'
    # 'acoustic guitar'
    # 'alto saxophone'
    # 'auxiliary percussion'
    # 'bamboo flute'
    # 'banjo'
    # 'baritone saxophone'
    # 'bass clarinet'
    # 'bass drum'
    # 'bassoon'
    # 'bongo'
    # 'brass section'
    # 'cello'
    'cello section': 'cello',
    'chimes': 'other',
    'claps': 'percussion',
    # 'clarinet'
    'clarinet section': 'clarinet',
    # 'clean electric guitar'
    'cymbal': 'drum set',
    'darbuka': 'other',
    # 'distorted electric guitar'
    # 'dizi'
    # 'double bass'
    'doumbek': 'other',
    # 'drum machine'
    # 'drum set'
    # 'electric bass'
    # 'electric piano'
    # 'erhu'
    'female singer': 'female vocalist',
    # 'flute'
    'flute section': 'flute',
    # 'french horn'
    'french horn section': 'french horn',
    'fx/processed sound': 'other',
    # 'glockenspiel'
    # 'gong'
    # 'gu'
    # 'guzheng'
    # 'harmonica'
    # 'harp'
    # 'horn section'
    'kick drum': 'drum set',
    # 'lap steel guitar'
    'liuqin': 'other',
    'male rapper': 'male vocalist',
    'male singer': 'male vocalist',
    'male speaker': 'male vocalist',
    # 'mandolin'
    # 'melodica'
    # 'oboe'
    # 'oud'
    # 'piano'
    # 'piccolo'
    'sampler': 'other', 
    'scratches': 'other', 
    # 'shaker': 'percussion', 
    'snare drum': 'drum set',
    # 'soprano saxophone'
    # 'string section'
    # 'synthesizer'
    # 'tabla'
    'tack piano': 'piano',
    'tambourine': 'percussion',
    # 'tenor saxophone'
    # 'timpani'
    'toms': 'percussion',
    # 'trombone'
    'trombone section': 'trombone',
    # 'trumpet'
    'trumpet section': 'trumpet',
    # 'tuba'
    # 'vibraphone'
    # 'viola'
    'viola section': 'viola',
    # 'violin'
    'violin section': 'violin',
    # 'vocalists'
    # 'yangqin'
    # 'zhongruan'
}

def generate_medleydb_metadata(chunk_len=None, sr=48000):
    """
    creates medleydb metadata. 
    params:
        chunk_len (float): length of each audio chunk (in seconds),
            if none, the entries arent chunked
    returns:
        metadata (list of dicts): each dict
        entry has the following fields:
        - path_to_audio (str)
        - instrument (str)
        - activations (list of lists): 
            List of time, activation confidence pairs
        
    """
    metadata = []

    # load all tracks
    mtrack_generator = mdb.load_all_multitracks(['V1', 'V2'])
    mtrack_generator = tqdm(mtrack_generator, desc='generating medleydb metadata')

    for mtrack in mtrack_generator:
        for stem_id, stem in mtrack.stems.items():
            # LOAD AUDIO
            path_to_audio = stem.audio_path
            audio, sr = librosa.load(path_to_audio, mono=True, sr=sr)

            # TRIM ALL SILENCE OFF AUDIO
            tfm = sox.Transformer()
            tfm.silence(min_silence_duration=0.3, buffer_around_silence=False)
            audio = tfm.build_array(input_array=audio, sample_rate_in=sr)

            # determine how many 1 second chunks we can get out of this
            n_chunks = int(np.ceil(len(audio)/(chunk_len*sr)))

            # iterate through chunks
            for start_time in range(n_chunks):
                # zero pad 
                audio_chunk = get_audio_chunk(audio, sr, start_time, chunk_len)

                # format save path
                audio_chunk_path = stem.audio_path.replace('medleydb/Audio', 
                            f'medleydb/Audio_chunklen-{chunk_len}_sr-{sr}')
                audio_chunk_path = audio_chunk_path.replace('.wav', f'_{start_time}.npy')
                
                os.makedirs(os.path.dirname(audio_chunk_path), exist_ok=True)
                print(f'saving {audio_chunk_path}')
                np.save(audio_chunk_path, audio_chunk)

                entry = dict(
                    # the definite stuff
                    path_to_audio=audio_chunk_path, 
                    instrument=stem.instrument, 
                    duration=chunk_len, 
                    start_time=start_time, 
                    
                    # the extras
                    track_id=mtrack.track_id, 
                    stem_idx=stem.stem_idx)
                
                metadata.append(entry)

    return metadata

def get_audio_chunk(audio, sr, start_time, chunk_len):
    """ given MONO audio 1d array, get a chunk of that 
    array determined by start_time (in seconds), chunk_len (in seconds)
    pad with zeros if necessary
    """
    assert audio.ndim == 1, 'audio needs to be 1 dimensional'
    duration = audio.shape[-1] / sr

    start_idx = start_time * sr
    end_time = start_time + chunk_len 
    end_time = min([end_time, duration])
    end_idx = int(end_time * sr)
    chunked_audio = audio[start_idx:end_idx]

    if not len(audio) / sr == chunk_len * sr:
        chunked_audio = audio_utils.zero_pad(chunked_audio, sr * chunk_len)
    return chunked_audio

class MDBDataset(torch.utils.data.Dataset):

    def __init__(self, 
                path_to_dataset='/home/hugo/data/medleydb',
                sr=48000, transform=None, train=True, 
                chunk_len=1, random_seed=420): 
        self.sr = sr
        self.transform = transform
        self.chunk_len = chunk_len # in seconds

        # define metadata path
        path_to_metadata = os.path.join(
            path_to_dataset, f'medleydb-metadata-chunk_len-{self.chunk_len}-sr-{self.sr}-npy.csv')

        # if we don't have the metadata, we need to generate it. 
        if not os.path.exists(path_to_metadata):
            print('generating dataset and metadata')
            self.metadata = generate_medleydb_metadata(chunk_len=self.chunk_len, sr=self.sr)
            
            print('done generating dataset and metadata')
            pd.DataFrame(self.metadata).to_csv(path_to_metadata, index=False)
        else:
            print(f'found metadata: {path_to_metadata}')
            self.metadata = pd.read_csv(path_to_metadata).to_dict('records')

        # only keep train/test metadata, if train is not NOne
        if train is not None:
            self.filter_train_test_split(train=train, seed=random_seed)
        
        # strip unwated characters
        for e in self.metadata:
            e['instrument'] = str(e['instrument'].strip('[]\''))
        self.classes = list(set([e['instrument'] for e in self.metadata]))
        
        # sort in alphabetical order
        self.classes.sort()

        # remap classes
        self.remap_classes(class_dict)

        self.class_weights = np.array([1/pair[1] for pair in self.get_class_frequencies()])
        self.class_weights = self.class_weights / max(self.class_weights)
        # [print(f'{c}-{w}') for c, w in zip(self.classes, self.class_weights)]

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        entry = self.metadata[idx]
        # load audio using numpy
        audio = np.load(entry['path_to_audio'], allow_pickle=False)

        audio = torch.from_numpy(audio)

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

        item = dict(entry) # copy entry so we don't update the metadata
        item.update(data) # add all that other shiz just incase

        return item

    def filter_train_test_split(self, train=True, seed=420):
        """ filter self.metadata and only keep  the classes belonging 
        to an artist conditional split provided my mdbutils.
        the point of this is to keep us from using samples belonging to the same
        song for both training and validaton
        """
        splits = mdb.utils.artist_conditional_split(test_size=0.15, num_splits=1, 
                                                random_state=seed)
        key = 'train' if train else 'test'
        splits = splits[0][key]

        self.metadata = [e for e in self.metadata if e['trackid'] in splits]

    def remap_classes(self, class_dict):
        """ remap instruments according to a dictionary provided
        that is, if 
            entry['instrument'] == 'electric guitar'
            
            and

            class_dict['electric guitar'] = 'guitar'

            then entry['instrument'] will be changed to 'guitar'
        """
        for idx, c in enumerate(self.classes):
            if c in class_dict.keys():
                self.classes[idx] = class_dict[c]

        self.classes = list(set(self.classes))
        self.classes.sort()

        for entry in self.metadata:
            entry['instrument'] = entry['instrument'].strip('[]\'')
            if entry['instrument'] in class_dict.keys():
                entry['instrument'] = class_dict[entry['instrument']]

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

    def get_onehot(self, label):
        return np.array([1 if l == label else 0 for l in self.classes])

class MDBDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=1, num_workers=2, 
                sr=48000):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.sr = sr
        self.collate_fn = CollateAudio(self.sr)
        
    def load_dataset(self):
        path = os.path.expanduser('~/data/slakh2100_flac')
        self.train_data = MDBDataset(sr=self.sr, train=True, chunk_len=1, 
                                random_seed=420)

        self.val_data = MDBDataset(sr=self.sr, train=False, chunk_len=1, 
                                random_seed=420)
        
        self.test_data = MDBDataset(sr=self.sr, train=False, chunk_len=1, 
                                random_seed=420)

    def _load_dataset(self):
        path = os.path.expanduser('~/data/slakh2100_flac')
        self.dataset = MDBDataset()

        splits = [int(ratio*len(self.dataset)) for ratio in (0.7, 0.2, 0.1)]
        while not (sum(splits) == len(self.dataset)):
            splits[0] +=1
        
        train_subset, val_subset, test_subset = \
            torch.utils.data.random_split(dataset=self.dataset, 
                                        lengths=splits)

        train_data = MDBDataset()
        val_data = MDBDataset()
        test_data = MDBDataset()

        train_data.metadata = [train_data.metadata[index] for index in train_subset.indices]
        val_data.metadata = [val_data.metadata[index] for index in val_subset.indices]
        test_data.metadata =  [test_data.metadata[index] for index in test_subset.indices]

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        print('TRAIN DATASET')
        [print(pair) for pair in self.train_data.get_class_frequencies()]
        print('VALIDATION DATASET')
        [print(pair) for pair in self.val_data.get_class_frequencies()]
        print('TEST DATASET')
        [print(pair) for pair in self.test_data.get_class_frequencies()]

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

if __name__ == "__main__":
    import time
    dataset = MDBDataset(train=True)
    print(dataset.get_class_frequencies())

    dataset = MDBDataset(train=False)
    print(dataset.get_class_frequencies())