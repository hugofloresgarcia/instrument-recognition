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
from instrument_recognition.utils.transforms  import random_transform

class_dict = {
    'Main System': 'other',
    # 'accordion'
    # 'acoustic guitar'
    # 'alto saxophone'
    'auxiliary percussion': 'percussion',
    # 'bamboo flute'
    # 'banjo'
    # 'baritone saxophone'
    # 'bass clarinet'
    # 'bass drum'
    # 'bassoon'
    'bongo': 'percussion',
    'brass section': 'brass',
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
    'french horn': 'brass',
    'french horn section': 'brass',
    'fx/processed sound': 'other',
    'glockenspiel' : 'mallet percussion',
    'gong': 'percussion',
    # 'gu'
    # 'guzheng'
    # 'harmonica'
    # 'harp': 'other',
    'horn section': 'brass',
    'kick drum': 'drum set',
    'lap steel guitar': 'acoustic guitar',
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
    'shaker': 'percussion', 
    'snare drum': 'drum set',
    # 'soprano saxophone'
    # 'string section'
    # 'synthesizer'
    'tabla': 'percussion',
    'tack piano': 'piano',
    'tambourine': 'percussion',
    # 'tenor saxophone'
    'timpani': 'percussion',
    'toms': 'percussion',
    'trombone': 'brass',
    'trombone section': 'brass',
    'trumpet': 'brass',
    'trumpet section': 'brass',
    'tuba': 'other',
    'vibraphone': 'mallet percussion',
    # 'viola'
    'viola section': 'viola',
    # 'violin'
    'violin section': 'violin',
    # 'vocalists'
    # 'yangqin'
    # 'zhongruan'
}

# rack up the workers
from multiprocessing import Pool

def process_mtrack(args):
    metadata = []
    mtrack, chunk_len, sr, hop_len, splits, transform_train = args
    # figure out whether we are going to add FX to this track or not
    if splits is not None:
        if mtrack.track_id in splits['train'] and transform_train:
            transform_audio = True
        else: 
            transform_audio = False

    for stem_id, stem in mtrack.stems.items():
        # LOAD AUDIO
        path_to_audio = stem.audio_path
        audio, sr = librosa.load(path_to_audio, mono=True, sr=sr)

        tfm = sox.Transformer()
        tfm.silence(min_silence_duration=0.3, buffer_around_silence=False)
        audio = tfm.build_array(input_array=audio, sample_rate_in=sr)
        audio_len = len(audio)
        
        # determine how many chunk_len second chunks we can get out of this
        n_chunks = int(np.ceil(audio_len/(chunk_len*sr)))
        audio = audio_utils.zero_pad(audio, n_chunks * sr)

        # format save path
        # send them bois to CHONK
        audio_npy_path = stem.audio_path.replace('data/medleydb/Audio', 
                    f'CHONK/data/medleydb/Audio_chunklen-{chunk_len}_sr-{sr}_hop-{hop_len}')
        audio_npy_path = audio_npy_path.replace('.wav', f'.npy')

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

        # iterate through chunks
        for start_time in np.arange(0, n_chunks, hop_len):

            # zero pad 
            # audio_chunk = get_audio_chunk(audio, sr, start_time, chunk_len)

            # # format save path
            # # SEND them bois to CHONK
            # audio_chunk_path = stem.audio_path.replace('data/medleydb/Audio', 
            #             f'CHONK/data/medleydb/Audio_chunklen-{chunk_len}_sr-{sr}_hop-{hop_len}')
            # audio_chunk_path = audio_chunk_path.replace('.wav', f'/{start_time}.npy')
            
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
                # the definite stuff
                path_to_audio=audio_chunk_path, 
                instrument_list=stem.instrument, 
                instrument=stem.instrument[0],
                duration=chunk_len, 
                start_time=start_time, 
                
                # the extras
                track_id=mtrack.track_id, 
                stem_idx=stem.stem_idx, 
                audio_transformed=transform_audio)
            
            metadata.append(entry)

    return metadata

def generate_medleydb_metadata(chunk_len=1, sr=48000, hop_len=1.0, splits=None, transform_train=True):
    """
    creates medleydb metadata. 
    params:
        chunk_len (float): length of each audio chunk (in seconds),
            if none, the entries arent chunked
        sr (int): export sample rate
        hop_len (float): hop length when creating new samples
        splits (dict[list]): medleydb artist cond split (for trainsfroms)
        transform_train (bool): whether to apply transforms for the 
            train test split track ids. If splits is None, then this doesn't matter
    returns:
        metadata (list of dicts): each dict
        entry has the following fields:
        - path_to_audio (str)
        - instrument (str)
        - activations (list of lists): 
            List of time, activation confidence pairs
        
    """
    # load all tracks
    mtrack_generator = mdb.load_all_multitracks(['V1', 'V2'])

    pool = Pool()

    args = []
    for mtrack in mtrack_generator:
        args.append((mtrack, chunk_len, sr, hop_len, splits, transform_train))
        
    print(len(args))

    # do the thing!
    metadata = pool.map(process_mtrack, args)
    metadata = [entry for submetadata in metadata for entry in submetadata]
    pool.close()
    pool.join()
    print(f'converting to list..')

    print(len(metadata))
    # exit()

    return metadata

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

class MDBDataset(torch.utils.data.Dataset):

    def __init__(self, 
                path_to_dataset='/home/hugo/CHONK/data/medleydb',
                sr=48000, transform=None, train=True, 
                chunk_len=1, random_seed=4, hop_len=1): 
        self.sr = sr
        self.transform = transform
        self.chunk_len = chunk_len # in seconds
        self.hop_len = hop_len

        # get the split (for generating metadata)
        self.splits = mdb.utils.artist_conditional_split(test_size=0.15, num_splits=1, 
                                                random_state=random_seed)[0]

        # define metadata path
        path_to_metadata = os.path.join(
            path_to_dataset, f'medleydb-metadata-chunk_len-{self.chunk_len}-sr-{self.sr}-hop-{self.hop_len}-npy-memmap.csv')

        # if we don't have the metadata, we need to generate it. 
        if not os.path.exists(path_to_metadata):
            print('generating dataset and metadata')
            self.metadata = generate_medleydb_metadata(chunk_len=self.chunk_len, sr=self.sr, hop_len=self.hop_len, 
                                                        splits=self.splits, transform_train=True)
            
            print('done generating dataset and metadata')
            pd.DataFrame(self.metadata).to_csv(path_to_metadata, index=False)
        else:
            print(f'found metadata: {path_to_metadata}')
            self.metadata = pd.read_csv(path_to_metadata).to_dict('records')

        # strip unwated characters
        self.classes = self.get_classlist(self.metadata)

        # remap classes
        self.remap_classes(self.metadata, class_dict)

        # only keep train/test metadata, if train is not NOne
        if train is not None:
            self.filter_train_test_split(train=train, seed=random_seed)

        # filter out all classes with 'other' tag
        self.metadata = [e for e in self.metadata if not e['instrument'] == 'other']
        self.classes = self.get_classlist(self.metadata)

        print(self.get_class_frequencies())

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

    def get_classlist(self, metadata):
        for e in metadata:
            e['instrument'] = str(e['instrument'].strip('[]\''))
        classes = list(set([e['instrument'] for e in metadata]))
        classes.sort()

        return classes

    def filter_train_test_split(self, train=True, seed=429):
        """ filter self.metadata and only keep  the classes belonging 
        to an artist conditional split provided my mdbutils.
        the point of this is to keep us from using samples belonging to the same
        song for both training and validaton
        """
        splits = self.splits
        key = 'train' if train else 'test'

        train_metadata = [e for e in self.metadata if e['track_id'] in splits['train']]
        test_metadata = [e for e in self.metadata if e['track_id'] in splits['test']]

        train_classes = self.get_classlist(train_metadata)
        test_classes = self.get_classlist(test_metadata)

        # find out what classes are missing
        missing_test_classes = []
        for c in train_classes:
            if c not in test_classes:
                missing_test_classes.append(c)

        # print(f'classes missing from test data:')
        # print(missing_test_classes)

        missing_train_classes = []
        for c in test_classes:
            if c not in train_classes:
                missing_train_classes.append(c)

        # print(f'classes missing from train data:')
        # print(missing_train_classes)

        # first, remap missing classes in train set to other
        soft_remap = {
            
            'clarinet': 'clarinet', 
            'saxophone': 'saxophone', 
            'drum': 'drum set', 
            'flute': 'flute', 
            'trombone': 'trombone', 
            'piano': 'piano',
            'male': 'male vocalist'}

        hard_remap = {}
        for classlist in [missing_train_classes, missing_test_classes]:
            for c in classlist:
                for cand in soft_remap:
                    # if we can find the key in the string
                    # then remap that class to the more general term
                    remapped = False
                    if cand in c:
                        hard_remap[c] = soft_remap[cand]
                        remapped = True
                if not remapped:
                    hard_remap[c] = 'other'
        
        self.remap_classes(train_metadata, hard_remap)
        self.remap_classes(test_metadata, hard_remap)

        if key == 'train':
            self.metadata = train_metadata
            self.classes = self.get_classlist(train_metadata)
            train_classes = self.get_classlist(train_metadata)
            test_classes = self.get_classlist(test_metadata)
        else:
            self.metadata = test_metadata
            self.classes = self.get_classlist(test_metadata)
            train_classes = self.get_classlist(train_metadata)
            test_classes = self.get_classlist(test_metadata)

        # find out what classes are missing
        missing_test_classes = []
        for c in train_classes:
            if c not in test_classes:
                missing_test_classes.append(c)

        # print(f'FFclasses missing from test data:')
        # print(missing_test_classes)
        missing_train_classes = []
        for c in test_classes:
            if c not in train_classes:
                missing_train_classes.append(c)

        # print(f'classes missing from train data:')
        # print(missing_train_classes)

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
        path = os.path.expanduser('~/CHONK/data/slakh2100_flac')
        self.train_data = MDBDataset(sr=self.sr, train=True, chunk_len=1, 
                                random_seed=4)
        
        self.dataset = self.train_data

        self.val_data = MDBDataset(sr=self.sr, train=False, chunk_len=1, 
                                random_seed=4)
        
        self.test_data = MDBDataset(sr=self.sr, train=False, chunk_len=1, 
                                random_seed=4)

    def _load_dataset(self):
        path = os.path.expanduser('~/CHONK/data/slakh2100_flac')
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
    # find a random split that maximizes overlap between instrument classes
    import time
    train_dataset = MDBDataset(train=True, random_seed=4)
    # print([p for p in dataset.get_class_frequencies()])

    val_dataset = MDBDataset(train=False, random_seed=4)
    # print([p for p in dataset.get_class_frequencies()])

    
    # scoreboard = {}
    # for seed in range(50):
    #     train_dataset = MDBDataset(train=True, random_seed=seed)
    #     val_dataset = MDBDataset(train=False, random_seed=seed)

    #     missing_val_classes = []
    #     for c in train_dataset.classes:
    #         if c not in val_dataset.classes:
    #             missing_val_classes.append(c)
        
    #     scoreboard[seed] = len(missing_val_classes)

    # scoreboard = sorted(scoreboard.items(), key=lambda x: x[1])
    # print(scoreboard)
    
