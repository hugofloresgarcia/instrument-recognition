import numpy as np
import torchaudio
import torch
import pandas as pd
from tqdm import tqdm
import pypianoroll

from instrument_recognition.utils import audio_utils


def is_silent(piano_roll):
    """
    figure out if piano roll if full of zeros
    """
    if np.all(piano_roll==0):
        return True
    else:
        return False


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


def debatch(data):
    """
     convert batch size 1 to None
    """
    for key in data:
        if isinstance(data[key], list):
            assert len(data[key]) == 1, "can't debatch with batch size greater than 1" 
            data[key] = data[key][0]
    return data

def dont_collate(batch):
    """
    yes
    """
    return batch

def collate_to_lists(batch):
    """
    collate batches w audio of variable lengths into lists
    """
    data = {}
    for e in batch:
        for key in e:
            if key not in data:
                data[key] = []
            data[key].append(e[key])
    return data

def collate_audio(batch):
    """
    collate a batch of dataset samples and preprocess audio?
    """
    data = {}

    audio = []
    labels = []
    onehot = []
    for e in batch:
        a = e['audio']
        l = e['labels']

        # resample audio to 48k (bc openl3)
        a = audio_utils.resample(a, int(e['sr']), 48000)
        n_chunks = np.ceil(a.shape[-1] / 48000)
        n_chunks = int(n_chunks) if n_chunks < 10 else 9
        a = a.view(-1)
        a = audio_utils.zero_pad(a, n_chunks * 48000)
        a = a[0:n_chunks * 48000]
        if isinstance(a, np.ndarray):
            a = torch.from_numpy(a)
        a = a.view((-1,1, 48000))
        
        l = torch.full((len(a),), l, dtype=torch.int64)
        o = torch.zeros((len(a), 19))
        o[:, l] = 1
        audio.append(a)
        labels.append(l)
        onehot.append(o)
    
    audio = torch.cat(audio)
    labels = torch.cat(labels)
    onehot = torch.cat(onehot)

    data['audio'] = audio
    data['labels'] = labels
    data['onehot'] = onehot
    data['filename'] = [e['filename'] for e in batch]

    return data


def is_silent_entry(data):
    path_to_audio = data['path_to_audio']
    if 'total_track_length' in data:
        track_length_seconds = data['total_track_length']
    else:
        audio, sr = torchaudio.load(path_to_audio)
        track_length_seconds = audio.shape[-1] / sr

    instrument = data['instrument']

    if 'piano_roll' in data:
        piano_roll = data['piano_roll']
    else:
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

def expand_metadata_per_timestep(metadata, time_res, remove_silent_pianorolls=False):
    """
    uh, I don't know how to explain this
    so if I have metadata for single track and the track is
    3 minutes long, and my model processes 10 seconds at a time, 
    I would like to load my audio in batches of 10 seconds so I can
    have a uniform batch size. This creates a new set of metadata with examples expanded
    to whatever time resuloution is desired

    metadata must be a list of dicts with at least a 'path_to_audio' attribute
    each dict in the metadata will have two new entries:
        start_time: start time of the audio clip (in seconds)
        audio_len: length of audio clip (in seconds)
    """

    extended_metadata = []
    pbar = tqdm(metadata, total=len(metadata))
    for entry in pbar:
        assert 'path_to_audio' in entry, f"metadata must have attribute 'path_to_audio' but received {entry}"
        pbar.set_description(f'expanding {entry["path_to_audio"]}')
        # load audio
        path_to_audio = entry['path_to_audio']
        audio, sr = torchaudio.load(path_to_audio)
        track_length_seconds = audio.shape[-1] / sr

        if remove_silent_pianorolls:
            try:
                entry['piano_roll'] = read_piano_roll(entry['path_to_midi'])
            except:
                print(f'failed to load: {entry["instrument"]}')
                continue

        chunk_len = time_res * sr

        # find out how many chunks we will have in total
        n_chunks = int(np.ceil(audio.shape[-1] / chunk_len))

        new_entries = []
        for i in range(n_chunks):
            start_time = i * time_res
            new_entry = dict(
                start_time=start_time,
                audio_len=time_res, 
                total_track_length=track_length_seconds)
            new_entry.update(entry)
            if remove_silent_pianorolls:
                # don't append the entry if it's all silence on the piano roll
                if is_silent_entry(new_entry):
                    continue
            new_entries.append(new_entry)
        
        extended_metadata.extend(new_entries)

    return extended_metadata
