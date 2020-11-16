import os
from multiprocessing import Pool

import numpy as np
import tqdm
import librosa
import soxbindings as sox
import soundfile as sf
import pandas as pd
import medleydb as mdb

import instrument_recognition as utils 

def get_abspath(path):
    return os.path.abspath(os.path.expanduser(path))

def _check_audio_types(audio):
    assert audio.ndim == 1, "audio must be mono"
    assert isinstance(audio, np.ndarray)

def load_audio_file(path_to_audio, sr=48000):
    """ wrapper for loading mono audio with librosa
    returns:
        audio (np.ndarray): monophonic audio with shape (samples,) 
    """
    audio, sr = librosa.load(path_to_audio, mono=True, sr=sr)
    return audio

def trim_silence(audio, sr, min_silence_duration=0.3):
    """ trim silence from audio array using sox
    """
    _check_audio_types(audio)
    tfm = sox.Transformer()
    tfm.silence(min_silence_duration=min_silence_duration, 
                buffer_around_silence=False)
    audio = tfm.build_array(input_array=audio, sample_rate_in=sr)
    audio = audio.T[0]
    return audio

def get_audio_chunk(audio, sr, start_time, chunk_size):
    """ given MONO audio 1d array, get a chunk of that 
    array determined by start_time (in seconds), chunk_size (in seconds)
    pad with zeros if necessary
    """
    _check_audio_types(audio)

    duration = audio.shape[-1] / sr

    start_idx = int(start_time * sr)

    end_time = start_time + chunk_size 
    end_time = min([end_time, duration])

    end_idx = int(end_time * sr)

    chunked_audio = audio[start_idx:end_idx]

    if not len(audio) / sr == chunk_size * sr:
        chunked_audio = utils.audio.zero_pad(chunked_audio, sr * chunk_size)
    return chunked_audio

def save_windowed_audio_events(audio, sr, chunk_size, hop_size, base_chunk_name, 
                             label, path_to_output, metadata_extras):
    """ this function will chunk a monophonic audio array 
    into chunks as determined by chunk_size and hop_size. 
    The output audio file will be saved to a foreground folder, scaper style, 
    under a subdirectory with a name determined by label. 
    Besides the output audio file, it will create a .json file with metadata
    for each corresponding audio file
    args:
        audio (np.ndarray): audio arrayshape (samples,)
        sr (int): sample rate
        chunk_size (float): chunk size, in seconds, to cut audio into
        hop_size (float): hop size, in seconds, to window audio
        base_chunk_name (str): base name for the output audio chunk file. The format for 
            the saved audio chunks is f"{base_chunk_name}-{start_time}"
        label (str): label for this audio example. 
        path_to_output (str): base path to save this example
        metadata_extras (dict): adds these extra entries to each metadata dict
    """
    _check_audio_types(audio)

    # determine how many chunk_size chunks we can get out of the audio array
    audio_len = len(audio)
    n_chunks = int(np.ceil(audio_len/(chunk_size*sr))) # use ceil because we can zero pad

    metadata = []

    # iterate through chunks
    for start_time in np.arange(0, n_chunks, hop_size):
        # round start time bc of floating pt errors
        start_time = np.around(start_time, 4)

        # get current audio_chunk
        audio_chunk = get_audio_chunk(audio, sr, start_time, chunk_size)

        audio_chunk_name = base_chunk_name + f'-{start_time}.wav'
        audio_chunk_path = os.path.join(path_to_output, 
                                        f'{label}', 
                                        audio_chunk_name)

        # make path for metadata
        chunk_metadata_path = audio_chunk_path.replace('.wav', '.json')
        
        os.makedirs(os.path.dirname(audio_chunk_path), exist_ok=True)

        # make a metadata entry
        entry = dict()
        entry.update(metadata_extras)
        entry.update(dict(
            path_to_audio=audio_chunk_path,
            label=label,
            chunk_size=chunk_size, 
            start_time=start_time, 
            sr=sr))

        # if both paths, already exists, bail
        if not os.path.exists(audio_chunk_path):
            sf.write(audio_chunk_path, audio_chunk, sr, 'PCM_24')
            utils.data.save_dict_json(entry, chunk_metadata_path)
        else:
            print(f'already found: {audio_chunk_path} and {chunk_metadata_path}')

        metadata.append(entry)
    
    return metadata

# args = path_to_output, mtrack, chunk_size, sr, hop_size
def _process_mdb_track(args):
    metadata = []
    path_to_output, mtrack, chunk_size, sr, hop_size = args
    # figure out whether we are going to add FX to this track or not

    for stem_id, stem in mtrack.stems.items():
        # get the audio path 
        path_to_audio = stem.audio_path

        # get the base pattern for our audio files
        base_chunk_name = f'{mtrack.track_id}-{stem_id}'

        # if we already found the base chunk pattern in our o
        # output path, then skip this stem
        print(f'processing {base_chunk_name}')

        try:
            audio = load_audio_file(path_to_audio, sr)

            # remove silent regions
            audio = trim_silence(audio, sr, min_silence_duration=0.3)

            # make sure we add medleydb metadata in just in case
            extras = dict(track_id=mtrack.track_id,
                        stem_idx=stem.stem_idx,
                        instrument_list=stem.instrument)

            # chunk up the audio and save it
            save_windowed_audio_events(
                audio=audio, 
                sr=sr, 
                chunk_size=chunk_size, 
                hop_size=hop_size,
                base_chunk_name=base_chunk_name, 
                label=stem.instrument[0],
                path_to_output=path_to_output, 
                metadata_extras=extras)
        except Exception as e:
            print(f'exception occured: {e}')
            print(f'FAILED TO LOAD: {path_to_audio}')
            stem_metadata = [{'track_id': mtrack.track_id, 
                            'stem_idx': stem.stem_idx, 
                            'error': True}]
            
def generate_medleydb_samples(path_to_output, sr=48000, chunk_size=1.0, 
                              hop_size=1.0, num_workers=-1):

    mtrack_generator = mdb.load_all_multitracks(['V1', 'V2'])
    args = []
    for mtrack in mtrack_generator:
        args.append((path_to_output, mtrack, chunk_size, sr, hop_size))

    # run as a single process if num workers is less than 1
    if num_workers == 0:
        for arg in args:
            _process_mdb_track(arg)
    else:
        num_workers = None if num_workers < 0 else num_workers 
        pool = Pool(num_workers)

        # do the thing!
        pool.map(_process_mdb_track, args)
        
        pool.close()
        pool.join()
    
def fix_metadata_and_save_separate_dicts(metadata):
    pbar = tqdm.tqdm(metadata)

    for entry in pbar:
        if not isinstance(entry['path_to_audio'], str):
            print(entry)
            continue
        path_to_json = entry['path_to_audio'].replace('.wav', '.json')
        utils.data.save_dict_json(entry, path_to_json)
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_data', type=str, required=True)
    parser.add_argument('--path_to_output', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)

    parser.add_argument('--sr', type=int, default=48000)
    parser.add_argument('--chunked', type=utils.train.str2bool, default=True)
    parser.add_argument('--chunk_size', type=float, default=1.0)
    parser.add_argument('--hop_size', type=float, default=1.0)

    parser.add_argument('--num_workers', type=int, default=-1)

    args = parser.parse_args()

    generate_medleydb_samples(
        path_to_output=args.path_to_output, 
        sr=args.sr,
        chunk_size=args.chunk_size, 
        hop_size=args.hop_size, 
        num_workers=args.num_workers)
