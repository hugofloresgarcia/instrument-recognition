import os

import torch
import pandas as pd
from tqdm.contrib.concurrent import process_map

from instrument_recognition.datasets.audio_dataset import AudioDataset
import instrument_recognition.datasets.utils as data_utils
from instrument_recognition.utils import effects, train_utils


def augment_from_file_to_file(input_path, output_path, effect_chain=None):
    effect_chain = ['compand','overdrive', 'eq', 'pitch', 'speed', 
                    'phaser', 'flanger', 'reverb', 'chorus', 'speed', 
                    'lowpass']
    tfm, effect_params = effects.get_random_transformer(effect_chain)
    tfm.build_file(input_path, output_path)
    return effect_params

def augment_metadata_entry(entry, path_to_data, path_to_output):
    # print(f'augmenting: {entry["path_to_audio"]}')
    # first, get new paths for entry
    if not isinstance(entry['path_to_audio'], str):
        return {}
    path_to_augmented_audio = entry['path_to_audio'].replace(path_to_data, path_to_output)
    os.makedirs(os.path.dirname(path_to_augmented_audio), exist_ok=True)
    effect_params = augment_from_file_to_file(entry['path_to_audio'], path_to_augmented_audio)
    entry['effect_params'] = effect_params
    entry['path_to_audio-augmented'] = path_to_augmented_audio

    # save entry using json
    path_to_json = path_to_augmented_audio.replace('.wav', '.json')
    data_utils.save_dict_json(entry, path_to_json)

    return entry

def augment_metadata_entry_unpack(kwargs):
    return augment_metadata_entry(**kwargs)

def augment_dataset(path_to_data, path_to_output, num_workers=0):
    # load metadata (should be a list of dicts)
    path_to_metadata = data_utils.get_path_to_metadata(path_to_data)
    metadata = data_utils.load_metadata(path_to_metadata)

    if num_workers == 0:
        # just iterate through every sample and send to soxbindings?
        new_metadata = []
        for entry in metadata:
            entry = augment_metadata_entry(entry, path_to_data, path_to_output)
        new_metadata.append(entry)
    else:
        num_workers = None if num_workers < 0 else num_workers
        # pool = Pool(num_workers)

        args = [dict(entry=e, path_to_data=path_to_data, path_to_output=path_to_output) for e in metadata]
        
        # new_metadata = pool.map(augment_metadata_entry_unpack, args)
        new_metadata = process_map(augment_metadata_entry_unpack, args, max_workers=num_workers, chunksize=4)
    
    path_to_output_metadata = data_utils.get_path_to_metadata(path_to_output)
    data_utils.save_metadata(new_metadata, path_to_output_metadata)

def get_effect_chain_statistics(metadata):
    stats = {}
    for entry in metadata:
        for effect, params in entry['effect_params'].items():
            stats[effect] += 1

    print(stats)            

def get_effect_chain_statistics_from_metadata_file(path_to_data):
    path_to_metadata = data_utils.get_path_to_metadata(path_to_data)
    metadata = data_utils.load_metadata(path_to_metadata)

    get_effect_chain_statistics(metadata)

if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_data', type=str, required=True)
    parser.add_argument('--path_to_output', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--get_effect_stats', type=train_utils.str2bool, default=False)

    args = parser.parse_args()

    if args.get_effect_stats:
        get_effect_chain_statistics_from_metadata_file(args.path_to_data)
    else:
        augment_dataset(args.path_to_data, args.path_to_output, args.num_workers)

