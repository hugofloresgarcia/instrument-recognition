from pathlib import Path

import numpy as np
import medleydb as mdb
import tqdm

import audio_utils as au

import instrument_recognition as ir
import instrument_recognition.utils as utils

unwanted_classes =  ['Main System', 'claps', 'fx/processed sound', 'tuba', 'piccolo', 'cymbal',
                     'glockenspiel', 'tambourine', 'timpani', 'snare drum', 'clarinet section',
                      'flute section', 'tenor saxophone', 'trumpet section']

def create_dataset_entry(entry_path, audio_format, metadata_format, label, 
                         start_time, end_time, sr, effect_params, **kwargs):
    "dict wrapper with required fiels"
    return dict(entry_path=entry_path, audio_format=audio_format, metadata_format=metadata_format, 
                label=label, start_time=start_time, end_time=end_time, sr=sr, effect_params=effect_params, **kwargs)

def save_dataset_entry(audio: np.ndarray, sr: int, audio_format: str, dataset: str, 
                      partition_key: str, label: str, filename: str, start_time: float, 
                      end_time: float, effect_params: dict, 
                      **kwargs): 
    # define a path for our dataset entry 
    entry_path = Path(dataset) / partition_key / 'foreground' / label / filename
    entry_path_absolute = Path(ir.core.DATA_DIR) / entry_path

    # create a metadata dict
    metadata = create_dataset_entry(entry_path=entry_path, audio_format=audio_format, metadata_format='yaml', 
                                    label=label, start_time=start_time, end_time=end_time, sr=sr, effect_params=effect_params, **kwargs)

    # save metadata
    utils.data.save_yaml(metadata, entry_path_absolute)

    # save audio file
    au.io.write_audio_file(audio=audio, path_to_audio=entry_path_absolute, sample_rate=sr, 
                                 audio_format=audio_format, exist_ok=True)

def medleydb_make_partition_map(test_size, random_seed):
    # the first thing to do is to partition the MDB track IDs and stem IDs into only the ones we will use.
    mtrack_generator = mdb.load_all_multitracks(['V1', 'V2'])
    splits = mdb.utils.artist_conditional_split(test_size=test_size, num_splits=1, 
                                                random_state=random_seed)[0]
    partition_map = {}

    for mtrack in mtrack_generator:

        # add appropriate partition key for this mtrack
        if mtrack.track_id in splits['test']:
            partition_key = 'test'
        elif mtrack.track_id in splits['train']:
            partition_key = 'train'
        else:
            continue
        
        # add the partition dict if we havent yet
        if partition_key not in partition_map:
            partition_map[partition_key] = []
        
        # shorten name so we don't have to call
        # the very nested dict every time
        partition_list = partition_map[partition_key]

        # iterate through the stems in this mtrack
        for stem_id, stem in mtrack.stems.items():
            label = stem.instrument[0]
            
            # continue if we don't want this class
            if label in unwanted_classes:
                continue

            # append the stem with it's corresponding info
            stem_info = dict(track_id=mtrack.track_id, stem_idx=stem.stem_idx, label=label, 
                            artist_id=mtrack.track_id.split('_')[0], path_to_audio=stem.audio_path, 
                            base_chunk_name=f'{mtrack.track_id}-{stem_id}-{label}')
            partition_list.append(stem_info)

    import instrument_recognition.utils as utils
    # get the unique set of classes for both partition
    classlists = {k: utils.data.get_classlist(metadata) for k, metadata in partition_map.items()}

    # filter classes so we only have the intersection of the two sets :)
    filtered_classes = list(set(classlists['train']) & set(classlists['test']))

    # filter out the partition map!!!
    for partition_key, metadata in partition_map.items():
        partition_map[partition_key] = [e for e in metadata if e['label'] in  filtered_classes]

    print(f'created a partition map with the following classes:\n{filtered_classes}')
    print(f'number of tracks in train set: {len(partition_map["train"])}')
    print(f'number of tracks in test set: {len(partition_map["test"])}')

    return partition_map

def save_partition_map(partition_map, name):
    save_path = Path(ir.core.ASSETS_DIR) / f'{name}-partition_map'
    utils.data.save_json(partition_map, save_path)

def split_on_silence_and_save(partition_map, target_sr, dataset, audio_format):
    for partition_key, records in partition_map.items():

        def _split_and_save(entry):
            path_to_audio = entry['path_to_audio']
            base_chunk_name = entry['base_chunk_name']
            label = entry['label']

            audio = au.io.load_audio_file(path_to_audio, target_sr)

            audio = au.librosa_input_wrap(audio)
            audio = utils.effects.trim_silence(audio, target_sr, min_silence_duration=0.5)
            audio = au.librosa_output_wrap(audio)

            if au.core._is_zero(audio):
                return

            # timestamps = au.split_on_silence(audio, target_sr, top_db=45, min_silence_duration=0.5)
    
            save_dataset_entry(audio, sr=target_sr, audio_format=audio_format, dataset=dataset, 
                                partition_key=partition_key, label=label, 
                                filename=f'{base_chunk_name}',
                                start_time=0, end_time=len(audio)/target_sr, effect_params=None)

        records_pb = tqdm.tqdm(records)
        for record in records_pb:
            records_pb.set_description(f'processing {record["path_to_audio"].split("/")[-1]}')
            _split_and_save(record)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True)
    
    parser.add_argument('--audio_format', type=str, default='wav')
    parser.add_argument('--test_size', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=ir.RANDOM_SEED)
    parser.add_argument('--sample_rate', type=int, default=ir.SAMPLE_RATE)
    
    args = parser.parse_args()

    # make partition map, save to assets dir
    partition = medleydb_make_partition_map(test_size=args.test_size, random_seed=args.seed)
    save_partition_map(partition, args.dataset)
    split_on_silence_and_save(partition, args.sample_rate, args.dataset, args.audio_format)