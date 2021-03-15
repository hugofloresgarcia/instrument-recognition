from pathlib import Path

import numpy as np
import medleydb as mdb
import tqdm
import librosa

import audio_utils as au

import instrument_recognition as ir



unwanted_classes =  ['Main System', 'fx/processed sound', 'sampler']

remap = {
    'claps': 'auxiliary percussion', 
    'tuba': 'brass', 
    'piccolo': 'flute', 
    'cymbal': 'auxiliary percussion', 
    'glockenspiel': 'auxiliary percussion', 
    'tambourine': 'auxiliary percussion', 
    'timpani': 'auxiliary percussion', 
    'clarinet section': 'clarinet', 
    'flute section': 'flute', 
    'snare drum': 'drum set',
    'tenor saxophone': 'saxophone', 
    'trumpet section': 'brass', 
    'string section': 'strings', 
    'violin section': 'strings',
    'drum machine': 'drum set', 
    'french horn section': 'brass', 
    'bassoon': 'reeds',
    'brass section': 'brass', 
    'cello section': 'strings', 
    'tack piano': 'piano',
    'vibraphone': 'auxiliary percussion', 
    'tabla': 'auxiliary percussion', 
    'kick drum': 'drum set', 
    'scratches': 'synthesizer', 
    'bongo': 'auxiliary percussion', 
    'bass drum': 'drum set', 
    'doumbek': 'auxiliary percussion', 
    'alto saxophone': 'saxophone', 
    'gu': 'acoustic guitar', 
    'gong': 'auxiliary percussion', 
    'drum machine': 'drum set', 
    'darbuka': 'auxiliary percussion',
    'soprano saxophone': 'saxophone', 
    'guzheng': 'strings', 
    'horn section': 'brass', 
    'liuqin': 'strings', 
    'shaker':'auxiliary percussion',  
    'zhongruan':'strings', 
    'yangqin':'strings', 
    'dizi':'flute', 
    'oud':'strings', 
    'mandolin':'strings', 
    'erhu':'strings', 
    'harp': 'strings', 
    'bamboo flute': 'flute', 
    'trombone section': 'brass', 
    'viola section': 'strings',
    'baritone saxophone': 'saxophone',
    'female singer': 'voice', 
    'male singer': 'voice', 
    'vocalists': 'voice', 
    'male speaker': 'voice', 
    'female speaker': 'voice',
    'male rapper': 'voice', 
    'female rapper': 'voice', 
    'toms': 'drum set',
    'oboe': 'reeds',
    'chimes': 'auxiliary percussion',  
    'french horn': 'brass', 
    'lap steel guitar': 'strings', 
    'bass clarinet':  'reeds', 
    'clarinet': 'reeds', 
    'oboe': 'reeds', 
    'banjo': 'strings', 
    'electric piano': 'piano'

}

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
    metadata = create_dataset_entry(entry_path=str(entry_path), audio_format=audio_format, metadata_format='yaml', 
                                    label=label, start_time=start_time, end_time=end_time, sr=sr, effect_params=effect_params, **kwargs)

    # save metadata
    ir.utils.data.save_metadata_entry(metadata, entry_path_absolute)

    # save audio file
    au.io.write_audio_file(audio=audio, path_to_audio=entry_path_absolute, sample_rate=sr, 
                                 audio_format=audio_format, exist_ok=True)
                                


def get_classlist(metadata):
    return list(set(e['label'] for e in metadata))


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
            
            # EDIT: Remapping for audacity
            if label in remap:
                label = remap[label]
            # # continue if we don't want this class
            if label in unwanted_classes:
                continue

            # append the stem with it's corresponding info
            stem_info = dict(track_id=mtrack.track_id, stem_idx=stem.stem_idx, label=label, 
                            artist_id=mtrack.track_id.split('_')[0], path_to_audio=stem.audio_path, 
                            base_chunk_name=f'{mtrack.track_id}-{stem_id}-{label}')
            partition_list.append(stem_info)
    

    # get the unique set of classes for both partition
    classlists = {k: get_classlist(metadata) for k, metadata in partition_map.items()}
    print(classlists)

    # filter classes so we only have the intersection of the two sets :)
    filtered_classes = list(set(classlists['train']) & set(classlists['test']))

    print(f'list of filtered classes is {filtered_classes}')
    ans = input('this ok?')
    if ans.lower() in ('y', 'yes', 'ye'):
        pass
    elif ans.lower() in ('n', 'no'):
        exit()
    else:
        exit()

    # filter out the partition map!!!
    for partition_key, metadata in partition_map.items():
        partition_map[partition_key] = [e for e in metadata if e['label'] in  filtered_classes]

    print(f'created a partition map with the following classes:\n{filtered_classes}')
    print(f'number of tracks in train set: {len(partition_map["train"])}')
    print(f'number of tracks in test set: {len(partition_map["test"])}')

    return partition_map

def save_partition_map(partition_map, name):
    save_path = Path(ir.core.ASSETS_DIR) / f'{name}-partition_map'
    ir.utils.data.save_metadata_entry(partition_map, save_path)

def split_on_silence_and_save(partition_map, target_sr, dataset, audio_format):
    for partition_key, records in partition_map.items():

        def _split_and_save(entry):
            path_to_audio = entry['path_to_audio']
            base_chunk_name = entry['base_chunk_name']
            label = entry['label']

            audio = au.io.load_audio_file(path_to_audio, target_sr)

            audio = au.librosa_input_wrap(audio)
            audio = ir.utils.effects.trim_silence(audio, target_sr, min_silence_duration=0.5)
            audio = au.librosa_output_wrap(audio)

            if au.core._is_zero(audio):
                return

            duration = librosa.core.get_duration(au.librosa_input_wrap(audio))

            labels = [label]
            events = []
            for label in labels:
                events.append(dict(label=label, start_time=0.0, end_time=duration, duration=duration))

            # save audio
            # define a path for our dataset entry 
            entry_path = Path(dataset) / partition_key / 'foreground' / label / entry['track_id']
            entry_path_absolute = Path(ir.core.DATA_DIR) / entry_path

            # save audio file
            au.io.write_audio_file(audio=audio, path_to_audio=entry_path_absolute, sample_rate=target_sr, 
                                        audio_format=audio_format, exist_ok=True)

            # create an output record
            new_record = {}
            new_record['events'] = events
            new_record['path_to_audio'] = str(entry_path_absolute.with_suffix('.wav'))
            new_record['path_to_record'] = str(entry_path_absolute.with_suffix('.json'))
            new_record['duration'] = duration

            # save metadata
            ir.utils.data.save_metadata_entry(new_record, entry_path_absolute, 'json')

        records_pb = tqdm.tqdm(records)
        for record in records_pb:
            records_pb.set_description(f'processing {record["path_to_audio"].split("/")[-1]}')
            _split_and_save(record)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="medleydb-full")
    
    parser.add_argument('--audio_format', type=str, default='wav')
    parser.add_argument('--test_size', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=ir.RANDOM_SEED)
    parser.add_argument('--sample_rate', type=int, default=ir.SAMPLE_RATE)
    
    args = parser.parse_args()

    # make partition map, save to assets dir
    partition = medleydb_make_partition_map(test_size=args.test_size, random_seed=args.seed)
    save_partition_map(partition, args.dataset)
    split_on_silence_and_save(partition, args.sample_rate, args.dataset, args.audio_format)