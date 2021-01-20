import os
from pathlib import Path
import glob

import logging
import numpy as np 
import jams

import instrument_recognition as ir

def create_records_for_scaper_dataset(name: str):
    dataset_root = ir.DATA_DIR / name 
    partitions = ir.utils.data.list_subdir(dataset_root)
    print(f'found partitions {partitions}')
    
    for partition in partitions:
        data_path = dataset_root / partition 
        all_jams_files = glob.glob(str(data_path / '*.jams'), recursive=True)
        all_jams = [jams.load(p) for p in all_jams_files]

        records = jams_to_records(all_jams)

        for r in records:
            f = r['path_to_jams']
            f = str(Path(f).with_suffix('.json'))
            r['path_to_record'] = f
            print(f'saving record {f}')
            ir.utils.data.save_metadata_entry(r, f, format='json')

def jams_to_records(jam_list):
    records = []
    for jam in jam_list:
        entry = {}
        entry['path_to_audio'] = get_path_to_audio(jam)
        entry['duration'] = jam.file_metadata.duration
        entry['events'] = get_all_events(jam)
        entry['path_to_jams'] = get_path_to_jams_file(jam)
        records.append(entry)

        print(f'processed record {entry["path_to_audio"]}')
    return records

def get_scaper_observations(jam):
    return jam.search(namespace='scaper')[0]['data']

def get_path_to_audio(jam):
    return jam.annotations[0].sandbox['scaper']['soundscape_audio_path']

def get_path_to_jams_file(jam):
     return jam.annotations[0].sandbox['scaper']['jams_path']

def get_simple_annotation(obv):
    """ for a jams Observation, returns a dict
    with label, start_time and end_time
    """
    return dict(label=obv.value['label'],
        start_time=obv.value['event_time'],
        end_time=obv.value['event_time']+obv.value['event_duration'], 
        duration=obv.value['event_duration']) 

def get_all_events(jam):
    # get obs
    obvs = get_scaper_observations(jam)
    events = [get_simple_annotation(obv) for obv in obvs]
    return events

def quantize_ceil(value, numerator, denominator, num_decimals=4, floor_threshold=0.10):
    ratio = (numerator/denominator)
    quant = round((value // ratio + 1) * ratio, num_decimals)
    if quant - value > ((1 - floor_threshold) * ratio):
        return quant - ratio
    else: 
        return quant

def quantize_floor(value, numerator, denominator, num_decimals=4, ceil_threshold=0.9):
    ratio = (numerator/denominator)
    quant = round(value // ratio * ratio, num_decimals)
    if value - quant > (ceil_threshold * ratio):
        return quant + ratio
    else:
        return quant

def get_one_hot_matrix(jam, classlist: list, resolution: float = 1.0):
    # get duration from file metadata
    duration = jam.file_metadata.duration
    obvs = get_scaper_observations(jam)

    # determine the number of bins in the time axis
    assert duration % resolution == 0, \
        f'resolution {resolution} is not divisible by audio duration: {duration}'
    num_time_bins = int(duration // resolution)

    # make an empty matrix shape (time, classes)
    one_hot = np.zeros((num_time_bins, len(classlist)))
    time_axis = list(np.arange(0.0, duration, resolution))

    # get the indices for each label
    for obv in obvs:
        ann = obv.value

        start_time = ann['event_time']
        end_time = start_time + ann['event_duration']

        start_idx = time_axis.index(quantize_floor(start_time, duration, duration / resolution))

        ceil = quantize_ceil(end_time, duration, duration / resolution)
        ceil = ceil if ceil != duration else time_axis[-1]
        end_idx = time_axis.index(ceil)

        label_idx = classlist.index(ann['label'])

        # now, index
        one_hot[start_idx:end_idx, label_idx] = 1
    
    return one_hot

def test_quantize():
    assert quantize_floor(3.901, 10, 10, 4, 0.9) == 4
    assert quantize_ceil(3.901, 10, 10, 4, 0.1) == 4

    assert quantize_floor(3.090, 10, 10, 4, 0.9) == 3
    assert quantize_ceil(3.09, 10, 10, 4, 0.1) == 3

    assert quantize_floor(0.001, 10, 40, 4, 0.9) == 0
    assert quantize_ceil(0.001, 10, 40, 4, 0.1) == 0

    assert quantize_floor(0.2301, 10, 40, 4, 0.9) == 0.25
    assert quantize_ceil(0.2301, 10, 40, 4, 0.1) == 0.25

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True, nargs='+')

    args = parser.parse_args()

    for name in args.name:
        create_records_for_scaper_dataset(name)