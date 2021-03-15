from pathlib import Path
import os 

import tqdm
import numpy as np

import audio_utils as au
import instrument_recognition as ir

def augment_and_save_record(entry: dict, source_dir: str, dest_dir: str):
    source_dir, dest_dir = Path(source_dir), Path(dest_dir)

    # make a new entry
    output_entry = dict(entry)

    # create output paths 
    output_entry['path_to_record'] = str(dest_dir /  Path(entry['path_to_record']).relative_to(source_dir))
    output_entry['path_to_audio'] = str(dest_dir / Path(entry['path_to_audio']).relative_to(source_dir))
    output_entry['path_to_effect_params'] = str(dest_dir / f'{str(Path(entry["path_to_record"]).relative_to(source_dir).stem)}-effect_params.yaml')
    os.makedirs(Path(output_entry['path_to_record']).parent, exist_ok=True)

    audio = au.io.load_audio_file(entry['path_to_audio'], sample_rate=ir.SAMPLE_RATE)
    # audio, effect_params = ir.utils.effects.augment_from_array_to_array(audio, ir.SAMPLE_RATE)

    num_samples = audio.shape[-1]
    index = 0
    future = np.random.randint(500, num_samples // 4)
    # breakpoint()
    while future < num_samples:
        # print(audio.shape)
        # exit()
        # print(audio[:, index:future].shape)
        audio[:, index:future], effect_params = ir.utils.effects.augment_from_array_to_array(audio[:, index:future], ir.SAMPLE_RATE)
        index = future
        new_min = min(index+ir.SAMPLE_RATE//2, num_samples)
        future = np.random.randint(new_min, num_samples+1)


    # print(audio.shape)
    au.io.write_audio_file(audio, output_entry['path_to_audio'], sample_rate=ir.SAMPLE_RATE, audio_format='wav')
    
    # save effect params and new record
    ir.utils.data.save_metadata_entry(effect_params, output_entry['path_to_effect_params'], format='yaml')
    ir.utils.data.save_metadata_entry(output_entry, output_entry['path_to_record'], format='json')

from tqdm.auto import tqdm as tqdm_auto
from copy import deepcopy
from os import cpu_count
import sys
def _executor_map(PoolExecutor, fn, *iterables, **tqdm_kwargs):
    """
    Implementation of `thread_map` and `process_map`.

    Parameters
    ----------
    tqdm_class  : [default: tqdm.auto.tqdm].
    """
    kwargs = deepcopy(tqdm_kwargs)
    if "total" not in kwargs:
        kwargs["total"] = len(iterables[0])
    tqdm_class = kwargs.pop("tqdm_class", tqdm_auto)
    max_workers = kwargs.pop("max_workers", min(32, cpu_count() + 4))
    pool_kwargs = dict(max_workers=max_workers)
    if sys.version_info[:2] >= (3, 7):
        # share lock in case workers are already using `tqdm`
        pool_kwargs.update(
            initializer=tqdm_class.set_lock, initargs=(tqdm_class.get_lock(),))
    with PoolExecutor(**pool_kwargs) as ex:
        return list(tqdm_class(ex.map(fn, *iterables), **kwargs))

def process_map(fn, *iterables, **tqdm_kwargs):
    """
    Equivalent of `list(map(fn, *iterables))`
    driven by `concurrent.futures.ProcessPoolExecutor`.

    Parameters
    ----------
    tqdm_class  : [default: tqdm.auto.tqdm].
    """
    from concurrent.futures import ProcessPoolExecutor
    return _executor_map(ProcessPoolExecutor, fn, *iterables, **tqdm_kwargs)


def augment_dataset(name: str, partition: str, num_folds: int):
    for fold in range(num_folds):
        source_dir = ir.DATA_DIR / name / partition
        dest_dir = ir.DATA_DIR / name / f'{partition}-augmented' / f'fold-{fold}'

        if dest_dir.exists():
            continue

        # get all dem metadata entries
        records = ir.utils.data.glob_all_metadata_entries(source_dir)
        source_dirs = [source_dir for r in range(len(records))]
        dest_dirs = [dest_dir for r in range(len(records))]
        
        process_map(augment_and_save_record, records, source_dirs, dest_dirs)
        # for r, s, d in zip(records, source_dirs, dest_dirs):
        #     augment_and_save_record(r, s, d)
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--partition', type=str, default='train')
    parser.add_argument('--num_folds', type=int, default=2)

    args = parser.parse_args()

    augment_dataset(args.name, args.partition, args.num_folds)
