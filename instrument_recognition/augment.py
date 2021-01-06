from pathlib import Path
import os 

import tqdm

import audio_utils as au
import instrument_recognition as ir

def augment_and_save_record(entry: dict, source_dir: str, dest_dir: str):
    source_dir, dest_dir = Path(source_dir), Path(dest_dir)

    # make a new entry
    output_entry = dict(entry)
    # create output paths 
    output_entry['path_to_record'] = str(dest_dir /  Path(entry['path_to_record']).relative_to(source_dir))
    output_entry['path_to_audio'] = str(dest_dir / Path(entry['path_to_audio']).relative_to(source_dir))
    output_entry['path_to_effect_params'] = str(dest_dir / f'{str(Path(entry["path_to_record"]).relative_to(source_dir).stem)}-effect_params.json')
    os.makedirs(Path(output_entry['path_to_record']).parent, exist_ok=True)

    audio = au.io.load_audio_file(entry['path_to_audio'], sample_rate=ir.SAMPLE_RATE)
    audio, effect_params = ir.utils.effects.augment_from_array_to_array(audio, ir.SAMPLE_RATE)
    au.io.write_audio_file(audio, output_entry['path_to_audio'], sample_rate=ir.SAMPLE_RATE, audio_format='wav')
    
    # save effect params and new record
    ir.utils.data.save_metadata_entry(effect_params, output_entry['path_to_effect_params'], format='json')
    ir.utils.data.save_metadata_entry(output_entry, output_entry['path_to_record'], format='json')

def augment_dataset(name: str, partition: str):
    source_dir = ir.DATA_DIR / name / partition
    dest_dir = ir.DATA_DIR / name / f'{partition}-augmented'

    # get all dem metadata entries
    records = ir.utils.data.glob_all_metadata_entries(source_dir)
    source_dirs = [source_dir for r in range(len(records))]
    dest_dirs = [dest_dir for r in range(len(records))]

    tqdm.contrib.concurrent.process_map(augment_and_save_record, records, source_dirs, dest_dirs)
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--partition', type=str, default='train')

    args = parser.parse_args()

    augment_dataset(args.name, args.partition)
