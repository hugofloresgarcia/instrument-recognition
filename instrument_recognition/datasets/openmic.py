from pathlib import Path
import shutil
import os

import pandas as pd
import tqdm
import librosa

import instrument_recognition as ir

def openmic2records(output_name: 'openmic'):
    source_dir = ir.DATA_DIR / 'openmic-2018-source'
    dest_dir = ir.DATA_DIR / output_name

    partitions = ('train', 'test')

    all_metadata = pd.read_csv(source_dir / 'openmic-2018-metadata.csv').to_dict('records')
    aggregated_labels = pd.read_csv(source_dir / 'openmic-2018-aggregated-labels.csv')

    for partition in partitions:
        partition_path = source_dir / 'partitions' / f'split01_{partition}.csv'

        split = pd.read_csv(partition_path, header=None, squeeze=True)
        split = set(split)

        records = [e for e in all_metadata if e['sample_key'] in split]

        # makeshift strong labels out of weak labels
        pbar = tqdm.tqdm(records)
        for record in pbar:
            labels = aggregated_labels.query(f'sample_key == "{record["sample_key"]}"')['instrument']
            labels = list(labels)
            
            events = []
            for label in labels:
                # NOTE: hardcoding with things I already know about the dataset
                events.append(dict(label=label, start_time=0.0, end_time=10.0, duration=10.0))


            # copy audio?
            src_path_to_audio = source_dir / 'audio' / record['sample_key'][0:3] / f'{record["sample_key"]}.ogg'
            dest_path_to_audio = dest_dir / partition / f'{record["sample_key"]}.ogg'
            os.makedirs(dest_path_to_audio.parent, exist_ok=True)
            shutil.copy(src_path_to_audio, dest_path_to_audio)

            # create an output record
            new_record = {}
            new_record['events'] = events
            new_record['path_to_audio'] = str(dest_path_to_audio)
            new_record['path_to_record'] = str(dest_dir / partition / f'{record["sample_key"]}.json')
            new_record['duration'] = librosa.core.get_duration(filename=str(dest_path_to_audio))

            # save output record
            ir.utils.data.save_metadata_entry(new_record, path=new_record['path_to_record'], 
                                         format='json')

if __name__ == "__main__":
    openmic2records('openmic')


