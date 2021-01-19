from pathlib import Path
import shutil
import os

import pandas as pd
import tqdm
import librosa

import instrument_recognition as ir
import audio_utils as au

DURATION = 3.0

def mdbsolo2records(output_name: 'mdb-solos'):
    source_dir = ir.DATA_DIR / 'mdb-solos-source'
    dest_dir = ir.DATA_DIR / output_name

    partitions = ('train', 'validation', 'test')

    all_metadata = pd.read_csv(source_dir / 'metadata.csv').to_dict('records')

    for partition in partitions:
        partition_alias = 'training' if partition == 'train' else partition
        records = [record for record in all_metadata if record['subset'] == partition_alias]
        # makeshift strong labels out of weak labels
        pbar = tqdm.tqdm(records)
        for record in pbar:
            label = record['instrument']
            label_id = record['instrument_id']
            uuid = record['uuid4']

            audio_filename = f'Medley-solos-DB_{partition_alias}-{label_id}_{uuid}.wav'
            record_filename = Path(audio_filename).with_suffix('.json')
            path_to_audio = source_dir / audio_filename

            duration = librosa.core.get_duration(filename=str(path_to_audio))

            # NOTE: only one label per file, but keeping the structure from polyphonic datasets
            labels = [label]
            events = []
            for label in labels:
                events.append(dict(label=label, start_time=0.0, end_time=duration, duration=duration))

            os.makedirs(path_to_audio.parent, exist_ok=True)

            # create an output record
            new_record = {}
            new_record['events'] = events
            new_record['path_to_audio'] = str(path_to_audio)
            new_record['path_to_record'] = str(dest_dir / partition / record_filename)
            new_record['duration'] = DURATION

            # save output record
            ir.utils.data.save_metadata_entry(new_record, path=new_record['path_to_record'], format='json')

if __name__ == "__main__":
    mdbsolo2records('mdb-solos')


