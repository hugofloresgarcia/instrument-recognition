""" soundscape.py - offline data augmentation and synthetic soundscape construction
"""
from pathlib import Path
import os

import scaper
import librosa
import numpy as np
import tqdm

import instrument_recognition as ir

# TODO: need to add partition scripts for mdb-synthetic-mono, mdb-synthethic-poly, openmic-2018
# should save to /data/
# TODO: the synthetic mdb should consist of 10s sequences and stuff. you should make it using scaper
# and add all required effects to the train partition 
# TODO: make sure to save partition maps in /instrument_recognition/assets/
# TODO: also, incoroporate your new fly packages, torchopenl3 and audio utils

def get_randn(mu, std, min=None, max=None):
    """ get a random float, sampled from a normal 
    distribution with mu and std, clipped between min and max
    """
    randn = mu + std * np.random.randn(1)
    return float(randn.clip(min, max))

def choose(collection_like):
    """ given a collection of items, draw one
    (sampled from a uniform distribution)
    """
    min_i = 0
    max_i = len(collection_like)
    idx = int(np.floor(np.random.uniform(min_i, max_i)))
    return collection_like[idx]

def uniform(min, max):
    return choose(np.linspace(min, max, 10000))

def sample_from_distribution_spec(spec):
    if spec['dist'] == 'uniform':
        return uniform(spec['min'], spec['max'])
    else:
        raise NotImplementedError

def get_audio_duration(path_to_audio):
    return librosa.core.get_duration(filename=path_to_audio)

def main():
    data_dir = Path(ir.core.DATA_DIR)

    dataset = 'medleydb'
    name = 'mdb-synthetic-mono'

    num_soundscapes = 1000
    ref_db = -45
    duration = 10.0
    monophonic = True

    min_events = 1
    max_events = 9

    seed = ir.core.RANDOM_SEED

    # lets go
    partitions = os.listdir(data_dir/dataset)
    for partition in partitions:
        fg_path = data_dir/dataset/partition/'foreground'
        bg_path = data_dir/dataset/partition/'background'
        soundscapes_path = data_dir/dataset/partition/name

        total_len_s = sum([get_audio_duration(str(fg_path/subd/d)) for subd in os.listdir(fg_path) for d in os.listdir(fg_path/subd) if Path(d).suffix == '.wav'])
        num_soundscapes = int(int(total_len_s) // duration * 3)
        print(f'generating {num_soundscapes} soundscapes')

        event_time_spec = dict(dist='truncnorm', mean=5.0, std=2.0, max=10.0)
        snr_spec = dict(dist='uniform', min=6, max=15)
        pitch_spec = dict(dist='uniform', min=-3.0, max=3.0)
        time_spec = dict(dist='uniform', min=0.8, max=1.2)
        source_time_spec = dict(dist='uniform', min=0.0, max=None)
        event_duration_spec = dict(dist='uniform', min=2.5, max= 10 / time_spec['max'])
        

        sc = scaper.Scaper(duration,  fg_path=str(fg_path), 
                        bg_path=str(bg_path), random_state=seed)
        sc.ref_db = ref_db
        
        pbar = tqdm.tqdm(range(num_soundscapes))
        for idx in pbar:
            sc.reset_bg_event_spec()
            sc.reset_fg_event_spec()

            # sc.add_background(label=('choose', []),
            #             source_file=('choose', []),
            #             source_time=('const', 0))

            # add random number of foreground events
            if monophonic:
                playhead = 0.0
                while playhead < (duration - event_duration_spec['min']):
                    label = choose(os.listdir(fg_path))
                    source_file = choose([str(fg_path/label/f) for f in os.listdir(fg_path/label) if Path(f).suffix == '.wav'])
                    event_duration = min(sample_from_distribution_spec(event_duration_spec), duration - playhead)
                    event_duration = event_duration / time_spec['max']
                    source_time = uniform(0, get_audio_duration(source_file) - event_duration)
                    
                    sc.add_event(label=('const', label),
                        source_file=('const', source_file),
                        source_time=('const', source_time),
                        event_time=('const', playhead),
                        event_duration=('const', event_duration),
                        snr=tuple(snr_spec.values()),
                        pitch_shift=tuple(pitch_spec.values()),
                        time_stretch=tuple(time_spec.values()))
                    
                    playhead += event_duration
                    playhead += get_randn(0.5, 0.2, min=0, max=3)
            else:
                n_events = np.random.randint(min_events, max_events+1)
                for _ in range(n_events):
                    label = choose(os.listdir(fg_path))
                    source_file = choose([f for f in os.listdir(fg_path/label) if Path(f).suffix == '.wav'])
                    event_duration = sample_from_distribution_spec(event_duration_spec) * time_spec['max']
                    source_time = uniform(0, get_audio_duration(source_file))

                    sc.add_event(label=('const', label),
                        source_file=('const', source_file),
                        source_time=('const', source_time),
                        event_time=tuple(event_time_spec.values()),
                        event_duration=tuple(event_duration_spec.values()),
                        snr=tuple(snr_spec.values()),
                        pitch_shift=tuple(pitch_spec.values()),
                        time_stretch=tuple(time_spec.values()))

            # generate
            audiofile = soundscapes_path / f"soundscape_{idx}.wav"
            jamsfile = soundscapes_path / f"soundscape_{idx}.jams"
            txtfile = soundscapes_path / f"soundscape_{idx}.txt"
            os.makedirs(soundscapes_path, exist_ok=True)


            sc.generate(str(audiofile), str(jamsfile),
                        allow_repeated_label=True,
                        allow_repeated_source=True,
                        disable_sox_warnings=True,
                        no_audio=False,
                        txt_path=str(txtfile), fix_clipping=True)


if __name__ == "__main__":
    main()