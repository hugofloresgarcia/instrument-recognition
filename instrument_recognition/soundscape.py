""" soundscape.py - offline data augmentation and synthetic soundscape construction
"""
from pathlib import Path
import os
import glob

import scaper
import librosa
import numpy as np
import tqdm

import instrument_recognition as ir

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

# NOTE: min_events and max_events will only do something if monophonic = False. Otherwise, 
# the number of events is determined by the length of the previously sampled events. 
def make_soundscapes(name: str, dataset: str = 'medleydb', monophonic: bool = True, 
                     min_events: int = 1, max_events: int = 9, duration: float = 10.0):
    ref_db = -45
    data_dir = Path(ir.core.DATA_DIR)
    seed = ir.core.RANDOM_SEED

    # lets go
    partitions = ir.utils.data.list_subdir(data_dir/dataset)
    for partition in partitions:
        fg_path = data_dir/dataset/partition/'foreground'
        bg_path = data_dir/dataset/partition/'background'
        if not bg_path.exists():
            os.makedirs(bg_path)
            
        soundscapes_path = data_dir/name/partition

        total_len_s = sum([get_audio_duration(str(fg_path/subd/d)) for subd in os.listdir(fg_path) for d in os.listdir(fg_path/subd) if Path(d).suffix == '.wav'])
        num_soundscapes = int(int(total_len_s) // duration * 3)
        print(f'generating {num_soundscapes} soundscapes')

        event_time_spec = dict(dist='truncnorm', mean=5.0, std=2.0, min=0.0, max=10.0)
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
                # ALWAYS put something betweeen seconds (0, 1)
                playhead = get_randn(0.3, 0.2, min=0, max=1)
                while playhead < (duration - event_duration_spec['min']):
                    # random choice for label and source file
                    label = choose(os.listdir(fg_path))
                    source_file = choose([str(fg_path/label/f) for f in os.listdir(fg_path/label) if Path(f).suffix == '.wav'])

                    # the event can last as long as it wants, as long as it doesn't run over the soundscape duration
                    event_duration = min(sample_from_distribution_spec(event_duration_spec), duration - playhead)
                    event_duration = event_duration / time_spec['max'] # take the worst case time stretch into account

                    # pick anywhere in the source file 
                    source_time = uniform(0, get_audio_duration(source_file) - event_duration)
                    
                    sc.add_event(label=('const', label),
                        source_file=('const', source_file),
                        source_time=('const', source_time),
                        event_time=('const', playhead),
                        event_duration=('const', event_duration),
                        snr=tuple(snr_spec.values()),
                        pitch_shift=tuple(pitch_spec.values()),
                        time_stretch=tuple(time_spec.values()))
                    
                    # move the playhead to the very end of the event
                    playhead += event_duration
                    # round the playhead to the nearest second (to only have 1 event per second frame)
                    playhead = np.ceil(playhead)
                    # add a random offset
                    playhead += get_randn(0.3, 0.2, min=0, max=1)
            else:
                n_events = np.random.randint(min_events, max_events+1)
                for _ in range(n_events):
                    label = choose(os.listdir(fg_path))
                    source_file = choose([str(fg_path/label/f) for f in os.listdir(fg_path/label) if Path(f).suffix == '.wav'])
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

def records2scaper(name: str, dataset: str):
    data_dir = ir.DATA_DIR

    # lets go
    partitions = ir.utils.data.list_subdir(data_dir/dataset)

    for partition in partitions:
        fg_path = data_dir/name/partition/'foreground'
        bg_path = data_dir/name/partition/'background'
        os.makedirs(fg_path, exist_ok=False)
        os.makedirs(bg_path, exist_ok=False)

        # read all metadata
        root_dir = data_dir/dataset/partition
        records = ir.utils.data.glob_all_metadata_entries(root_dir=root_dir)
        for record in records:
            assert len(record['events']) == 1
            label = record['events'][0]['label']
            filename = Path(record['path_to_audio']).relative_to(root_dir)
            assert Path(record['path_to_audio']).exists()
            os.makedirs(fg_path/label/filename.parent, exist_ok=True)
            os.symlink(src=record['path_to_audio'], dst=fg_path/label/filename)

if __name__ == "__main__":
    import argparse
    from instrument_recognition import utils

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--records2scaper', type=utils.str2bool, default=False)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='medleydb')
    parser.add_argument('--monophonic', type=utils.str2bool, default=True)
    parser.add_argument('--min_events', type=int, default=1)
    parser.add_argument('--max_events', type=int, default=9)
    parser.add_argument('--duration', type=float, default=10.0)

    args = parser.parse_args()

    if args.records2scaper:
        records2scaper(args.name, args.dataset)
    else:
        make_soundscapes(name=args.name, dataset=args.dataset, monophonic=args.monophonic, 
                     min_events=args.min_events, max_events=args.max_events, duration=args.duration)