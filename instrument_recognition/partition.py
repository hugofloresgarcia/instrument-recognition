import os

import numpy as np 
import medleydb as mdb
import soundfile as sf
import tqdm

from instrument_recognition.utils.audio import load_audio_file
from instrument_recognition.utils.effects import augment_from_array_to_array, trim_silence

RANDOM_SEED = 20
TEST_SIZE = 0.15

CHUNK_SIZE = 1

SR = 48000
HOP_SIZE = 0.25 
AUGMENT_TRAIN_SET = True
PATH_TO_OUTPUT = f'/home/hugo/data/mono_music_sed/mdb/AUDIO/'

unwanted_classes = ['Main System', 'claps', 'fx/processed sound', 'tuba', 'piccolo', 'cymbal',
                     'glockenspiel', 'tambourine', 'timpani', 'snare drum', 'clarinet section',
                      'flute section', 'tenor saxophone', 'trumpet section']

def _check_audio_types(audio):
    assert audio.ndim == 1, "audio must be mono"
    assert isinstance(audio, np.ndarray)

def get_audio_chunk(audio, sr, start_time, chunk_size):
    """ given MONO audio 1d array, get a chunk of that 
    array determined by start_time (in seconds), chunk_size (in seconds)
    pad with zeros if necessary
    """
    _check_audio_types(audio)

    duration = audio.shape[-1] / sr

    start_idx = int(start_time * sr)

    end_time = start_time + chunk_size 
    end_time = min([end_time, duration])

    end_idx = int(end_time * sr)

    chunked_audio = audio[start_idx:end_idx]

    if not len(audio) / sr == chunk_size * sr:
        chunked_audio = utils.audio.zero_pad(chunked_audio, sr * chunk_size)
    return chunked_audio

def save_windowed_audio_events(audio, sr, chunk_size, hop_size, base_chunk_name, 
                             label, path_to_output, metadata_extras, augment=True):
    """ this function will chunk a monophonic audio array 
    into chunks as determined by chunk_size and hop_size. 
    The output audio file will be saved to a foreground folder, scaper style, 
    under a subdirectory with a name determined by label. 
    Besides the output audio file, it will create a .yaml file with metadata
    for each corresponding audio file
    args:
        audio (np.ndarray): audio arrayshape (samples,)
        sr (int): sample rate
        chunk_size (float): chunk size, in seconds, to cut audio into
        hop_size (float): hop size, in seconds, to window audio
        base_chunk_name (str): base name for the output audio chunk file. The format for 
            the saved audio chunks is f"{base_chunk_name}-{start_time}"
        label (str): label for this audio example. 
        path_to_output (str): base path to save this example
        metadata_extras (dict): adds these extra entries to each metadata dict
    """
    _check_audio_types(audio)

    # determine how many chunk_size chunks we can get out of the audio array
    audio_len = len(audio)
    n_chunks = int(np.ceil(audio_len/(chunk_size*sr))) # use ceil because we can zero pad

    start_times = np.arange(0, n_chunks, hop_size)

    def save_chunk(start_time):
        # round start time bc of floating pt errors
        start_time = np.around(start_time, 4)

        # get current audio_chunk
        audio_chunk = get_audio_chunk(audio, sr, start_time, chunk_size)

        audio_chunk_name = base_chunk_name + f'/{start_time}.wav'
        audio_chunk_path = os.path.join(path_to_output, 
                                        f'{label}', 
                                        audio_chunk_name)
        
        if augment:
            audio_chunk, effect_params = augment_from_array_to_array(audio_chunk, sr)
        else:
            effect_params = []

        # make path for metadata
        chunk_metadata_path = audio_chunk_path.replace('.wav', '.yaml')
        
        os.makedirs(os.path.dirname(audio_chunk_path), exist_ok=True)

        # make a metadata entry
        entry = dict()
        entry.update(metadata_extras)
        entry.update(dict(
            path_to_dataset=path_to_output,
            path_to_metadata=chunk_metadata_path, 
            path_to_audio=audio_chunk_path,
            label=label,
            chunk_size=chunk_size, 
            start_time=float(start_time), 
            sr=sr, 
            effect_params=effect_params))

        # if either of these don't exist, create both
        if (not os.path.exists(chunk_metadata_path)) \
            or (not os.path.exists(audio_chunk_path)):
#             print(f'\t saving {chunk_metadata_path}', sep='', end='', flush=True)

            sf.write(audio_chunk_path, audio_chunk, sr, 'PCM_24')
            utils.data.save_dict_yaml(entry, chunk_metadata_path)
        else:
            pass
            # print(f'already found: {audio_chunk_path} and {chunk_metadata_path}')
        return entry

    for start_time in start_times:
        save_chunk(start_time)

    # with concurrent.futures.ThreadPoolExecutor(max_workers) as p:
    #     p.map(save_chunk, start_times)

if __name__ == "__main__":

    # the first thing to do is to partition the MDB track IDs and stem IDs into only the ones we will use. 
    mtrack_generator = mdb.load_all_multitracks(['V1', 'V2'])
    splits = mdb.utils.artist_conditional_split(test_size=TEST_SIZE, num_splits=1, 
                                                random_state=RANDOM_SEED)[0]
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
            stem_info = dict(track_id=mtrack.track_id, stem_idx=stem.stem_idx, 
                            label=label, 
                            artist_id=mtrack.track_id.split('_')[0], 
                            path_to_audio=stem.audio_path, 
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

    print(len(utils.data.get_classlist(partition_map['train'])))
    print(len(utils.data.get_classlist(partition_map['test'])))

    # now, save and do the magic
    for partition_key, metadata in partition_map.items():
        augment = True if partition_key == 'train' else False

        def split_and_augment(entry):
            # try:
            path_to_audio = entry['path_to_audio']
            base_chunk_name = entry['base_chunk_name']
            label = entry['label']
            output_path = os.path.join(PATH_TO_OUTPUT, partition_key)

            audio = load_audio_file(path_to_audio, SR)
            # trim silence
            audio = trim_silence(audio, SR, min_silence_duration=0.3)

            save_windowed_audio_events(audio=audio, sr=SR, chunk_size=CHUNK_SIZE, 
                                    hop_size=HOP_SIZE, base_chunk_name=base_chunk_name, 
                                    label=label, path_to_output=output_path, 
                                    metadata_extras=entry, augment=augment)

        # DO IT IN PARALLEL
        tqdm.contrib.concurrent.process_map(split_and_augment, metadata)