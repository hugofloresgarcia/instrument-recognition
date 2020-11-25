import scaper
import os
import philharmonia_dataset
import numpy as np

def make_soundscapes():
    generate_soundscape(foreground_folder='../CHONK/data/philharmonia-synthesized/foreground', 
                        background_folder='../CHONK/data/philharmonia-synthesized/background')

def generate_soundscape(foreground_folder, background_folder, 
                        seed=42, parent_dir='../CHONK/data/philharmonia-synthesized/soundscapes/', 
                        soundscape_duration=10, sample_rate=48000, ref_db=-40, 
                        num_soundscapes=750, num_events_mean=25, num_events_std=5):

    np.random.seed(seed)

    sc = scaper.Scaper(soundscape_duration, foreground_folder, background_folder, 
                    random_state=seed)
    sc.ref_db = ref_db

    for soundscape_idx in range(num_soundscapes):
        sc.reset_bg_event_spec()
        sc.reset_fg_event_spec()

        # add background
        sc.add_background(label=('choose', []), 
                        source_file=('choose', []), 
                        source_time=('const', 0))

        num_events = int(max(0, num_events_mean + np.random.standard_normal() * num_events_std))
        for event in range(num_events):
            sc.add_event(label=('choose', []), 
                        source_file=('choose', []),
                        source_time=('const', 0),
                        event_time=('uniform', 0, soundscape_duration),
                        event_duration=('truncnorm', 1, 0.5, 0.25, 5), 
                        snr=('normal', 6, 10), 
                        pitch_shift=('uniform', -3, 3), 
                        time_stretch=('uniform', 0.75, 1.25))

        
        audiofile = os.path.join(parent_dir, 'audio', f'soundscape_{soundscape_idx}.wav')
        jamsfile = os.path.join(parent_dir, 'jamsfiles', f'soundscape_{soundscape_idx}.jams')
        txtfile = os.path.join(parent_dir, 'labels', f'soundscape_{soundscape_idx}.txt')

        
        print(f'generating: {audiofile}')
        sc.generate(audiofile, jamsfile, 
                    allow_repeated_label=True, 
                    allow_repeated_source=True,
                    reverb=0.1, 
                    disable_sox_warnings=True, 
                    no_audio=False,
                    txt_path=txtfile)             

def transform_background_audio(list_of_audio_paths, list_of_output_paths,
                            sample_rate=16000, gain_db=-10, lowpass_fc=4000, 
                            output_format='mp3'):
    """
    this is my way of synthesizing audio into making it more background-ey

    signal chain:
        downmix -> downsample -> lowpass -> compress 
                                    -> gain reduction -> convert to mp3

    the input will be fully mixed orchestra recordings
    """
    import sox
    tfm = sox.Transformer()
    tfm.channels(1)
    tfm.convert(samplerate=sample_rate)
    tfm.lowpass(lowpass_fc)
    tfm.compand()
    tfm.gain(gain_db=gain_db)
    tfm.set_output_format(file_type=output_format)

    for audio_path, output_path in zip(list_of_audio_paths, list_of_output_paths):
        print(f'processing: {audio_path}')
        tfm.build(audio_path, output_path)
    
def make_background_soundscapes():
    input_parent_dir = os.path.join(os.getcwd(), 'data', 'philharmonia-synthesized', 'background-src')
    output_parent_dir = os.path.join(os.getcwd(), 'data', 'philharmonia-synthesized', 'background')

    # let's get the lists of paths for the audio that we're going to process
    list_of_audio_paths = []
    list_of_output_paths = []

    for root, dirs, files in os.walk(input_parent_dir):
        for f in files:
            # add all audio files to list of audio paths
            if f.split('.')[-1] in ('flac', 'mp3', 'wav'):
                # to get an output path, replace the input parent dir 
                # with the output parent dir
                audio_path = os.path.join(root, f)
                output_path = audio_path.replace('flac', 'mp3').replace(input_parent_dir, output_parent_dir)

                # print(output_path)
                list_of_audio_paths.append(audio_path)
                list_of_output_paths.append(output_path)

    # # for debug
    # list_of_audio_paths = ['/Users/hugoffg/Documents/lab/music_sed/CHONK/data/philharmonia-synthesized/background-src/01.Carmina Burana O Fortruna.flac']
    # list_of_output_paths = ['/Users/hugoffg/Documents/lab/music_sed/CHONK/data/philharmonia-synthesized/background-src/01.Carmina Burana O Fortruna.mp3']

    # done collecting! now, do all the hard work
    transform_background_audio(list_of_audio_paths, list_of_output_paths)
                

if __name__ == "__main__":
    make_soundscapes()          

