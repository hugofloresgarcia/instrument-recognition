import numpy as np
import torch
import sox
import librosa

def get_randn(mu, std, min=None, max=None):
    randn = mu + std * np.random.randn(1)
    return float(randn.clip(min, max))

def random_transform(audio, sr, transforms):
    tfm = sox.Transformer()
   
    if 'compand' in transforms:
        tfm.compand(attack_time=get_randn(0.5, 0.1, 0.1), 
                    decay_time=get_randn(0.8, 0.3, 0.1))
    if 'overdrive' in transforms:
        tfm.overdrive(get_randn(5, 15, 0, 30))
    if 'pitch' in transforms:
        tfm.pitch(get_randn(0, 6, -12, 12))
    if 'reverb' in transforms:
        tfm.reverb(reverberance=get_randn(25, 25, 0, 50), 
                   room_scale=get_randn(50, 25,  0, 100))
    # if 'stretch' in transforms:
    #     factor = 1 + 0.125 * np.random.randn(1)
    #     tfm.tempo(get_randn(1, 0.125, 0.5, 2))
    if 'tremolo' in transforms:
        speed = 0 + 2 * np.random.randn(1)
        tfm.tremolo(get_randn(0, 2, 0.1))

    # keep backup of old audio to keep track of device
    old_audio = audio
    # transform to sox
    audio = audio.squeeze(1).t().cpu().detach().numpy()
    audio = tfm.build_array(input_array=audio, sample_rate_in=sr)
    # transform back from sox
    audio = torch.from_numpy(audio).type_as(old_audio)
    audio = audio.t().unsqueeze(1)

    assert audio.shape == old_audio.shape
    return audio
