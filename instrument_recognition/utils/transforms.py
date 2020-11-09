import numpy as np
import torch
import sox
import librosa

from instrument_recognition.utils.audio_utils import zero_pad

def get_randn(mu, std, min=None, max=None):
    randn = mu + std * np.random.randn(1)
    return float(randn.clip(min, max))

def random_transform(audio, sr, transforms):
    tfm = sox.Transformer()
   
    if 'compand' in transforms:
        tfm.compand(attack_time=get_randn(0.5, 0.1, min=0.05, max=0.3), 
                    decay_time=get_randn(0.8, 0.3, min=0.5, max=2))
    if 'overdrive' in transforms:
        tfm.overdrive(get_randn(5, 5, 0, 30))
    if 'pitch' in transforms:
        tfm.pitch(get_randn(0, 0.5, -3, 3), quick=False)
    if 'reverb' in transforms:
        tfm.reverb(reverberance=get_randn(25, 15, 0, 50), 
                   room_scale=get_randn(50, 25,  0, 100))
    if 'stretch' in transforms:
        tfm.stretch(get_randn(1, 0.0125, 0.875, 1.125))
    if 'tremolo' in transforms:
        speed = 0 + 2 * np.random.randn(1)
        tfm.tremolo(get_randn(0, 2, 0.1))

    # keep backup of old audio to keep track of device
    old_audio = audio
    # transform to sox
    audio = audio.squeeze(1).t().cpu().detach().numpy()
    audio = tfm.build_array(input_array=audio, sample_rate_in=sr)
    # transform back from sox
    # transpose to (channels, time)
    audio = audio.T
    # zero pad and cut off samples past the old audio's shape
    audio = zero_pad(audio, old_audio.shape[-1])
    if audio.ndim > 1:
        audio = audio[:, 0:old_audio.shape[-1]]
        audio = torch.from_numpy(audio).type_as(old_audio)
        # go back to old dims
        audio = audio.unsqueeze(1)
    else:
        audio = audio[0:old_audio.shape[-1]]
        audio = torch.from_numpy(audio).type_as(old_audio)
        # go back to old dims
        audio = audio.unsqueeze(0).unsqueeze(0)
    
    
    assert audio.shape == old_audio.shape, f'{audio.shape}-{old_audio.shape}'
    return audio
