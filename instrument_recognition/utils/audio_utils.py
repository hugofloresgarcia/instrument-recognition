import numpy as np
import librosa
import resampy
import torch
import torchaudio
import pydub

def _check_audio_types(audio):
    if not isinstance(audio, torch.Tensor) and not isinstance(audio, np.ndarray):
        raise Exception(f'got {type(audio)} but was expeting torch tensor or np array')

def _check_audio_shapes(audio):
    if not (audio.ndim == 2 or audio.ndim == 3):
        raise Exception(f'got audio ndim:{audio.ndim} but was expeting shape (channels, time or (batch, channels, time)')

def _test_max_functions(fn1, fn2):
    # make a couple of test_arrays
    n_tests = 100
    test_arrays = torch.randn((n_tests, 64, 1, 128, 199)).numpy()
   
    for test in test_arrays:
        max_axis = tuple(range(test.ndim)[1:])
        ans1 = fn1(test, axis=max_axis, keepdims=True)
        ans2 = fn2(test, axis=max_axis, keepdims=True)

    result = np.allclose(ans1, ans2, rtol=1e-5)
    assert result
    print(f'test result: {result}')
    return resut

def get_stft_filterbank(N, window='hann'):
    """
    returns real and imaginary kernels for a dft filterbank
    an assymetric window is used (see librosa.filters.get_window)

    params
    ------
    N (int): number of dft components

    returns
    -------
    f_real (np.ndarray): real filterbank with shape (N, N//2+1)
    f_imag (np.ndarray): imag filterbank with shape (N, N//2+1)
    """
    K = N // 2 + 1 
    # discrete time axis 
    n = np.arange(N)

    w_k = np.arange(K) * 2  * np.pi / float(N)
    f_real = np.cos(w_k.reshape(-1, 1) * n.reshape(1, -1))
    f_imag = np.sin(w_k.reshape(-1, 1) * n.reshape(1, -1))

    window = librosa.filters.get_window(window, N, fftbins=True)
    window = window.reshape((1, -1))

    f_real = np.multiply(f_real, window)
    f_imag = np.multiply(f_imag, window)

    return f_real, f_imag

def amplitude_to_db(
        x: torch.Tensor, amin: float = 1e-10, 
        dynamic_range: float = 80.0, 
        to_torchscript: bool = True):
    """
    per kapre's amplitude to db
    for torchscript compiling reasons
        x must be shape (batch, channels, height, width)
    update: use to_torchscript flag as false to use different array shapes
    """
    
    # apply log transformation (amplitude to db)
    amin = torch.full_like(x, 1e-10).float()
    log10 = torch.tensor(np.log(10)).float()
    x = 10 * torch.log(torch.max(x, amin)) / log10
        
    xmax, v = x.max(dim=1, keepdim=True)
    xmax, v = xmax.max(dim=2, keepdim=True)
    xmax, v = xmax.max(dim=3, keepdim=True)

    x = x - xmax 

    x = x.clamp(min=float(-dynamic_range), max=None) # [-80, 0]
    return x

def resample(audio, old_sr, new_sr): 
    """ 
    resample audio using torchaudio
    params:
        audio (torch.tensor): audio with shape (batch, channels, time)
            or (channels, time) or (time,)
    """
    audio = torchaudio.transforms.Resample(old_sr,new_sr)(audio)
    return audio

def downmix(audio, keepdim = True):
    """
    downmix audio by taking the mean at the channel dimension. 
    audio should be shape (channels, time) or (batch, channels, time)
    audio should be np array
    """
    _check_audio_types(audio)
    _check_audio_shapes(audio)

    if isinstance(audio, np.ndarray):
        audio = audio.mean(axis=-2, keepdims=keepdim)
    elif isinstance(audio, torch.Tensor):
        audio = audio.mean(axis=-2, keepdim=keepdim)

    return audio

def split_on_silence(audio, top_db=80):
    """
    split audio on silence using librosa
    returns:
        split_audio (list): list of np.arrays with split audio
        intervals (np.ndarray):  intervals[i] == 
                    (start_i, end_i) are the start and end time 
                    (in samples) of non-silent interval i.
    """
    _check_audio_types(audio)
    intervals = librosa.effects.split(audio, 
                                    top_db=top_db, 
                                    frame_length=2048, 
                                    hop_length=512)
    
    split_audio = [audio[i[0]:i[1]] for i in intervals]
    return split_audio, intervals

def zero_pad(audio, length):
    """
    zero pad audio left and right to match length
    audio must be shape (samples, ) (mono) or (batch, samples)
    """
    # _check_audio_types(audio)
    if audio.ndim == 2:
        if isinstance(audio, torch.Tensor):
            return torch.stack([zero_pad(a, length) for a in audio])
        return np.stack([zero_pad(a, length) for a in audio])

    if isinstance(audio, torch.Tensor):
        return_tensor = True
        audio_type_ptr = audio
    else:
        return_tensor = False
    
    if len(audio) < length:
        pad_length = length - len(audio)
        pad_right = int(np.floor(pad_length/2))
        pad_left = int(np.ceil(pad_length/2))
        audio = np.pad(audio, (pad_left, pad_right), 'constant', constant_values=(0, 0))
    
    assert len(audio) >= length

    if return_tensor and isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio).type_as(audio_type_ptr)
    
    return audio
    