import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from instrument_recognition.utils.audio_utils import amplitude_to_db, get_stft_filterbank
from instrument_recognition.utils.plot_utils import plot_filterbank

def plot_filterbank(fb, n_rows, n_cols, title='filterbank', figsize=(6.4, 4.8)):
    """
    plot a group of time domain filters
    params:
        fb (np.ndarray or torch.Tensor): filterbank w shape (filters, sample)
        n_rows: rows for plot
        n_cols: cols for plot

    returns:
        matplotlib figure
    """
    n_subplots = fb.shape[0]
    n = np.arange(fb.shape[1]) # time axis

    fig, axes = plt.subplots(n_rows, n_cols, squeeze=True)
    fig.set_size_inches(figsize)
    fig.suptitle(title)

    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.plot(fb[i][0])
        ax.set_xlabel('sample')

    fig.tight_layout()
    
    return fig

class Melspectrogram(pl.LightningModule):

    def __init__(self, 
                sr=48000, n_mels=128, fmin=0.0, fmax=None, 
                power_melgram=1.0, return_decibel_melgram=True, 
                trainable_fb=False, htk=True):
        """
        creates a single 1D convolutional layers with filters fixed to
        a mel filterbank.
        """
        #TODO: make two separate classes for spec and melspec ala kapre
        super().__init__()
        if fmax is None:
            fmax = sr / 2
        
        self.sr = sr
        # scale some parameters according to openl3
        self.sr_scale = self.sr // 48000
        self.n_fft = int(self.sr_scale  * 2048)
        self.hop_size = int(self.sr_scale * 242)
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.htk = htk
        self.return_decibel_melgram = return_decibel_melgram
        self.power_melgram = power_melgram
        

        f_real, f_imag = get_stft_filterbank(self.n_fft, window='hann')
        self.n_bins = self.n_fft // 2 + 1
        
        mel_filters = librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=n_mels,
                                        fmin=self.fmin, fmax=self.fmax, htk=self.htk) # (mel, 1+n_fft/2)

        fmel_real = np.matmul(mel_filters, f_real)
        fmel_imag = np.matmul(mel_filters, f_imag)
        
        self.conv1d_real = nn.Conv1d(
            in_channels=1, 
            out_channels=self.n_bins,
            kernel_size=self.n_fft,
            stride=self.hop_size, 
            padding=1101,
            bias=False
        )

        self.conv1d_imag = nn.Conv1d(
            in_channels=1, 
            out_channels=self.n_bins,
            kernel_size=self.n_fft,
            stride=self.hop_size, 
            padding=1101,
            bias=False
        )

        self.freq2mel = nn.Linear(self.n_bins, self.n_mels, bias=False)

        # fix the weights to value
        f_real = torch.from_numpy(f_real).float()
        f_imag = torch.from_numpy(f_imag).float()
        mel_filters = torch.from_numpy(mel_filters).float()

        self.mel_filters = nn.Parameter(mel_filters)
        self.conv1d_real.weight = nn.Parameter(f_real.unsqueeze(1))
        self.conv1d_imag.weight = nn.Parameter(f_imag.unsqueeze(1))
        self.freq2mel.weights = nn.Parameter(mel_filters)
        
        self.freeze()

    def forward(self, x):
        self.freeze()
        # input should be shape (batch, channels, time)
        # STFT and mel filters
        # forward pass through filterbank
        real = self.conv1d_real(x)
        imag = self.conv1d_imag(x)

        x = real ** 2 + imag ** 2
        
        x = x.permute(0, 2, 1)
        x = torch.matmul(x, self.mel_filters.T)
        x = x.permute(0, 2, 1)

        # NOW, take the square root to make it a power 1 melgram
        x = torch.pow(torch.sqrt(x), self.power_melgram)

        x = x.view(-1, 1, 128, 199)
        
        if self.return_decibel_melgram:
            x = amplitude_to_db(x)
        return x

    def freeze_up_to(self, layer_name):
        """
        freeze model from input up to (and including) a layer specified by layer_name
        a warning will be raised if you use an incorrect layer name
        """
        for name, module in self.named_modules():
            module.eval()
            print(f'setting {name} to eval')
            if layer_name in name:
                break

        for name, param in self.named_parameters():
            param.requires_grad = False
            print(f'freezing {name}')
            if layer_name in name:
                break
            
    def unfreeze_starting_at(self, layer_name):
        """
        unfreeze starting at a  layer specified w layer_name
        """
        unfreeze=False
        self.train()
        #TODO: it could be that the batchnorms that are around frozen areas need to be frozen with .eval() as well
        for name, module in self.named_modules():
            if layer_name in name:
                unfreeze=True

            if 'batch' in name and not unfreeze:
                print(f'setting {name} to eval')
                module.eval()
            if unfreeze:
                print(f'setting {name} to train')
                module.train()

        for name, param in self.named_parameters():
            if layer_name in name:
                unfreeze=True

            if unfreeze:
                print(f'unfreezing: {name}')
                param.requires_grad=Tru