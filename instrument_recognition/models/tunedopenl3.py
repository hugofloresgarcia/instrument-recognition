
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from instrument_recognition.models.timefreq import Melspectrogram
from instrument_recognition.models.mlp import MLP512, MLP6144
from instrument_recognition.models.torchopenl3 import OpenL3Mel128, OpenL3Embedding

import instrument_recognition.utils as utils

class OpenL3MLP(pl.LightningModule):

    def __init__(self, embedding_size: int, dropout: float, 
                num_output_units: int, sr=48000):
        super().__init__()
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.num_output_units = num_output_units

        # set up the layers we want to log
        self.log_layers = ('filters', 'fc_seq.0') 
        
        # model architecture stuff
        n_mels = 128
        self.openl3 = OpenL3Embedding(n_mels, embedding_size=embedding_size)
        if embedding_size == 512:
            self.mlp = MLP512(dropout=dropout, num_output_units=num_output_units)
        elif embedding_size == 6144:
            self.mlp = MLP6144(dropout=dropout, num_output_units=num_output_units)
        else:
            raise ValueError(f'incorrect embedding size: {embedding_size}')
    
    def forward(self, x):
        x = self.openl3(x)
        x = self.mlp(x)
        return x

class TunedOpenL3(pl.LightningModule):

    def __init__(self, num_output_units=19, 
                 dropout=0.5,
                 sr=48000, 
                 use_kapre=False):
        super().__init__()
        # required things for working with instrument detection task
        self.sr = sr
        self.dropout = dropout

        # set up the layers we want to log
        self.log_layers = ('filters', 'fc_seq.0') 
        
        # model architecture stuff
        self.filters = Melspectrogram(sr=self.sr)
        self.openl3 = OpenL3Mel128(
            weight_file='/home/hugo/lab/mono_music_sed/instrument_recognition/weights/openl3/openl3_music_6144_no_mel_layer_pytorch_weights', 
            input_shape=(1, self.sr), 
            maxpool_kernel=(16, 24), #512
            # maxpool_kernel=(4, 8), #6144
            maxpool_stride=(4, 8), 
            use_kapre=use_kapre)
        # self.flatten = nn.Flatten()
        self.fc_seq = nn.Sequential(
                nn.Flatten(),

                nn.Linear(512, 128),
                nn.ReLU(), 
                nn.Dropout(p=self.dropout),

                nn.Linear(128, num_output_units))
                
    def forward(self, x):
        if self.openl3.use_kapre:
            raise NotImplementedError("todo: fix gpuid bug for keras melspectrogram")
            # NOTE: still need to do pass thru filters
            # to appease the hooks. so it will be twice a slow
            f = self.filters(x)
            spec = self.openl3.melspec(x, gpuid=None)
            x = spec.type_as(x)
        else:
            x = self.filters(x)
        x = self.openl3(x)
        # x = self.flatten(x)
        x = self.fc_seq(x)
        return x
