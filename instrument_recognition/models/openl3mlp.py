
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
                num_output_units: int, sr=48000, pretrained: bool =  True, 
                mlp_state_dict_path=None):
        super().__init__()
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.num_output_units = num_output_units

        # set up the layers we want to log
        self.log_layers = ('filters', 'fc_seq.0') 
        
        # model architecture stuff
        n_mels = 128
        self.openl3 = OpenL3Embedding(n_mels, embedding_size=embedding_size, pretrained=pretrained)
        if embedding_size == 512:
            self.mlp = MLP512(dropout=dropout, num_output_units=num_output_units)
        elif embedding_size == 6144:
            self.mlp = MLP6144(dropout=dropout, num_output_units=num_output_units)
        else:
            raise ValueError(f'incorrect embedding size: {embedding_size}')

        if mlp_state_dict_path is not None:
            print(f'LOADING MLP STATE DICT: {mlp_state_dict_path}')
            mlp_state_dict = torch.load(mlp_state_dict_path)
            print(mlp_state_dict.keys())
            self.mlp.load_state_dict(mlp_state_dict)
            
    def forward(self, x):
        x = self.openl3(x)
        x = self.mlp(x)
        return x
