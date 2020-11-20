import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from instrument_recognition.models.timefreq import Melspectrogram
from instrument_recognition.models.torchopenl3 import OpenL3Mel128

import instrument_recognition.utils as utils

class TunedOpenL3(pl.LightningModule):

    def __init__(self, hparams=None, num_output_units=19):
        super().__init__()
        # required things for working with instrument detection task
        self.hparams = hparams

        # set up the layers we want to log
        self.log_layers = ('filters', 'fc_seq.0') 
        
        # model architecture stuff
        self.filters = Melspectrogram(sr=self.hparams.sr)
        self.openl3 = OpenL3Mel128(
            weight_file='./weights/openl3/openl3_music_6144_no_mel_layer_pytorch_weights', 
            input_shape=(1, self.hparams.sr), 
            maxpool_kernel=(16, 24), #512
            # maxpool_kernel=(4, 8), #6144
            maxpool_stride=(4, 8), 
            use_kapre=self.hparams.use_kapre)
        # self.flatten = nn.Flatten()
        self.fc_seq = nn.Sequential(
                nn.Flatten(),

                nn.Linear(512, 128),
                nn.ReLU(), 
                nn.Dropout(p=self.hparams.dropout),

                nn.Linear(128, num_output_units))
                
    def forward(self, x):
        if self.openl3.use_kapre:
            # NOTE: still need to do pass thru filters
            # to appease the hooks. so it will be twice a slow
            f = self.filters(x)
            spec = self.openl3.melspec(x, gpuid=self.hparams.gpuid)
            x = spec.type_as(x)
        else:
            x = self.filters(x)
        x = self.openl3(x)
        # x = self.flatten(x)
        x = self.fc_seq(x)
        return x

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--sr', default=48000, type=int)
        parser.add_argument('--openl3_freeze', default=False, type=utils.train.str2bool)
        parser.add_argument('--openl3_unfreeze_epoch', default=0, type=int)
        parser.add_argument('--use_kapre', default=False, type=utils.train.str2bool)

        return parser