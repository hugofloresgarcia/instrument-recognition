from collections import OrderedDict

import pytorch_lightning as pl
import torch.nn as nn
import torch


class MLP512(pl.LightningModule):

    def __init__(self, hparams, num_output_units):
        super().__init__()

        self.hparams = hparams

        self.fc = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Linear(512, 128), 
            nn.ReLU(), 
            nn.Dropout(p=self.hparams.dropout), 

            nn.BatchNorm1d(128),
            nn.Linear(128, num_output_units))
        
    def forward(self, x):
        # print(x.shape)
        return self.fc(x)
    
class MLP6144(pl.LightningModule):

    def __init__(self, hparams, num_output_units):
        super().__init__()

        self.hparams = hparams

        self.fc = nn.Sequential(
            nn.BatchNorm1d(6144),
            nn.Linear(6144, 512), 
            nn.ReLU(), 
            nn.Dropout(p=self.hparams.dropout), 

            nn.BatchNorm1d(512),
            nn.Linear(512, 128), 
            nn.ReLU(), 
            nn.Dropout(p=self.hparams.dropout), 

            nn.BatchNorm1d(128),
            nn.Linear(128, num_output_units))

    def forward(self, x):
        # print(x.shape)
        return self.fc(x)