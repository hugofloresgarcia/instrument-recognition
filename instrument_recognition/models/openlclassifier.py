from collections import OrderedDict

import pytorch_lightning as pl
import torch.nn as nn
import torch


class OpenLClassifier(pl.LightningModule):

    def __init__(self, hparams, num_output_units):
        super().__init__()

        self.hparams = hparams

        self.fc = nn.Sequential(
            nn.Linear(512, 128), 
            nn.ReLU(), 
            nn.Dropout(p=self.hparams.dropout), 

            nn.Linear(128, num_output_units))
        
    def forward(self, x):
        return self.fc(x)