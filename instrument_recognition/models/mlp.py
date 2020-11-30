from collections import OrderedDict

import pytorch_lightning as pl
import torch.nn as nn
import torch


class MLP512(pl.LightningModule):

    def __init__(self, dropout, num_output_units):
        super().__init__()

        self.fc = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Linear(512, 128), 
            nn.ReLU(), 
            nn.Dropout(dropout), 

            nn.BatchNorm1d(128),
            nn.Linear(128, num_output_units))
        
    def forward(self, x):
        return self.fc(x)
    
class MLP6144(pl.LightningModule):

    def __init__(self, dropout, num_output_units):
        super().__init__()

        self.fc = nn.Sequential(
            nn.BatchNorm1d(6144),
            nn.Linear(6144, 512), 
            nn.ReLU(), 
            nn.Dropout(dropout), 

            nn.BatchNorm1d(512),
            nn.Linear(512, 128), 
            nn.ReLU(), 
            nn.Dropout(dropout), 

            nn.BatchNorm1d(128),
            nn.Linear(128, num_output_units))

    def forward(self, x):
        return self.fc(x)