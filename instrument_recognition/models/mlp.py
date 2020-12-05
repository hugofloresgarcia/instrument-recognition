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

class Ensemble(pl.LightningModule):
    
    def __init__(self, model_cls, n_members, random_seed=20, **model_kwargs):
        super().__init__()
        
        self.members = nn.ModuleList()
        torch.manual_seed(random_seed)
        for idx in range(n_members):
            temp_seed = random_seed + idx + 1
            torch.manual_seed(temp_seed)
            member = model_cls(**model_kwargs)
            self.members.append(member)
        
        # don't forget to reconfigure the seed
        torch.manual_seed(random_seed)
    
    def forward(self, x):
        ensemble_preds = torch.stack([m(x) for m in self.members], dim=0)
        return ensemble_preds.mean(dim=0)
            
    

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
    