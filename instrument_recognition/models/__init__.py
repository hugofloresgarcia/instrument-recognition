from collections import OrderedDict

import torch
import torch.nn as nn
import pytorch_lightning as pl

from instrument_recognition.models.transformer import TransformerEncoder

recurrent_model_sizes = {
    'lstm': {
        'num_layers': 4, 
        'num_heads': None},
    'gru': {
        'num_layers': 4, 
        'num_heads': None},
    'transformer': {
        'num_layers': 2, 
        'num_heads': 4},
}

model_sizes = {
    'tiny':  dict(d_input=512,  d_intermediate=128, has_linear_proj=True),
    'small': dict(d_input=512,  d_intermediate=512, has_linear_proj=False),
    'mid':   dict(d_input=6144, d_intermediate=512, has_linear_proj=True),
    'huge':  dict(d_input=6144, d_intermediate=6144, has_linear_proj=False)
}

class Model(pl.LightningModule):

    def __init__(self, model_size: str, output_dim: int, 
                recurrence_type: str = 'bilstm', dropout: float = 0.3):
        super().__init__()
        self.save_hyperparameters()

        self.output_dim = output_dim
        self.model_size = model_size
        self.recurrence_type = recurrence_type
        self.has_linear_proj = model_sizes[model_size]['has_linear_proj']

        assert self.output_dim > 0
        assert self.model_size in model_sizes.keys(), f'model_size must be one of {model_sizes.keys()}'

        d_input = model_sizes[model_size]['d_input']
        d_intermediate = model_sizes[model_size]['d_intermediate']

        # add the proper linear transformation depending on model size
        if self.has_linear_proj:
            self.fc_proj = nn.Linear(d_input, d_intermediate)

        # add recurrent layers
        num_layers = recurrent_model_sizes[recurrence_type]['num_layers']
        num_heads = recurrent_model_sizes[recurrence_type]['num_heads']
        recurrent_layer, r_dim = get_recurrent_layer(layer_name=recurrence_type, d_in=d_intermediate, num_layers=num_layers, 
                                                        d_hidden=d_intermediate, num_heads=num_heads, dropout=dropout)
        self.__setattr__(name=recurrence_type, value=recurrent_layer)

        # add the fully connected classifier :)
        self.fc_output = nn.Linear(r_dim, output_dim)
    
    @classmethod
    def from_hparams(cls, hparams):
        obj = cls.__init__(model_size=hparams.model_size, output_dim=hparams.output_dim, 
                           recurrence_type=hparams.recurrence_type, dropout=hparams.dropout)
        obj.hparams = hparams
        return obj


    def forward(self, x):
        # input should be (batch, sequence, embedding)
        assert x.ndim == 3
        assert x.shape[-1] == model_sizes[self.model_size]['d_input']

        if self.has_linear_proj:
            x = self.fc_proj(x)
        
        recurrent_layer = self.__getattr__(self.recurrence_type)
        if 'lstm' in self.recurrence_type \
            or 'gru' in self.recurrence_type:
            x, hiddens = recurrent_layer(x)
        else:
            x = recurrent_layer(x)

        x = self.fc_output(x)

        return x

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parent_parser.add_argument

def get_recurrent_layer(layer_name: str = 'bilstm', d_in: int = 512, num_layers: int = 4,
                        d_hidden: int = 128, num_heads: int = 4, dropout: float = 0.3):
    if 'lstm' in layer_name:
        bidirectional = 'bi' in layer_name
        num_directions = 2 if 'bi' in layer_name else 1
        layer = nn.LSTM(input_size=d_in, hidden_size=d_hidden, num_layers=num_layers, 
                        batch_first=True, dropout=dropout, bidirectional=bidirectional)
        output_dim = d_in * num_directions

    elif 'gru' in layer_name:
        bidirectional = 'bi' in layer_name
        num_directions = 2 if 'bi' in layer_name else 1
        layer = nn.GRU(input_size=d_in, hidden_size=d_hidden, num_layers=num_layers, batch_first=True, 
                       dropout=dropout, bidirectional=bidirectional)
        output_dim = d_hidden * num_directions
    
    elif 'transformer' in layer_name:
        layer = TransformerEncoder(d_model=d_in, num_heads=num_heads, d_hidden=d_hidden, 
                                   num_layers=num_layers, dropout=dropout)
        output_dim = d_in
    
    return layer, output_dim

if __name__ == '__main__':
    from itertools import product
    from torchsummaryX import summary
    # get a param count for all models
    for size, recurrence_type in product(model_sizes.keys(), recurrent_model_sizes.keys()):
        model = Model(model_size=size, output_dim=19, recurrence_type=recurrence_type)
        sample_input = torch.zeros((1, 10, model_sizes[size]['d_input']))
        print(summary(model, sample_input))

