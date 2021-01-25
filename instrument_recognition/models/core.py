from collections import OrderedDict

import torch
import torch.nn as nn
import pytorch_lightning as pl

from instrument_recognition.models.transformer import TransformerEncoder

BATCH_FIRST = False
DEFAULT_SEQ_LEN = 10

model_sizes = {
    'cqt2dft': dict(d_input=128, d_intermediate=128, has_linear_proj=True),
    'vggish': dict(d_input=128, d_intermediate=128, has_linear_proj=True),
    'tiny':  dict(d_input=512,  d_intermediate=128, has_linear_proj=True),
    'small': dict(d_input=512,  d_intermediate=512, has_linear_proj=True),
    'mid':   dict(d_input=6144, d_intermediate=512, has_linear_proj=True),
    'huge':  dict(d_input=6144, d_intermediate=1024, has_linear_proj=True)
}

INPUT_DIMS = {
    'vggish': 128,

    'openl3-mel128-512-music':  512,
    'openl3-mel256-512-music':  512,
    'openl3-mel128-512-env':  512,
    'openl3-mel256-512-env':  512,

    'openl3-mel128-6144-music':  6144,
    'openl3-mel256-6144-music':  6144,
    'openl3-mel128-6144-env':  6144,
    'openl3-mel256-6144-env':  6144,
}

class Embedding(nn.Module):
    """ stack of linear layers with residual connections
    """

    def __init__(self, d_embedding: int, depth: int):
        super().__init__()
        assert depth > 0

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm1d(d_embedding),
                nn.Linear(d_embedding, d_embedding), 
                nn.ReLU(),
             ) for d in range(depth)])
             
    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x

#TODO: add batchnorm
class Model(pl.LightningModule):

    def __init__(self, embedding_name: str, hidden_dim: int, output_dim: int, 
               recurrence_type: str = 'bilstm', recurrence_num_layers: int = 2, 
                dropout: float = 0.3, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.output_dim = output_dim
        self.recurrence_type = recurrence_type
        self.recurrence_num_layers = recurrence_num_layers
        self.has_recurrent_layer = False if recurrence_type.lower() == 'none' else True

        assert self.output_dim > 0
        assert self.model_size in model_sizes.keys(), f'model_size must be one of {model_sizes.keys()}'

        d_input = INPUT_DIMS[embedding_name]
        d_intermediate = hidden_dim

        self.input_shape = (1, DEFAULT_SEQ_LEN, d_input)

        if embedding_name == 'cqt2dft':
            from instrument_recognition.models.embed import CQT2DFTEmbedding
            self.conv = CQT2DFTEmbedding()
            self.input_shape = (1, DEFAULT_SEQ_LEN, 240, 76)

        # linear projection layer
        self.fc_proj = nn.Sequential(
            nn.BatchNorm1d(d_input), 
            nn.Linear(d_input, d_intermediate), 
            nn.ReLU()
        )

        # add recurrent layers
        if self.has_recurrent_layer:
            num_layers = self.recurrence_num_layers
            recurrent_layer, r_dim = get_recurrent_layer(layer_name=recurrence_type, d_in=d_intermediate, 
                                                        num_layers=num_layers, d_hidden=d_intermediate, dropout=dropout)
            # recurrent_layer = nn.Sequential(nn.BatchNorm1d(d_intermediate), recurrent_layer)
            self.__setattr__(name=recurrence_type, value=recurrent_layer)
        else:
            r_dim = d_intermediate

        # add the fully connected classifier :)
        self.fc_output = nn.Sequential(
                    nn.Linear(r_dim, output_dim))
    
    @classmethod
    def from_hparams(cls, hparams):
        assert hasattr(hparams, 'embedding_name')
        obj = cls(**vars(hparams))
        obj.hparams = hparams
        return obj

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = parent_parser
        parser.add_argument('--hidden_dim', type=int, required=True, 
            help=f'hidden dimension size.')
        parser.add_argument('--recurrence_type', type=str, default='none',
            help=f'type of recurrence.')
        parser.add_argument('--recurrence_num_layers', type=int, default=2, 
            help=f'number of stacked recurrent layers for model')
        parser.add_argument('--dropout', type=float, default=0.3, 
            help='dropout for model')
        return parser

    def _linear(self, x, layer):
        #input: collapse sequence and batch dims
        assert not BATCH_FIRST
        seq_dim, batch_dim, feature_dim = x.shape
        x = x.contiguous()
        x = x.view(-1, feature_dim)

        x = layer(x)

        # output: expand sequence and batch dims
        x = x.view(seq_dim, batch_dim, -1)
        return x
    
    def _conv_embed(self, x, layer):
        assert not BATCH_FIRST
        seq_dim, batch_dim, h_dim, w_dim = x.shape
        x = x.contiguous()
        x = x.view(-1, 1, h_dim, w_dim)

        x = layer(x)

        # output: expand sequence and batch dims
        x = x.view(seq_dim, batch_dim, -1)
        return x

    def forward(self, x):
        if hasattr(self, 'conv'):
            x = self._conv_embed(x, self.conv)

        # input should be (sequence, batch, embedding)
        assert x.ndim == 3
        assert x.shape[-1] == model_sizes[self.model_size]['d_input'], f'{x.shape[-1]}-{model_sizes[self.model_size]["d_input"]}'

        if self.has_linear_proj:
            x = self._linear(x, self.fc_proj)
        
        if self.has_recurrent_layer:
            recurrent_layer = self.__getattr__(self.recurrence_type)
            if 'lstm' in self.recurrence_type \
                or 'gru' in self.recurrence_type:
                x, hiddens = recurrent_layer(x)
                # x = hiddens[0][-1].unsqueeze(0)
            else:
                x = recurrent_layer(x)

        seq_dim, batch_dim, feature_dim = x.shape
        x = x.contiguous()
        x = x.view(-1, feature_dim)
        x = self.fc_output(x)
        x = x.view(seq_dim, batch_dim, -1)

        return x

def get_recurrent_layer(layer_name: str = 'bilstm', d_in: int = 512, num_layers: int = 4,
                        d_hidden: int = 128, dropout: float = 0.3):
    if 'lstm' in layer_name:
        bidirectional = 'bi' in layer_name
        num_directions = 2 if 'bi' in layer_name else 1
        layer = nn.LSTM(input_size=d_in, hidden_size=d_hidden, num_layers=num_layers, 
                        batch_first=BATCH_FIRST, dropout=dropout, bidirectional=bidirectional)
        output_dim = d_in * num_directions

    elif 'gru' in layer_name:
        bidirectional = 'bi' in layer_name
        num_directions = 2 if 'bi' in layer_name else 1
        layer = nn.GRU(input_size=d_in, hidden_size=d_hidden, num_layers=num_layers, 
                       batch_first=BATCH_FIRST, dropout=dropout, bidirectional=bidirectional)
        output_dim = d_hidden * num_directions
    
    elif 'transformer' in layer_name:
        # NOTE: DEFAULT NUMBER OF HEADS IN TRANSFORMER
        if '-' not in layer_name:
            num_heads = 4
        _, num_heads = layer_name.split('-')
        layer = TransformerEncoder(d_model=d_in, num_heads=num_heads, d_hidden=d_hidden, 
                                   num_layers=num_layers, dropout=dropout)
        output_dim = d_in

    else: 
        raise ValueError(f'incorrect layer name: {layer_name}')
    
    return layer, output_dim

