import math
import torch
import torch.nn as nn
import instrument_recognition as ir

class TransformerEncoder(nn.Module):

    def __init__(self, d_model, num_heads, d_hidden, num_layers, dropout):
        super().__init__()

        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=5000)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, 
                                                    dim_feedforward=d_hidden, dropout=dropout, 
                                                    activation='relu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """ input should be (batch, sequence, embedding)
        """
        # the transformer uses (sequence, batch, embedding) as a default
        # I personally like (batch, sequence, embedding), so lets
        # reshape as necessary
        
        if ir.models.BATCH_FIRST:
            x = x.permute(1, 0, 2)

        x = self.pos_encoder(x)
        x = self.encoder(x)

        if ir.models.BATCH_FIRST:
            x = x.permute(1, 0, 2)

        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)