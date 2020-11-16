import torch.nn as nn
import torch
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        took this from the pytorch docs (I think?)
        """
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

class SelfAttention(nn.Module):

    def __init__(self, embedding_dim = 512):
        super().__init__()
        self.embedding_dim = torch.tensor(embedding_dim).float()
        self.wq = nn.Linear(embedding_dim, embedding_dim)
        self.wk = nn.Linear(embedding_dim, embedding_dim)
        self.wv = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        # x should be shape (batch, sequence, dim)
        q = self.wq(x) # shape (batch, sequence, embedding_dim)
        k = self.wk(x) # shape (batch, sequence, embedding_dim)
        v = self.wv(x) # shape (batch, sequence, embedding_dim)

        # dot product between q and k is just correlation at different
        # time steps
        t_corr = torch.matmul(q, k.permute(0, 2, 1)) # shape (batch, sequence, sequence)
        score = torch.sigmoid(t_corr / torch.sqrt(self.embedding_dim)) 

        o = torch.matmul(score, v) # should be shape (batch, sequence, embedding_dim)

        return o

class MultiheadAttention(nn.Module):

    def __init__(self, n_heads,  embedding_dim):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(embedding_dim) for head in range(n_heads)])
        self.w = nn.Linear(n_heads * embedding_dim, embedding_dim)

    def forward(self, x):
        # x should be shape (batch, sequence, dim)
        # concatenate head outputs along embedding dim
        x = torch.cat([head(x) for head in self.heads], dim=-1) 

        return self.w(x)
