import math
import torch
import torch.nn as nn
from model.TransformerEncoderLayerNoLayerNorm import *
from model.transformer import TransformerBlock

from simulator import SubleqSim
from simulator.simulator_v2 import SubleqSimV2

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        # pe = torch.nn.Parameter(pe)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class LoopedTransformerModel(nn.Module):
    r"""Looped transformer model. Takes in the current machine state and return the next machine state."""
    def __init__(self, sim: SubleqSim, nhead, num_encoder_layers, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        encoder_layer = TransformerEncoderLayerNoLayerNorm(sim.col_dim, nhead, dim_feedforward, dropout, activation, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, norm=None)
        # self.linear = nn.Linear(d_model, d_model)
        # self.transformer = nn.Transformer(d_model=sim.col_dim, nhead=nhead, num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, batch_first=True)
        self.sim = sim
        self.d_model = sim.col_dim
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers

        N, s, m, n, log_n, d, inst_len = self.sim.N, self.sim.s, self.sim.m, self.sim.n, self.sim.log_n, self.sim.d, self.sim.inst_len
        src_mask = torch.ones(n, n, dtype=torch.bool)
        # memory can attend to instructions
        src_mask[s:s+m, s+m:n] = False
        # memory can attend to PC
        src_mask[s:s+m, 0] = False
        # instructions can attend to PC
        src_mask[s+m:n, 0] = False
        # scratch can attend to everything
        src_mask[0:s, :] = False
        self.src_mask = src_mask
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src_mask = self.src_mask.to(src.device)
        # print(src.shape)
        output = self.encoder(src, src_mask, src_key_padding_mask)
        # tgt = src
        # output = self.transformer(src, tgt,src_mask, src_key_padding_mask)
        # print(output.shape)
        return output

class LoopedTransformerModelV2(nn.Module):
    r"""Looped transformer model. Takes in the current machine state and return the next machine state."""
    def __init__(self, sim: SubleqSimV2, emsize, nhead, num_encoder_layers, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.d_model = emsize
        self.sim = sim
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers

        self.pos_encoder = PositionalEncoding(self.d_model, dropout)
        encoder_layer = TransformerEncoderLayer(self.d_model, nhead, dim_feedforward, dropout, activation, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        # self.blocks = nn.ModuleList([TransformerBlock(d_model=self.d_model, d_mlp=dim_feedforward, d_head=self.d_model//self.nhead, num_heads=self.nhead, n_ctx=sim.num_tokens, act_type=activation, attn_only=False, model=[self]) for i in range(num_encoder_layers)])
        self.encoder = nn.Embedding(sim.num_tokens, self.d_model)
        self.decoder = nn.Linear(self.d_model, sim.num_tokens)
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        # x = src
        # for block in self.blocks:
        #     x = block(x)
        # output = x
        output = self.decoder(output)

        return output
