import torch

from simulator import SubleqSim

class LoopedTransformerModel(torch.nn.Module):
    r"""Looped transformer model. Takes in the current machine state and return the next machine state."""
    def __init__(self, sim: SubleqSim, nhead, num_encoder_layers, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(sim.col_dim, nhead, dim_feedforward, dropout, activation, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        # self.linear = torch.nn.Linear(d_model, d_model)
        # self.transformer = torch.nn.Transformer(d_model=sim.col_dim, nhead=nhead, num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, batch_first=True)
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
