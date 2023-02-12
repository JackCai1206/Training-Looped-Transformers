import torch

from simulator import SubleqSim

class LoopedTransformerModel(torch.nn.Module):
    r"""Looped transformer model. Takes in the current machine state and return the next machine state."""
    def __init__(self, sim: SubleqSim, nhead, num_encoder_layers, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(sim.col_dim, nhead, dim_feedforward, dropout, activation, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        # self.linear = torch.nn.Linear(d_model, d_model)
        self.sim = sim
        self.d_model = sim.col_dim
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        N, s, m, n, log_n, d, inst_len = self.sim.N, self.sim.s, self.sim.m, self.sim.n, self.sim.log_n, self.sim.d, self.sim.inst_len
        src_mask = torch.ones(src.shape[1], src.shape[1], dtype=torch.bool, device=src.device)
        # memory can attend to instructions
        src_mask[s:s+m, s+m:n] = False
        # memory can attend to PC
        src_mask[s:s+m, 0] = False
        # instructions can attend to PC
        src_mask[s+m:n, 0] = False
        # scratch can attend to everything
        src_mask[0:s, :] = False
        return self.encoder(src, src_mask, src_key_padding_mask)
