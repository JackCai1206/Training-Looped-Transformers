from math import ceil, log2
import torch

# Convert a row vector into a binary matrix
def to_binary_col(x: torch.Tensor, bits):
    assert x.max() < 2**bits
    assert len(x.shape) == 0 or x.shape == (x.shape[0],) # x is a row vector or a scalar
    assert x.dtype == torch.int64
    if len(x.shape) == 0:
        x = x.unsqueeze(0)
    mask = 1 << torch.arange(bits-1, -1, -1, device=x.device)
    return (x.unsqueeze(1) & mask).T.ne(0).float()

# Convert a binary matrix into a row vector
def from_binary_col(x: torch.Tensor):
    assert len(x.shape) <= 2
    if len(x.shape) == 1:
        x = x.unsqueeze(1)
    x = x.T
    mask = torch.arange(x.shape[1]-1, -1, -1, device=x.device)
    return torch.sum(x * (1 << mask), dim=1).long()

# Simulator for the SUBLEQ instruction
class SubleqSim():
    def __init__(self, N, s, m, n):
        self.N = N
        self.s = s
        self.m = m
        self.n = n
        self.log_n = log_n = ceil(log2(n))
        self.d = d = 5 * log_n + 3 * N + 1
        self.col_dim = 2 ** ceil(log2(d))
        self.inst_len = n - m - s
        assert self.inst_len > 0
        self.PC_slice = slice(d-2*log_n-1,d-log_n-1), slice(0,1)
        self.pos_enc_slice = slice(d-log_n-1,d-1), slice(0,n)
        self.scratch_ind_slice = slice(d-1,d), slice(0,n)
        self.M_slice = slice(3*log_n,3*log_n+N), slice(s,s+m)
        self.C_slice = slice(0,3*log_n), slice(s+m,n)
        self.create_state()
    
    def create_state(self):
        N, s, m, n, log_n, d, inst_len = self.N, self.s, self.m, self.n, self.log_n, self.d, self.inst_len
        self.X = torch.zeros(self.col_dim, n)
        PC = to_binary_col(torch.randint(s+m, n, (1,)), log_n)
        pos_enc = to_binary_col(torch.arange(0, n), log_n)
        scratch_ind = torch.cat([torch.ones(1, s), torch.zeros(1, n-s)], dim=1)
        M = torch.randint(0, 2, (N, m)).float()
        C = torch.cat([
        to_binary_col(torch.randint(s, s+m, (inst_len,)), log_n),
        to_binary_col(torch.randint(s, s+m, (inst_len,)), log_n),
        to_binary_col(torch.randint(s+m, n, (inst_len,)), log_n)], dim=0)

        # print(PC.shape, X[d-2*log_n-1:d-log_n-1,0].shape)
        self.X[self.PC_slice] = PC
        self.X[self.pos_enc_slice] = pos_enc
        self.X[self.scratch_ind_slice] = scratch_ind
        self.X[self.M_slice] = M
        self.X[self.C_slice] = C
        return self.X
    
    def step(self):
        N, s, m, n, log_n, d, inst_len = self.N, self.s, self.m, self.n, self.log_n, self.d, self.inst_len
        PC = self.X[self.PC_slice]
        pos_enc = self.X[self.pos_enc_slice]
        scratch_ind = self.X[self.scratch_ind_slice]
        M = self.X[self.M_slice]
        C = self.X[self.C_slice]

        PC_num = from_binary_col(PC)
        args = from_binary_col(C[:,PC_num-s-m].reshape(-1,3).T)
        diff = from_binary_col(M[:,args[1]-s]) - from_binary_col(M[:,args[0]-s])
        M[:,args[1]-s] = to_binary_col(diff, N).squeeze()
        if diff <= 0:
            PC_num = args[2]
        else:
            PC_num = PC_num + 1
        PC = to_binary_col((PC_num + 1) % n, log_n)

        self.X[self.PC_slice] = PC
        self.X[self.pos_enc_slice] = pos_enc
        self.X[self.scratch_ind_slice] = scratch_ind
        self.X[self.M_slice] = M
        self.X[self.C_slice] = C

        return self.X

if __name__ == '__main__':
    # sim = SubleqSim(10, 5, 5, 20)
    # for i in range(100):
    #     sim.step()
    #     print(sim.X)
    
    # x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # print(to_binary_col(x, 4))

    x = torch.tensor([[1, 0, 1, 1], [0, 1, 0, 1]]).T
    print(from_binary_col(x))
    pass
