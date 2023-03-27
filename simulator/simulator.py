from math import ceil, log2
import torch

# Simulator for the SUBLEQ instruction
class SubleqSim():
    def __init__(self, N, s, m, n, signed_mag=100, block_diag=True):
        self.N = N
        self.s = s
        self.m = m
        self.n = n
        self.mag = signed_mag
        self.log_n = log_n = ceil(log2(n))
        self.d = d = 4 * log_n + N + 1 if block_diag else max(N, 3*log_n) + log_n + 1
        self.col_dim = 2 ** ceil(log2(d))
        self.inst_len = n - m - s
        assert self.inst_len > 0
        self.PC_slice = slice(d-2*log_n-1,d-log_n-1), slice(0,1)
        self.pos_enc_slice = slice(d-log_n-1,d-1), slice(0,n)
        self.scratch_ind_slice = slice(d-1,d), slice(0,n)
        self.M_slice = (slice(d-log_n-N-1,d-log_n-1), slice(s,s+m)) if block_diag else (slice(d-log_n-N-1,d-log_n-1), slice(s,s+m))
        self.C_slice = (slice(d-4*log_n-N-1,d-log_n-N-1), slice(s+m,n)) if block_diag else (slice(d-4*log_n-1,d-log_n-1), slice(s+m,n))
        self.create_state()
    
    def create_state(self):
        N, s, m, n, log_n, d, inst_len = self.N, self.s, self.m, self.n, self.log_n, self.d, self.inst_len
        self.X = torch.ones(self.col_dim, n) * (-self.mag)
        PC = self.to_binary_col(torch.randint(s+m, n, (1,)), log_n)
        pos_enc = self.to_binary_col(torch.arange(0, n), log_n)
        scratch_ind = self.to_signed_binary(torch.cat([torch.ones(1, s), torch.zeros(1, n-s)], dim=1))
        M = self.to_binary_col(torch.randint(0, 2**N, (m,)), N)
        C = torch.cat([
        self.to_binary_col(torch.randint(s, s+m, (inst_len,)), log_n),
        self.to_binary_col(torch.randint(s, s+m, (inst_len,)), log_n),
        self.to_binary_col(torch.randint(s+m, n, (inst_len,)), log_n)], dim=0)

        # print(PC.shape, X[d-2*log_n-1:d-log_n-1,0].shape)
        # assert torch.sum(self.X[self.PC_slice]) == -torch.sum(torch.zeros_like(self.X[self.PC_slice]))
        self.X[self.PC_slice] = PC
        # assert torch.sum(self.X[self.pos_enc_slice]) == -torch.sum(torch.zeros_like(self.X[self.pos_enc_slice]))
        self.X[self.pos_enc_slice] = pos_enc
        # assert torch.sum(self.X[self.scratch_ind_slice]) == -torch.sum(torch.zeros_like(self.X[self.scratch_ind_slice]))
        self.X[self.scratch_ind_slice] = scratch_ind
        # assert torch.sum(self.X[self.M_slice]) == -torch.sum(torch.zeros_like(self.X[self.M_slice]))
        self.X[self.M_slice] = M
        # assert torch.sum(self.X[self.C_slice]) == -torch.sum(torch.zeros_like(self.X[self.C_slice]))
        self.X[self.C_slice] = C
        return self.X
    
    def step_copy(self):
        # Copy the state
        self.X = self.X.clone()
        return self.X
    
    def step_PC_add1(self):
        # Increment the PC
        N, s, m, n, log_n, d, inst_len = self.N, self.s, self.m, self.n, self.log_n, self.d, self.inst_len
        PC = self.X[self.PC_slice]
        
        PC_num = self.from_binary_col(PC)
        PC_num = (PC_num + 1) % n
        PC = self.to_binary_col(PC_num, log_n)

        self.X[self.PC_slice] = PC
        return self.X
    
    def step_subleq(self):
        N, s, m, n, log_n, d, inst_len = self.N, self.s, self.m, self.n, self.log_n, self.d, self.inst_len
        PC = self.X[self.PC_slice]
        pos_enc = self.X[self.pos_enc_slice]
        scratch_ind = self.X[self.scratch_ind_slice]
        M = self.X[self.M_slice]
        C = self.X[self.C_slice]

        PC_num = self.from_binary_col(PC)
        args = self.from_binary_col(C[:,PC_num-s-m].reshape(-1,3).T)
        diff = self.from_binary_col(M[:,args[1]-s]) - self.from_binary_col(M[:,args[0]-s])
        M[:,args[1]-s] = self.to_binary_col(diff, N).squeeze()
        if diff <= 0:
            PC_num = args[2]
        else:
            PC_num = PC_num + 1
        PC = self.to_binary_col((PC_num + 1) % n, log_n)

        self.X[self.PC_slice] = PC
        self.X[self.pos_enc_slice] = pos_enc
        self.X[self.scratch_ind_slice] = scratch_ind
        self.X[self.M_slice] = M
        self.X[self.C_slice] = C

        return self.X
    
    def step(self, task):
        if task == 0:
            return self.step_copy()
        elif task == 1:
            return self.step_PC_add1()
        elif task == 2:
            return self.step_subleq()
        else:
            raise ValueError("Invalid task")
    
    # Convert 0/1 binary matrix into -1000/1000 binary matrix
    def to_signed_binary(self, x: torch.Tensor):
        return (x * 2 - 1) * self.mag

    # Convert -1000/1000 binary matrix into 0/1 binary matrix
    def from_signed_binary(self, x: torch.Tensor):
        return (x // self.mag + 1) // 2

    # Convert a row vector into a binary matrix
    def to_binary_col(self, x: torch.Tensor, bits):
        assert x.max() < 2**bits
        assert len(x.shape) == 0 or x.shape == (x.shape[0],) # x is a row vector or a scalar
        assert x.dtype == torch.int64
        if len(x.shape) == 0:
            x = x.unsqueeze(0)
        mask = 1 << torch.arange(bits-1, -1, -1, device=x.device)
        return self.to_signed_binary((x.unsqueeze(1) & mask).T.ne(0).float())

    # Convert a binary matrix into a row vector
    def from_binary_col(self, x: torch.Tensor):
        assert len(x.shape) <= 2
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
        x = self.from_signed_binary(x)
        x = x.T
        mask = torch.arange(x.shape[1]-1, -1, -1, device=x.device)
        return torch.sum(x * (1 << mask), dim=1).long()

if __name__ == '__main__':
    sim = SubleqSim(10, 5, 5, 20)
    # for i in range(100):
    #     sim.step()
    #     print(sim.X)
    
    x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(sim.to_binary_col(x, 4))

    x = torch.tensor([[1, 0, 1, 1], [0, 1, 0, 1]]).T
    print(sim.from_binary_col(x))
    pass
