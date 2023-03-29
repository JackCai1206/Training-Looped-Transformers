import torch



class SubleqSimV2():
    def __init__(self, max_val, num_mem, num_inst):
        self.max_val = max_val
        self.num_mem = num_mem
        self.num_inst = num_inst

        self.mem = torch.zeros(num_mem, dtype=torch.int64)
        self.inst = torch.zeros((num_inst, 3), dtype=torch.int64)
        self.pc = torch.zeros(1, dtype=torch.int64)

        self.dec_dictionary = {i: i for i in range(-self.max_val, self.max_val)}
        self.dec_dictionary[self.max_val] = "pc"
        self.dec_dictionary[self.max_val + 1] = "mem"
        self.dec_dictionary[self.max_val + 2] = "inst"
        # self.dictionary[self.num_tokens + 3] = ","
        # invert the dictionary
        self.enc_dictionary = {v: k for k, v in self.dec_dictionary.items()}
        self.num_tokens = len(self.dec_dictionary)

        # pc + mem + inst + extra 
        self.col_dim = 1 + self.num_mem + self.num_inst * 3 + 3
    
    def create_state(self):
        self.mem = torch.randint(0, self.max_val, (self.num_mem,))
        self.inst = torch.randint(0, self.num_mem, (self.num_inst, 3))
        self.pc = torch.randint(0, self.num_inst, (1,))
        return self.tokenize_state()

    def step(self):
        a = self.inst[self.pc, 0]
        b = self.inst[self.pc, 1]
        c = self.inst[self.pc, 2]
        # Account for overflow and underflow
        self.mem[b] = (self.mem[b] - self.mem[a]) % self.max_val
        if self.mem[b] > 0:
            self.pc = c
        else:
            # Account for overflow
            self.pc = (self.pc + 1) % self.num_inst
        return self.tokenize_state()

    def tokenize_state(self):
        # x = ['pc'] + self.pc.tolist() + ['mem'] + self.mem.tolist() + ['inst'] + self.inst.flatten().tolist()
        # return torch.tensor([self.enc_dictionary[t] for t in x])
        pc_token = torch.tensor([self.enc_dictionary['pc']])
        mem_token = torch.tensor([self.enc_dictionary['mem']])
        inst_token = torch.tensor([self.enc_dictionary['inst']])
        return torch.clone(torch.cat((pc_token, self.pc, mem_token, self.mem, inst_token, self.inst.flatten())))

    def detok(self, tokens: torch.Tensor):
        return ' '.join([str(self.dec_dictionary[t.item()]) for t in tokens])


if __name__ == "__main__":
    sim = SubleqSimV2(10, 10, 10)
    state = sim.create_state()
    print(sim.detok(state))

    for i in range(10):
        state = sim.step()
        print(sim.detok(state))

    print(sim.mem)

    print(sim.pc)
