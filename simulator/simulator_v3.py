import itertools
import torch
import math

class SubleqSimV3():
    def __init__(self, mem_bits, num_mem, ary, curriculum=False, curriculum_num=0):
        # if (math.ceil(math.log(num_mem, ary)) * 3 > mem_bits):
        #     raise Exception("mem_bits too small for num_mem")
        if (num_mem < ary):
            raise Exception("num_mem must be greater than ary, otherwise would fail to generate all possible states")
        self.num_mem = num_mem
        self.mem_bits = mem_bits
        self.ary = ary
        self.max_val = self.ary ** self.mem_bits

        self.mem = torch.zeros((num_mem, mem_bits), dtype=torch.int8)
        self.dec_dictionary = {i: i for i in range(ary)}
        c = len(self.dec_dictionary)
        self.dec_dictionary[c] = ","
        # invert the dictionary
        self.enc_dictionary = {v: k for k, v in self.dec_dictionary.items()}
        self.num_tokens = len(self.dec_dictionary)

    def to_base10(self, x: torch.Tensor):
        x.squeeze()
        return sum([x[i] * (self.ary ** i) for i in range(len(x))])

    def to_digits(self, x: torch.Tensor, num_digits: int):
        x_pos = x.clone()
        x_pos[x_pos < 0] = self.max_val + x_pos[x_pos < 0]
        digits = [(x_pos // (self.ary ** i)) % self.ary for i in range(num_digits)]
        return torch.tensor(digits, dtype=torch.int8)

    def create_state(self, force_diff=True):
        self.mem = torch.randint(0, self.ary, (self.num_mem, self.mem_bits))
        # With low probability, set the difference between A and B to be close to 0
        if torch.rand(1) < 0.6 and force_diff:
            a_loc = self.to_base10(self.mem[0]) % self.num_mem
            b_loc = (a_loc + 1) % self.num_mem
            b = self.to_base10(self.mem[b_loc])
            b_noise = torch.clip(b + torch.round(torch.normal(0, 2, (1,))), 0, self.max_val).int()
            self.mem[a_loc] = self.to_digits(b_noise, self.mem_bits)
        return self.tokenize_state()

    def set_state(self, tokens: torch.Tensor):
        comma = torch.tensor([self.enc_dictionary[',']], dtype=torch.int8)
        self.mem = torch.stack(torch.cat((tokens, comma)).split(self.mem_bits + 1))[:, :-1]
        return self.tokenize_state()

    def step(self, verbose=False):
        pc_og = self.to_base10(self.mem[0])
        pc = pc_og % self.num_mem
        if verbose:
            print(f"pc_og: {pc_og}, pc: {pc}, pc_raw: {self.mem[0]}, mem_pc: {self.mem[pc % self.num_mem]}")

        a = self.to_base10(self.mem[pc])
        b = self.to_base10(self.mem[(pc + 1) % self.num_mem])
        c_og = self.to_base10(self.mem[(pc + 2) % self.num_mem])
        if verbose:
            print(f"a: {a}, b: {b}, c: {c_og}")
        a = a % self.num_mem
        b = b % self.num_mem
        c = c_og % self.num_mem
        if verbose:
            print(f"a: {a}, b: {b}, c: {c}")

        # Account for overflow and underflow
        mem_b = self.to_base10(self.mem[b])
        mem_a = self.to_base10(self.mem[a])
        diff = mem_b - mem_a
        self.mem[b] = self.to_digits(diff, self.mem_bits)
        if verbose:
            print(f"mem_b: {mem_b}, mem_a: {mem_a}, mem_b - mem_a: {diff}, diff_final: {self.mem[b]}")

        if diff <= 0: 
            self.mem[0] = self.to_digits(c_og, self.mem_bits)
        else:
            self.mem[0] = self.to_digits((pc_og + 3) % (self.max_val), self.mem_bits)
        if verbose:
            print(f"pc_final: {self.mem[0]}")

        return self.tokenize_state()

    def tokenize_state(self):
        comma = self.enc_dictionary[',']
        mem_digits = torch.tensor([list(m) + [comma] for m in self.mem]).flatten()[:-1]
        return mem_digits

    def readable_state(self, tokens: torch.Tensor):
        comma = torch.tensor([self.enc_dictionary[',']], dtype=torch.int8)
        state = torch.cat((tokens, comma)).split(int((len(tokens) + 1) / self.num_mem))
        state = torch.stack(state)[:, :-1]
        state = [int(self.to_base10(s)) for s in state]
        pc = state[0] % self.num_mem
        a = state[pc]
        b = state[(pc + 1) % self.num_mem]
        c = state[(pc + 2) % self.num_mem]
        a_val = int(state[state[pc] % self.num_mem])
        b_val = int(state[state[(pc + 1) % self.num_mem] % self.num_mem])
        diff = b_val - a_val
        return {'pc': int(pc),
                'eval': 'mem[{}]({}) <-- mem[{}]({}) - mem[{}]({}) = {}'.format(b, b_val, b, b_val, a, a_val, diff),
                'jump': 'pc <-- {}'.format(c) if diff <= 0 else 'pc <-- pc + 3, c={}'.format(c),
                'diff': diff,
                'mem': state}

    def detok(self, tokens: torch.Tensor, verbose=False):
        og_str = ''.join([str(self.dec_dictionary[t.item()]) for t in tokens])
        return ''.join([''.join(list(reversed(s))) + ',' for s in og_str.split(',')])
