import itertools
import torch
import math

# Convert a number to a list of digits
def to_digits(x: torch.Tensor, num_digits: int, ary: int):
    return [x // (ary ** i) % ary for i in reversed(range(num_digits))]

def to_base10(x: torch.Tensor, ary: int):
    return sum([x[i] * (ary ** i) for i in range(len(x))])

class SubleqSimV3():
    def __init__(self, mem_bits, num_mem, ary, curriculum=False, curriculum_num=0):
        if (math.ceil(math.log(num_mem, ary)) * 3 > mem_bits):
            raise Exception("mem_bits too small for num_mem")
        self.num_mem = num_mem
        self.mem_bits = mem_bits
        self.inst_arg_bits = mem_bits // 3
        self.ary = ary

        self.mem = torch.zeros((num_mem, mem_bits), dtype=torch.int64)
        self.dec_dictionary = {i: i for i in range(ary)}
        c = len(self.dec_dictionary)
        self.dec_dictionary[c] = ","
        # invert the dictionary
        self.enc_dictionary = {v: k for k, v in self.dec_dictionary.items()}
        self.num_tokens = len(self.dec_dictionary)

    def create_state(self):
        self.mem = torch.randint(0, self.ary, (self.num_mem, self.mem_bits))
        return self.tokenize_state()

    def step(self):
        a = to_base10(self.mem[self.pc, 0: self.inst_arg_bits])
        b = to_base10(self.mem[self.pc, self.inst_arg_bits: 2 * self.inst_arg_bits])
        c = to_base10(self.mem[self.pc, 2 * self.inst_arg_bits: 3 * self.inst_arg_bits])
        # Account for overflow and underflow
        max_val = self.ary ** self.mem_bits
        diff = self.mem[b] - self.mem[a]
        self.mem[b] = diff % max_val

        if diff <= 0:
            self.pc = c
        else:
            self.pc = (self.pc + 1) % self.mem.shape[0]
        return self.tokenize_state()

    def tokenize_state(self):
        comma = self.enc_dictionary[',']
        mem_digits = torch.tensor([to_digits(m, self.num_mem_digit) + [comma] for m in self.mem]).flatten()[:-1]

        return mem_digits

    def detok(self, tokens: torch.Tensor):
        return ''.join([str(self.dec_dictionary[t.item()]) for t in tokens])
