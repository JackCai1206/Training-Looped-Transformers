import torch

# Allow sign bit to be encoded as a separate token
# internal representation of memory is a list of int values, instead of a list of digits
class SubleqSimV4():
    def __init__(self, mem_bits, num_mem, ary, num_inst, num_scratch=5, use_modulo=False):
        if use_modulo:
            raise Exception("use_modulo is not supported in this version")
        if (num_mem < ary):
            raise Exception("num_mem must be greater than ary, otherwise would fail to generate all possible states")
        if num_inst > num_mem // 3:
            raise Exception("num_inst must be less than or equal to num_mem // 3")
        self.num_mem = num_mem
        self.num_inst = num_inst
        self.num_scratch = num_scratch
        self.mem_bits = mem_bits
        self.ary = ary
        self.max_val = self.ary ** self.mem_bits

        self.mem = torch.zeros((num_mem, mem_bits), dtype=torch.int8)
        self.dec_dictionary = {i: i for i in range(ary)}
        c = len(self.dec_dictionary)
        self.dec_dictionary[c] = ","
        # add the sign bits
        self.dec_dictionary[c + 1] = "-"
        self.dec_dictionary[c + 2] = "+"
        # invert the dictionary
        self.enc_dictionary = {v: k for k, v in self.dec_dictionary.items()}
        self.num_tokens = len(self.dec_dictionary)
        self.use_modulo = use_modulo

        self.mem = torch.zeros((self.num_mem,), dtype=torch.int8)
        self.scratch = torch.zeros((self.num_scratch,), dtype=torch.int8)

    # convert a list of digits with sign bit to a base 10 number
    def to_base10(self, x: torch.Tensor):
        x.squeeze()
        return sum([x[i] * (self.ary ** i) for i in range(1, len(x))]) * (-1 if x[0] == self.enc_dictionary['+'] else 0)

    # convert a base 10 number to a list of digits with sign bit
    def to_digits(self, x: int, num_digits: int):
        x_pos = -x_pos if x_pos < 0 else x_pos
        digits = [(x_pos // (self.ary ** i)) % self.ary for i in range(num_digits)]
        digits.insert(0, self.enc_dictionary['+'] if x >= 0 else self.enc_dictionary['-'])
        return torch.tensor(digits, dtype=torch.int8)

    def create_state(self, force_diff=True):
        self.mem[1::3] = torch.randint(-self.max_val, self.max_val, (self.num_mem // 3,)) # A
        self.mem[2::3] = torch.tensor([
            torch.randint(-self.max_val + self.mem[a_loc], self.max_val +  + self.mem[a_loc], (0,))
            for a_loc in range(1, self.num_mem, 3)
        ]) # B, with the constraint that -max_val <= B - A <= max_val
        self.mem[3::3] = 1 + 3 * torch.randint(0, self.num_inst, (self.num_mem // 3,)) # C, jump address must be within the instructions
        self.mem[3*self.num_inst::] = torch.randint(-self.max_val, self.max_val, (self.num_mem // 3,)) # free variables after the instructions

    def tokenize_state(self):
        comma = self.enc_dictionary[',']
        mem_digits = torch.tensor([self.to_digits(v, self.mem_bits) + comma for v in self.mem]).flatten()
        scratch_digits = torch.tensor([self.to_digits(v, self.mem_bits) + comma for v in self.scratch]).flatten()
        return torch.cat((mem_digits, scratch_digits))

    def step(self):
        if self.scratch[0] == 0: # copy instruction to scratch
            pc = self.mem[0]
            inst = self.mem[pc:pc+3]
            self.scratch[1:4] = inst
            self.scratch[0] = 1
        elif self.scratch[0] == 1: # calculate B - A
            inst = self.scratch[1:4]
            a_loc = inst[0]
            b_loc = inst[1]
            a = self.mem[a_loc]
            b = self.mem[b_loc]
            diff = b - a
            assert diff >= -self.max_val and diff <= self.max_val
            self.scratch[4] = diff
        elif self.scratch[0] == 2: # store diff into B
            inst = self.scratch[1:4]
            b_loc = inst[1]
            diff = self.scratch[4]
            self.mem[b_loc] = diff
        elif self.scratch[0] == 3: # decide whether to jump
            diff = self.scratch[4]
            if diff <= 0:
                self.scratch[0] = 3
            else:
                self.scratch[0] = 4
        elif self.scratch[0] == 4: # jump
            inst = self.scratch[1:4]
            c_loc = inst[2]
            self.mem[0] = c_loc
            self.scratch[0] = 0
        elif self.scratch[0] == 5: # increment PC
            self.mem[0] += 3
            self.scratch[0] = 0
        else:
            raise Exception("Invalid scratch value")

        return self.tokenize_state()

    def step_random(self): # step 1 to 5 times randomly
        for i in range(torch.randint(1, 6, (1,))):
            self.step()

# testing
if __name__ == "__main__":
    sim = SubleqSimV4(mem_bits=2, num_mem=32, ary=10, num_inst=4, num_scratch=5)
    sim.create_state()
    print(sim.mem)
    print(sim.tokenize_state())
    print(sim.step())
    print(sim.step())
    print(sim.step())
    print(sim.step())
