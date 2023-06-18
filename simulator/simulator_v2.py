import itertools
import torch


def to_digits(x: torch.Tensor, num_digits: int):
    return [x // (10 ** i) % 10 for i in reversed(range(num_digits))]

class SubleqSimV2():
    def __init__(self, max_val, num_mem, num_inst, curriculum=False, curriculum_num=0):
        self.max_val = max_val
        # self.max_val = 100
        self.num_mem = num_mem
        self.num_inst = num_inst

        self.mem = torch.zeros(num_mem, dtype=torch.int64)
        self.inst = torch.zeros((num_inst, 3), dtype=torch.int64)
        self.pc = torch.zeros(1, dtype=torch.int64)

        # self.dec_dictionary = {}
        # for prefix in ['m', 'i', 'p']:
        #     for i in range(10):
        #         c = len(self.dec_dictionary)
        #         self.dec_dictionary[c] = prefix + str(i)

        # self.dec_dictionary = {i: i for i in range(self.max_val)}
        self.dec_dictionary = {i: i for i in range(10)}
        # self.dec_dictionary[self.max_val] = "pc"
        # self.dec_dictionary[self.max_val + 1] = "mem"
        # self.dec_dictionary[self.max_val + 2] = "inst"
        # self.dec_dictionary[self.max_val + 3] = ","
        # self.dec_dictionary[10 + 3] = "ans"
        # self.dec_dictionary[10 + 4] = "+"
        # self.dec_dictionary[10 + 5] = "-"
        # self.dec_dictionary[10 + 6] = "="
        c = len(self.dec_dictionary)
        self.dec_dictionary[c + 0] = "pc"
        self.dec_dictionary[c + 1] = "mem"
        self.dec_dictionary[c + 2] = "inst"
        self.dec_dictionary[c + 3] = ","
        self.dec_dictionary[c + 4] = " "
        # invert the dictionary
        self.enc_dictionary = {v: k for k, v in self.dec_dictionary.items()}
        self.num_tokens = len(self.dec_dictionary)
        self.num_mem_digit = len(str(self.max_val - 1))
        self.num_inst_digit = len(str(self.num_mem - 1))

        # pc + mem + inst + extra 
        self.col_dim = 1 + self.num_mem + self.num_inst * 3 + 3
        self.curriculum_num = curriculum_num
        self.curriculum = curriculum
    
    def set_curriculum_num(self, curriculum_num):
        self.curriculum_num = curriculum_num

    def check_curriculum_done(self):
        if not self.curriculum:
            return True
        else:
            return 2**self.curriculum_num > self.num_inst

    def create_state(self):
        self.mem = torch.randint(0, self.max_val, (self.num_mem,))
        self.inst = torch.randint(0, self.num_mem, (self.num_inst, 3))
        if self.curriculum:
            # self.pc = torch.randint(0, min(self.num_inst, max(1, 2 * self.curriculum_num)), (1,))
            self.pc = torch.randint(0, min(self.num_inst, max(1, 2 ** self.curriculum_num)), (1,))
        else:
            self.pc = torch.randint(0, self.num_inst, (1,))
        # self.pc = torch.tensor([0])
        return self.tokenize_state()
        # return torch.cat((self.tokenize_state(), torch.tensor([self.enc_dictionary['pc']])), dim=-1)
        # return torch.cat((self.tokenize_state(), torch.tensor([self.enc_dictionary['ans']] * self.num_digit)), dim=-1)

    def step(self):
        a = self.inst[self.pc, 0]
        b = self.inst[self.pc, 1]
        c = self.inst[self.pc, 2]
        # b=1
        # a=0
        # Account for overflow and underflow
        self.mem[b] = (self.mem[b] - self.mem[a]) % self.max_val

        # if (self.curriculum and self.num_inst < 2 ** self.curriculum_num) or not self.curriculum:
        if self.mem[b] > self.max_val // 2:
            self.pc = c
        else:
            self.pc = (self.pc + 1) % self.num_inst
        # else:
        #     self.pc = (self.pc + 1) % self.num_inst
        return self.tokenize_state()
        # diff = ((self.mem[b] - self.mem[a]) % self.max_val).long()
        # if diff < 0:
        #     diff += self.max_val
        # elif diff > 31:
        #     diff -= 32
        # return torch.cat((self.tokenize_state(), torch.Tensor([diff]).long()))
        # diff = torch.Tensor(to_digits(diff, self.num_digit)).long()
        # return torch.cat((self.tokenize_state(), diff))

    def tokenize_state(self):
        # x = ['pc'] + self.pc.tolist() + ['mem'] + self.mem.tolist() + ['inst'] + self.inst.flatten().tolist()
        # return torch.tensor([self.enc_dictionary[t] for t in x])
        pc_token = torch.tensor([self.enc_dictionary['pc']])
        mem_token = torch.tensor([self.enc_dictionary['mem']])
        inst_token = torch.tensor([self.enc_dictionary['inst']])
        # return torch.clone(torch.cat((pc_token, self.pc, mem_token, self.mem)))
        # return torch.clone(torch.cat((pc_token, self.pc, mem_token, self.mem, inst_token, self.inst.flatten())))
        # return torch.clone(self.mem[:2])

        # plus_token = self.enc_dictionary['+']
        # minus_token = self.enc_dictionary['-']
        # equal_token = self.enc_dictionary['=']
        # return torch.tensor(to_digits(self.mem[0], self.num_digit) + [plus_token] + to_digits(self.mem[1], self.num_digit) + [equal_token])

        def add_prefix(x, prefix):
            # return [self.enc_dictionary[prefix + str(i.item())] for i in x]
            return x

        comma = self.enc_dictionary[',']
        space = self.enc_dictionary[' ']
        mem_digits = torch.tensor([add_prefix(to_digits(m, self.num_mem_digit), 'm') + [space] for m in self.mem]).flatten()[:-1]
        inst_digits = torch.tensor([
            list(itertools.chain.from_iterable([add_prefix(to_digits(i, self.num_inst_digit), 'i') + [space] for i in self.inst[j]]))[:-1] + [comma]
        for j in range(self.num_inst)]).flatten()[:-1]
        pc_digits = torch.tensor([add_prefix(to_digits(self.pc, self.num_inst_digit), 'p')]).flatten()

        return torch.cat((pc_token, pc_digits, mem_token, mem_digits, inst_token, inst_digits))

    def detok(self, tokens: torch.Tensor):
        # return ' '.join([str(self.dec_dictionary[t.item()]) for t in tokens])
        return ''.join([str(self.dec_dictionary[t.item()]) for t in tokens])




if __name__ == "__main__":
    sim = SubleqSimV2(10, 10, 10)
    state = sim.create_state()
    print(sim.detok(state))

    for i in range(10):
        state = sim.step()
        print(sim.detok(state))

    print(sim.mem)

    print(sim.pc)
