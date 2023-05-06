from collections import defaultdict
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn

from simulator import SubleqSim
from simulator.simulator_v2 import SubleqSimV2

class SubleqDataSet(Dataset):
    data_cache = defaultdict(dict)
    targets_cache = defaultdict(dict)
    def __init__(self, sim: SubleqSim, num_data, device, mode="train", task=2, fix_set=True):
        self.num_data = num_data
        self.data_iter = self.get_data_iter(sim, num_data, device, mode, task, fix_set)

    def get_data_iter(self, sim: SubleqSim, num_data, device, mode="train", task=2, fix_set=True):
        i = 0
        while True:
            i = i % num_data
            if fix_set and i in self.data_cache[mode]:
                x = self.data_cache[mode][i]
                y = self.targets_cache[mode][i]
            else:
                sim.create_state()
                x = torch.clone(sim.X).T.to(device)
                y = torch.clone(sim.step(task)).T.to(device)
                print(x,y)
                if fix_set:
                    self.data_cache[mode][i] = x
                    self.targets_cache[mode][i] = y
            yield x, y
            i += 1

    def __len__(self):
        return self.num_data
    
    def __getitem__(self, idx):
        return next(self.data_iter)


class SubleqDataSetV2(Dataset):
    data_cache = defaultdict(dict)
    targets_cache = defaultdict(dict)
    def __init__(self, sim: SubleqSimV2, num_data, device, mode="train", task=2, fix_set=True):
        self.num_data = num_data
        self.data_iter = self.get_data_iter(sim, num_data, device, mode, task, fix_set)

    def get_data_iter(self, sim: SubleqSimV2, num_data, device, mode="train", task=2, fix_set=True):
        i = 0
        # sim.create_state()
        while True:
            i = i % num_data
            if fix_set and i in self.data_cache[mode]:
                x = self.data_cache[mode][i]
                y = self.targets_cache[mode][i]
            else:
                x = torch.clone(sim.create_state()).to(device)
                # x = torch.clone(sim.tokenize_state()).to(device)
                y = torch.clone(sim.step()).to(device)
                y = nn.functional.one_hot(y, sim.num_tokens).float()
                if fix_set:
                    self.data_cache[mode][i] = x
                    self.targets_cache[mode][i] = y
            yield x, y
            i += 1
    
    def clear_cache(self):
        self.data_cache.clear()
        self.targets_cache.clear()

    def __len__(self):
        return self.num_data
    
    def __getitem__(self, idx):
        return next(self.data_iter)

if __name__ == "__main__":
    sim = SubleqSim(1, 1, 1, 1, 1, 1, 1)
    dataset = SubleqDataSet(sim, 10, "cpu")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for x, y in dataloader:
        print(x.shape, y.shape)
        break
