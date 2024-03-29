from collections import defaultdict
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn

from simulator import SubleqSim
from simulator.simulator_v2 import SubleqSimV2
from simulator.simulator_v3 import SubleqSimV3

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
    def __init__(self, sim: SubleqSimV3, num_data, task=2, fix_set=True, force_diff=True):
        self.num_data = num_data
        self.data_iter = self.get_data_iter(sim, num_data, task, fix_set)
        self.data_cache = defaultdict(dict)
        self.targets_cache = defaultdict(dict)
        self.force_diff = force_diff

    def get_data_iter(self, sim: SubleqSimV3, num_data, task=2, fix_set=True):
        i = 0
        # sim.create_state()
        while True:
            i = i % num_data
            if fix_set and i in self.data_cache:
                x = self.data_cache[i]
                y = self.targets_cache[i]
            else:
                x = torch.clone(sim.create_state(force_diff=self.force_diff))
                # x = torch.clone(sim.tokenize_state()).to(device)
                y = torch.clone(sim.step(verbose=i==0))
                y = nn.functional.one_hot(y, sim.num_tokens).float()
                if fix_set:
                    self.data_cache[i] = x
                    self.targets_cache[i] = y
            yield x, y
            i += 1
    
    def clear_cache(self):
        self.data_cache.clear()
        self.targets_cache.clear()
    
    def get_cache(self):
        return self.data_cache, self.targets_cache

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
