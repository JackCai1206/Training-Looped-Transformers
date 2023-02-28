from collections import defaultdict
import torch
from torch.utils.data import DataLoader, Dataset


from simulator import SubleqSim

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
            yield self.data_cache[mode][i], self.targets_cache[mode][i] 
            i += 1

    def __len__(self):
        return self.num_data
    
    def __getitem__(self, idx):
        return next(self.data_iter)
