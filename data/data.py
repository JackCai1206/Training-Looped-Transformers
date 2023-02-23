from collections import defaultdict
import torch

from simulator import SubleqSim

data_cache = defaultdict(dict)
targets_cache = defaultdict(dict)

def get_data_iter(sim: SubleqSim, batch_size, num_batches, device, mode="train", task=2, fix_set=True):
    for i in range(num_batches):
        batch = []
        if fix_set and i in data_cache[mode]:
            data = data_cache[mode][i]
            targets = targets_cache[mode][i]
        else:
            for j in range(batch_size):
                sim.create_state()
                x = torch.clone(sim.X)
                y = torch.clone(sim.step(task))
                batch.append((x, y))
            data = torch.stack([x[0] for x in batch], dim=0)
            targets = torch.stack([x[1] for x in batch], dim=0)
            if fix_set:
                data_cache[mode][i] = data.transpose(-2, -1).to(device)
                targets_cache[mode][i] = targets.transpose(-2, -1).to(device)
        yield data_cache[mode][i], targets_cache[mode][i] 
