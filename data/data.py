import torch

from simulator import SubleqSim

def get_data_iter(sim: SubleqSim, batch_size, num_batches, device):
    for i in range(num_batches):
        batch = []
        for j in range(batch_size):
            sim.create_state()
            batch.append((sim.X, sim.step()))
            if len(batch) == batch_size:
                data = torch.stack([x[0] for x in batch], dim=0)
                targets = torch.stack([x[1] for x in batch], dim=0)
                yield data.transpose(-2, -1).to(device), targets.transpose(-2, -1).to(device)
                batch = []
