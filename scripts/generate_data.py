import sys
import os
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ))
import argparse
import code
from simulator import SubleqSim
import torch
from math import *
from tqdm import tqdm
import json

parser = argparse.ArgumentParser('Data generater for OISC input output pairs')
parser.add_argument('-N', type=int, default=4, required=False, help='Number of bits for integers stored in memory column')
parser.add_argument('-s', type=int, default=4, required=False, help='Number of scratch pad columns')
parser.add_argument('-m', type=int, default=4, required=False, help='Number of memory locations')
parser.add_argument('-n', type=int, default=16, required=False, help='Total number of columns')
parser.add_argument('--num_train', type=int, default=10000, required=False, help='Number of training data points')
parser.add_argument('--num_valid', type=int, default=500, required=False, help='Number of validutation data points')
parser.add_argument('--output', type=str, default='.', required=False, help='Output directory')

args = parser.parse_args()

if not path.exists(args.output):
    os.makedirs(args.output)

train_data = []
valid_data = []
sim = SubleqSim(args.N, args.s, args.m, args.n)
for i in tqdm(range(args.num_train)):
    train_data.append(torch.stack([sim.X, sim.step()]))
    sim.create_state()

for i in tqdm(range(args.num_valid)):
    valid_data.append(torch.stack([sim.X, sim.step()]))
    sim.create_state()

train_data = torch.stack(train_data)
valid_data = torch.stack(valid_data)
torch.save(train_data, path.join(args.output, 'train.pt'))
torch.save(valid_data, path.join(args.output, 'valid.pt'))
with open(path.join(args.output, 'config.json'), 'w') as f:
    json.dump(sim.__dict__, f)

print(torch.load(path.join(args.output, 'train.pt')).shape)
print(torch.load(path.join(args.output, 'valid.pt')).shape)
