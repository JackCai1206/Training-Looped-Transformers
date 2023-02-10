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

parser = argparse.ArgumentParser('Data generater for OISC input output pairs')
parser.add_argument('-N', type=int, default=4, required=False, help='Number of bits for integers stored in memory column')
parser.add_argument('-s', type=int, default=4, required=False, help='Number of scratch pad columns')
parser.add_argument('-m', type=int, default=4, required=False, help='Number of memory locations')
parser.add_argument('-n', type=int, default=16, required=False, help='Total number of columns')
parser.add_argument('--num_train', type=int, default=10000, required=False, help='Number of training data points')
parser.add_argument('--num_eval', type=int, default=500, required=False, help='Number of evalutation data points')
parser.add_argument('--output', type=str, default='.', required=False, help='Output directory')

args = parser.parse_args()

if not path.exists(args.output):
    os.makedirs(args.output)

train_data = []
eval_data = []
for i in tqdm(range(args.num_train)):
    sim = SubleqSim(args.N, args.s, args.m, args.n)    
    train_data.append(torch.stack([sim.X, sim.step()]))
    sim.create_state()

for i in tqdm(range(args.num_eval)):
    sim = SubleqSim(args.N, args.s, args.m, args.n)
    eval_data.append(torch.stack([sim.X, sim.step()]))
    sim.create_state()

train_data = torch.stack(train_data)
eval_data = torch.stack(eval_data)
torch.save(train_data, path.join(args.output, 'train.pt'))
torch.save(eval_data, path.join(args.output, 'eval.pt'))

print(torch.load(path.join(args.output, 'train.pt')).shape)
print(torch.load(path.join(args.output, 'eval.pt')).shape)
