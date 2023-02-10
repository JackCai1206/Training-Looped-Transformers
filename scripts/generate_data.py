import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ))
import argparse
import code
from simulator import SubleqSim
import torch
from math import *

parser = argparse.ArgumentParser('Data generater for OISC input output pairs')
parser.add_argument('-N', type=int, default=4, required=False, help='Number of bits for integers stored in memory column')
parser.add_argument('-s', type=int, default=4, required=False, help='Number of scratch pad columns')
parser.add_argument('-m', type=int, default=2, required=False, help='Number of memory locations')
parser.add_argument('-n', type=int, default=8, required=False, help='Total number of columns')
parser.add_argument('--num_train', type=int, default=100000, required=False, help='Number of training data points')
parser.add_argument('--num_eval', type=int, default=5000, required=False, help='Number of evalutation data points')
parser.add_argument('--output', type=str, default='', required=False, help='Output directory')

args = parser.parse_args()


for i in range(args.num_train):
    sim = SubleqSim(args.N, args.s, args.m, args.n)

    print(sim.X)
    print(sim.step())

    break
