# coding: utf-8
import argparse
import json
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
from tqdm import tqdm

from data import get_data_iter
import model
import simulator

import wandb

parser = argparse.ArgumentParser(description='Training Looped Transformers')
parser.add_argument('--data', type=str, default='.',
                    help='location of the data')
parser.add_argument('--model', type=str, default='Transformer',
                    help='type of network')
# parser.add_argument('--emsize', type=int, default=200,
#                     help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=512,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=16,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=200, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=200, metavar='N',
                    help='batch size')
# parser.add_argument('--bptt', type=int, default=35,
#                     help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='use CUDA')
parser.add_argument('--mps', action='store_true', default=False,
                        help='enables macOS GPU training')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--nhead', type=int, default=4,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')

parser.add_argument('-N', type=int, default=4, required=False, help='Number of bits for integers stored in memory column')
parser.add_argument('-s', type=int, default=4, required=False, help='Number of scratch pad columns')
parser.add_argument('-m', type=int, default=4, required=False, help='Number of memory locations')
parser.add_argument('-n', type=int, default=16, required=False, help='Total number of columns')
parser.add_argument('--num_train', type=int, default=10000, required=False, help='Number of training data points')
parser.add_argument('--num_valid', type=int, default=500, required=False, help='Number of validutation data points')
parser.add_argument('--output', type=str, default='.', required=False, help='Output directory')

parser.add_argument('--wandb', action='store_true', default=False)

args = parser.parse_args()

if args.wandb:
    wandb.init(project="training-looped-transformers")

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda.")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    if not args.mps:
        print("WARNING: You have mps device, to enable macOS GPU run with --mps.")

use_mps = args.mps and torch.backends.mps.is_available()
if args.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

###############################################################################
# Load data
###############################################################################

# train_data = torch.load(os.path.join(args.data, 'train.pt'))
# valid_data = torch.load(os.path.join(args.data, 'valid.pt'))
# with open(os.path.join(args.data, 'config.json'), 'r') as f:
#     data_config = json.load(f)

# print('train_data', train_data.shape)
# print('valid_data', valid_data.shape)
# print('data_config', data_config)
sim = simulator.SubleqSim(args.N, args.s, args.m, args.n)

###############################################################################
# Build the model
###############################################################################

if args.model == 'Transformer':
    model = model.LoopedTransformerModel(sim, args.nhead, args.nlayers, args.nhid, args.dropout).to(device)
criterion = nn.MSELoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

# def get_batch(source, i):
#     seq_len = min(args.bptt, len(source) - 1 - i)
#     data = source[i:i+seq_len]
#     target = source[i+1:i+1+seq_len].view(-1)
#     return data, target


def evaluate():
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        num_batches = args.num_valid // args.eval_batch_size
        for i, (data, targets) in enumerate(get_data_iter(sim, args.eval_batch_size, num_batches, device)):
            if args.model == 'Transformer':
                output = model(data)
            total_loss += criterion(output, targets).item()
    return total_loss / (num_batches * args.eval_batch_size)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    num_batches = args.num_train // args.batch_size
    for i, (data, targets) in enumerate(get_data_iter(sim, args.batch_size, num_batches, device)):
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        if args.model == 'Transformer':
            output = model(data)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, i, args.batch_size * num_batches, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}.'.format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate()
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate()
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
