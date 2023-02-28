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
from torch_lr_finder import LRFinder
from torch.utils.data.dataloader import DataLoader
import yaml

from data import SubleqDataSet
from model import LoopedTransformerModel
import simulator

import wandb

parser = argparse.ArgumentParser(description='Training Looped Transformers')
parser.add_argument('--data', type=str, default='.',
                    help='location of the data')
parser.add_argument('--model', type=str, default='Transformer',
                    help='type of network')
# parser.add_argument('--emsize', type=int, default=200,
#                     help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=2048,
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
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='use CUDA')
parser.add_argument('--mps', action='store_true', default=False,
                        help='enables macOS GPU training')
parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--nhead', type=int, default=8,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry_run', action='store_true',
                    help='verify the code and the model')
parser.add_argument('--grad_noise', type=float, default=5e-2)
parser.add_argument('--block_diag', type=bool, default=True)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--criterion', type=str, default='l1')
parser.add_argument('--scheduler', type=str, default='cosine')
parser.add_argument('--warmup_steps', type=int, default=4000)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999))

parser.add_argument('-N', type=int, default=4, required=False, help='Number of bits for integers stored in memory column')
parser.add_argument('-s', type=int, default=4, required=False, help='Number of scratch pad columns')
parser.add_argument('-m', type=int, default=4, required=False, help='Number of memory locations')
parser.add_argument('-n', type=int, default=16, required=False, help='Total number of columns')
parser.add_argument('--num_train', type=int, default=100000, required=False, help='Number of training data points')
parser.add_argument('--num_valid', type=int, default=5000, required=False, help='Number of validutation data points')
parser.add_argument('--signed_mag', type=int, default=1, required=False, help='Magnitude of signed binary numbers')
parser.add_argument('--task', type=int, default=2, required=False, help='Task for curriculum learning')
parser.add_argument('--fix_set', type=bool, default=True, help="Fix the train/val set for each epoch.")

parser.add_argument('--lr_finder', action='store_true', default=False)
parser.add_argument('--wandb', action='store_true', default=False)
parser.add_argument('--sweep', action='store_true', default=False)
parser.add_argument('--sweep_config', type=str, default=None)

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

def quantize_data(data):
    # set data to signed_mag if it is close
    data[data > 1] = args.signed_mag
    data[data < -1] = -args.signed_mag
    return data

def evaluate(model, step, eval_loader, criterion, args):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for i, (data, targets) in enumerate(eval_loader):
            output = model(data)
            output = quantize_data(output)
            total_loss += criterion(output, targets).item()
            if i == 0 and args.wandb:
                wandb.log({ 'example_input': wandb.Image(data[0].T.detach().cpu().numpy()),
                            'example_output': wandb.Image(output[0].T.detach().cpu().numpy()),
                            'example_target': wandb.Image(targets[0].T.detach().cpu().numpy()),
                            'register_diff': wandb.Image(torch.abs(output[0] - targets[0]).T.detach().cpu().numpy())}, step=step, commit=False)
    return total_loss / (args.num_eval_batches)


def train(model, step, epoch, train_loader, optimizer, criterion, args):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    for i, (data, targets) in enumerate(train_loader):
        step += 1
        i = i+1
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()
        output = model(data)
        output = quantize_data(output)
        loss = criterion(output, targets)
        loss.backward()
        # add gaussian noise to gradients
        for p in model.parameters():
            if p.grad is not None:
                p.grad += torch.randn(p.grad.shape).to(args.device) * args.grad_noise

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        # for p in model.parameters():
        #     p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time

            if args.wandb:
                wandb.log({'train_loss': cur_loss}, step=step, commit=False)

            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:f} | ms/batch {:5.2f} | '
                    'loss {:5.2f}'.format(
                epoch, i, args.num_train_batches, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break
    return step

# def export_onnx(path, batch_size, seq_len):
#     print('The model is also exported in ONNX format at {}.'.format(os.path.realpath(args.onnx_export)))
#     model.eval()
#     dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
#     hidden = model.init_hidden(batch_size)
#     torch.onnx.export(model, (dummy_input, hidden), path)

def main(args):
    if args.wandb:
        wandb.init(project="training-looped-transformers", resume="allow", config=args, id=run_id)
        if args.sweep:
            args_dict = vars(args)
            args_dict.update(wandb.config)
            args = argparse.Namespace(**args_dict)
        wandb.log({'args': str(args)})

    if not os.path.exists(args.save):
        os.makedirs(args.save)

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
        args.device = torch.device("cuda")
    elif use_mps:
        args.device = torch.device("mps")
    else:
        args.device = torch.device("cpu")

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
    sim = simulator.SubleqSim(args.N, args.s, args.m, args.n, args.signed_mag, block_diag=args.block_diag)

    ###############################################################################
    # Build the model
    ###############################################################################

    model = LoopedTransformerModel(sim, args.nhead, args.nlayers, args.nhid, args.dropout).to(args.device)
    if args.criterion == 'mse':
        criterion = nn.MSELoss(reduction='mean')
    elif args.criterion == 'l1':
        criterion = nn.L1Loss(reduction='mean')
    if args.wandb:
        wandb.log({'src_mask': wandb.Image(model.src_mask.float())})
        wandb.watch(model, log='all')

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print('Trainable parameters', trainable_params)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5, betas=args.betas)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-5)

    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, verbose=True, min_lr=1e-6)

    if args.resume is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch_resume = checkpoint['epoch']
        loss_resume = checkpoint['loss']
        print('Resuming from epoch', epoch_resume, 'with loss', loss_resume)
    else:
        epoch_resume = 0
        loss_resume = 0

    args.num_eval_batches = args.num_valid // args.eval_batch_size
    args.num_train_batches = args.num_train // args.batch_size
    train_loader = DataLoader(SubleqDataSet(sim, args.num_train, args.device, task=args.task, fix_set=args.fix_set), batch_size=args.batch_size, shuffle=False)
    eval_loader = DataLoader(SubleqDataSet(sim, args.num_valid, args.device, task=args.task, fix_set=args.fix_set), batch_size=args.eval_batch_size, shuffle=False)

    # Loop over epochs.
    best_val_loss = loss_resume
    step = 0

    if args.lr_finder:
        lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
        lr_finder.range_test(train_loader, val_loader=eval_loader, start_lr=1e-10, end_lr=1, num_iter=10000, step_mode="linear")
        plt = lr_finder.plot(log_lr=False)[0]
        if args.wandb:
            wandb.log({ 'lr_finder': wandb.Image(plt)})
        lr_finder.reset()
        exit()

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(epoch_resume, args.epochs+1):
            epoch_start_time = time.time()
            step = train(model, step, epoch, train_loader, optimizer, criterion, args)
            val_loss = evaluate(model, step, eval_loader, criterion, args)
            scheduler.step(val_loss)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.8f} |'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss))
            print('-' * 89)
            if args.wandb:
                wandb.log({ 'val_loss': val_loss,
                            'lr': optimizer.param_groups[0]['lr']}, step=step, commit=True)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': val_loss,
                    'wandb_id': run_id if args.wandb else None
                }, os.path.join(args.save, f'model-task{args.task}.pt'))
                best_val_loss = val_loss
            # elif val_loss < best_val_loss + 0.01:
            #     pass
            # else:
            #     # Anneal the learning rate if no improvement has been seen in the validation dataset.
            #     lr /= 1.01
            
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    checkpoint = torch.load(os.path.join(args.save, f'model-task{args.task}.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    # with open(os.path.join(args.save, 'model.pt'), 'rb') as f:
    #     model = torch.load(f)
    #     # after load the rnn params are not a continuous chunk of memory
    #     # this makes them a continuous chunk, and will speed up forward pass
    #     # Currently, only rnn model supports flatten_parameters function.
    #     if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
    #         model.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(step)
    torch.save(checkpoint, os.path.join(args.save, f"model{checkpoint['epoch']}.pt"))
    print('=' * 89)
    print('| End of training | test loss {:5.2f}'.format(
        test_loss))
    print('=' * 89)
    if args.wandb:
        wandb.log({ 'test_loss': test_loss}, step=step, commit=True)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        run_id = None if 'wandb_id' not in checkpoint else checkpoint['wandb_id']
    else:
        run_id = wandb.util.generate_id()

    if args.wandb and args.sweep:
        def main_fac(args):
            return lambda: main(args)
        with open(args.sweep_config) as f:
            sweep_config = yaml.load(f, Loader=yaml.FullLoader)
        sweep_id = wandb.sweep(sweep_config, project="training-looped-transformers")
        wandb.agent(sweep_id, function=main_fac(args), count=1)
    else:
        main(args)
