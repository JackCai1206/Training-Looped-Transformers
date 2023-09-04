import difflib
import torch

from model.model import LoopedTransformerModelV2
import simulator

checkpoint = 'checkpoints/breezy-sweep-2/final-val-acc-1.0.pt-epoch-4771.pt'
num_trials = 100

checkpoint = torch.load(checkpoint)
args = checkpoint['args']

train_sim = simulator.SubleqSimV3(mem_bits=args.N, num_mem=args.num_mem, ary=args.ary)
model = LoopedTransformerModelV2(train_sim, args.emsize, args.nhead, args.nlayers, args.nhid, args.dropout).to(args.device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def model_step(model, state):
    state = state.to(args.device).unsqueeze(0)
    output = model(state)
    output = torch.argmax(output, dim=-1)
    return output.squeeze().to('cpu')

# Test 1: simple loop

