import ast
import difflib
import torch

from model.model import LoopedTransformerModelV2
import simulator
import sys

checkpoint_path = sys.argv[1]
num_trials = 100

checkpoint = torch.load(checkpoint_path)
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

trials = []
failed = []
fail_state = []
diffs = []
diff_pos = []
for i in range(num_trials):
    print(f"Trial {i}")
    start = train_sim.create_state(force_diff=False)
    trials.append({str(start.tolist())})
    failed.append(0)
    fail_state.append(None)
    has_loop = False
    while not has_loop: 
        print(f"Step {len(trials[i])}", end='\r')
        if failed[i] == 0: # don't bother if we've already failed
            current_state = train_sim.tokenize_state()
            next_state_hyp = model_step(model, current_state)
        next_state = train_sim.step()
        diff_pos += torch.atleast_1d(torch.nonzero(current_state != next_state).squeeze()).tolist()
        diffs.append(abs(train_sim.readable_state(next_state)['diff']))
        if failed[i] == 0 and not torch.all(next_state_hyp == next_state, dim=-1):
            diff = difflib.ndiff([str(next_state_hyp.tolist())], [str(next_state.tolist())])
            print('\n'.join(diff))
            failed[i] = len(trials[i])
            print(f"Failed at step {len(trials[i])}")
            fail_state[i] = (current_state, next_state_hyp, next_state)
        if str(next_state.tolist()) not in trials[i]:
            trials[i].add(str(next_state.tolist()))
        else:
            has_loop = True
    print()

# for trial in trials:
#     for state in trial:
#         print(train_sim.readable_state(torch.tensor(ast.literal_eval(state)))['mem'])
#     print('------------------')

print([(len(trials[i]), failed[i]) for i in range(len(trials))])

print('Failed {} out of {} trials'.format(sum([1 for i in range(len(trials)) if failed[i] > 0]), num_trials))

torch.save(fail_state, 'fail_state/' + checkpoint_path.split('/')[1] + '_fail-state.pt')

import matplotlib.pyplot as plt
import numpy as np

sorted_idx = np.argsort(np.array([len(trials[i]) for i in range(len(trials))]))[::-1]
trials = [trials[i] for i in sorted_idx]
failed = [failed[i] for i in sorted_idx]
plt.figure(figsize=(20, 20))
plt.subplot(411)
plt.bar(range(len(trials)), [len(trials[i]) for i in range(len(trials))])
plt.bar(range(len(trials)), failed, color='red')
plt.subplots_adjust(wspace=0)
plt.yscale('log')
plt.xlabel('Trial')
plt.ylabel('Loop Length')
plt.title('Loop Lengths')

plt.subplot(412)
plt.hist([len(trials[i]) for i in range(len(trials))])
plt.subplots_adjust(wspace=0)
plt.xlabel('Loop Length')
plt.ylabel('Frequency')
plt.yscale('log')
plt.title('Loop Length frequencies')

plt.subplot(413)
plt.hist(diffs)
plt.subplots_adjust(wspace=0)
plt.xlabel('Difference')
plt.ylabel('Frequency')

print(diff_pos)
plt.subplot(414)
plt.hist(diff_pos)
plt.subplots_adjust(wspace=0)
plt.xlabel('Modified Register Position')
plt.ylabel('Frequency')

plt.savefig('figures/' + checkpoint_path.split('/')[1] + '_loop-lengths.png')
