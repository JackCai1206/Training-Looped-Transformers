import simulator
import torch

from model.model import LoopedTransformerModelV2

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

with open('./compiler/hw3') as f:
    mem = f.read().replace('\n', '').strip().split(' ')
    mem = torch.tensor([1] + [int(m) + 1 for m in mem])

print(mem)
print(train_sim.set_state_val(mem))

stop = False
while not stop: 
    print(train_sim.readable_state(train_sim.tokenize_state()))
    train_sim.step(verbose=False)
    pc_val = train_sim.to_base10(train_sim.mem[0])
    if pc_val > 21 or pc_val <= 0:
        stop = True
    input()
