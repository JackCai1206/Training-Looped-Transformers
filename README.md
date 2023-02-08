# Training-Looped-Transformers

We investigate whether a transformer initialized at random can be trained to execute OISC programs, as a follow up of "[Looped Transformers as Programmable Computers](https://arxiv.org/pdf/2301.13196v1.pdf)"

## OISC definition
We use SUBLEQ, which is capable of defining a universal computer. 

    SUBLEQ(a, b, c):
        mem[c] = mem[a] - mem[b]
        if mem[flag] ≤ 0:
            goto instruction p
        else:
            PC = PC + 1

## Data format
The input and output space is a $n\times d$ matrix. For each $d$-dimensional token, 
## Architecture
