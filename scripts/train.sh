NUM_TRAINERS=1

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_TRAINERS \
    ../main.py \
    --data=/data/hulab/zcai75/looped-transformers/ \
    --cuda \
    --wandb 