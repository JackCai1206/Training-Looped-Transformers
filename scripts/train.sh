NUM_TRAINERS=1
export CUDA_VISIBLE_DEVICES=3 \

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_TRAINERS \
    ../main.py \
    --data=/data/hulab/zcai75/looped-transformers/ \
    --cuda \
    --wandb \
    --epochs=12000 \
    --log_interval=2 \
    --save=/data/hulab/zcai75/looped-transformers/checkpoints/ \
    --batch_size=500 \
    --eval_batch_size=500 \
    --nhid=1024 \
    --nlayers=11 \
    --nhead=4 \
    --dropout=0.3 \
    --lr=10 \
    --grad_noise=0 \
    --block_diag=False \
    --signed_mag=100 \
    --resume=/data/hulab/zcai75/looped-transformers/checkpoints/model.pt
