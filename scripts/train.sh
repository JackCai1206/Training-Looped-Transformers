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
    --epochs=3000 \
    --log_interval=2 \
    --save=/data/hulab/zcai75/looped-transformers/checkpoints/ \
    --batch_size=500 \
    --eval_batch_size=500 \
    --nhid=512 \
    --nlayers=9 \
    --nhead=4 \
    --dropout=0 \
    --lr=20
