NUM_TRAINERS=1
export CUDA_VISIBLE_DEVICES=3 

cmd=(
torchrun 
    --standalone
    --nnodes=1 
    --nproc_per_node=$NUM_TRAINERS 
    ../main.py 
    --data=/data/hulab/zcai75/looped-transformers/ 
    --cuda 
    --epochs=60000 
    --log_interval=20 
    --save=/data/hulab/zcai75/looped-transformers/checkpoints/ 
    --batch_size=800 
    --eval_batch_size=800 
    --nhid=1024 
    --nlayers=11 
    --nhead=4 
    --dropout=0 
    --lr=10 
    --grad_noise=0 
    --block_diag=False 
    --signed_mag=100 
    --optimizer=adam 
    --scheduler=plateau 
    --task=0 
    --fix_set=False 
    # --lr_finder 
    --sweep_config=../sweep.yaml 
    --sweep 
    --wandb 
    # --resume=/data/hulab/zcai75/looped-transformers/checkpoints/model.pt 
)

echo "Executing: "
printf "%q " "${cmd[@]}"
echo

"${cmd[@]}" 
