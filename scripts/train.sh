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
    --epochs=800
    --log_interval=20 
    --save=/data/hulab/zcai75/looped-transformers/checkpoints/ 
    --batch_size=1000 
    --eval_batch_size=1000 
    --nhid=1024 
    --nlayers=16
    --nhead=4 
    --dropout=0 
    --lr=0.0013
    --grad_noise=0 
    --block_diag=False
    --signed_mag=100 
    --optimizer=adam
    --scheduler=plateau 
    --task=2
    --fix_set=True
    --lr_finder
    # --sweep_config=../sweep.yaml 
    # --sweep 
    --wandb 
    # --resume=/data/hulab/zcai75/looped-transformers/checkpoints/model.pt 
)

echo "Executing: "
printf "%q " "${cmd[@]}"
echo

"${cmd[@]}" 
