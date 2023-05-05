NUM_TRAINERS=1
export CUDA_VISIBLE_DEVICES=2

cmd=(
torchrun 
    --standalone
    --nnodes=1 
    --nproc_per_node=$NUM_TRAINERS 
    ../main.py 
    --data=/data/hulab/zcai75/looped-transformers/ 
    --cuda 
    -N=5
    --epochs=10000
    --log_interval=20 
    --save=/data/hulab/zcai75/looped-transformers/checkpoints/ 
    --batch_size=1250
    --eval_batch_size=5000 
    --emsize=128
    --nhid=512 
    --nlayers=10
    --nhead=4 
    --dropout=0 
    # --lr=3.28E-05
    --lr=5.57E-05
    --grad_noise=0 
    --block_diag=False
    --signed_mag=100 
    --optimizer=adam
    --scheduler=plateau
    --criterion=ce
    --task=1
    --fix_set=True
    --weight_decay=0
    # --lr_finder
    # --sweep_config=../sweep.yaml 
    # --sweep 
    --wandb 
    # --resume=/data/hulab/zcai75/looped-transformers/checkpoints/model.pt 
)

echo "Executing: "
printf "%q " "${cmd[@]}"
echo

"${cmd[@]}" 
