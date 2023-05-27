NUM_TRAINERS=1
export CUDA_VISIBLE_DEVICES=1

cmd=(
torchrun \
    --rdzv_backend=c10d
    --rdzv_endpoint=localhost:20000
    --standalone
    --nnodes=1 
    --nproc_per_node=$NUM_TRAINERS 
    ../main.py 
    --cuda 
    --sim_type=v2
    # -N=5
    # -n=24
    --num_train=100000
    --num_valid=5000
    --epochs=3000
    --log_interval=20 
    --save=/data/hulab/zcai75/looped-transformers/checkpoints/ 
    --batch_size=250
    --eval_batch_size=5000 
    --emsize=256
    --nhid=2048 
    --nlayers=20
    --nhead=4 
    --dropout=0.2 
    --lr=5.28E-05
    # --lr=3.06E-05
    # --lr=4.25E-02
    # --clip=-1
    --grad_noise=5e-2
    --block_diag=False
    --signed_mag=10 
    --optimizer=adam
    --scheduler=plateau
    --criterion=ce
    --task=1
    --fix_set=True
    --weight_decay=0.1
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
