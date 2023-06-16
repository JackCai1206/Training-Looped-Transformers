NUM_TRAINERS=1
export CUDA_VISIBLE_DEVICES=0,1

cmd=(
torchrun \
    --rdzv_backend=c10d
    --rdzv_endpoint=localhost:20001
    --standalone
    --nnodes=1 
    --nproc_per_node=$NUM_TRAINERS 
    ../main.py 
    --cuda 
    --sim_type=v2
    # -N=5
    # -n=24
    --num_mem=8
    --num_inst=8
    -N=8
    --num_train=100000
    --num_valid=5000
    --epochs=1600
    --log_interval=20 
    --save=../checkpoints
    --batch_size=1250
    --eval_batch_size=5000 
    --emsize=256
    --nhid=512
    --nlayers=14
    --nhead=4 
    --dropout=0 
    # --lr=1.72E-05
    --lr=5.28E-05
    # --lr=2.37E-05
    # --lr=3.06E-05
    # --lr=4.25E-02
    # --clip=-1
    --grad_noise=0
    --block_diag=False
    --signed_mag=10 
    --optimizer=adam
    --scheduler=constant
    --patience=300
    --criterion=ce
    --label_smoothing=0.5
    --task=1
    --fix_set=True
    --weight_decay=0
    # --lr_finder
    --sweep_config=../sweep.yaml 
    --sweep 
    # --sweep_id=asyggw01
    --wandb 
    # --resume=../checkpoints/super-snowflake-694/best-val-acc-0.0594-epoch-200.pt
)

echo "Executing: "
printf "%q " "${cmd[@]}"
echo

"${cmd[@]}" 
