NUM_TRAINERS=1
export CUDA_VISIBLE_DEVICES=0
# export NCCL_DEBUG=WARN
# CUDA_LAUNCH_BLOCKING=1 
cmd=(
    # --rdzv_backend=c10d
    # --rdzv_endpoint=localhost:20001
    # --standalone
    # --nnodes=1 
    # --nproc_per_node=$NUM_TRAINERS 
    ../main.py 
    # --cuda 
    --sim_type=v2
    # -N=5
    # -n=24
    # --curriculum
    --num_mem=128
    --num_inst=32
    -N=4
    --ary=4
    --num_train=100000
    --num_valid=5000
    --epochs=10000
    --log_interval=20 
    --save=../checkpoints
    --batch_size=1000
    --eval_batch_size=1000
    --emsize=128
    --nhid=512
    --nlayers=14
    --nhead=4 
    --dropout=0 
    # --lr=1.72E-05
    --lr=5E-05
    # --lr=2.37E-05
    # --lr=3.06E-05
    # --lr=4.25E-02
    # --clip=-1
    --grad_noise=0
    # --block_diag
    --signed_mag=10 
    --optimizer=adam
    --scheduler=constant
    --patience=300
    --criterion=ce
    --label_smoothing=0.3
    --task=1
    --fix_set
    --weight_decay=0
    # --lr_finder
    # --sweep_config=../sweep.yaml 
    # --sweep 
    # --sweep_id=sjp4vhpp
    --wandb 
    # --resume=
    --run_id=195v4skm
)

echo "Executing: "
printf "%q " "${cmd[@]}"
echo

python "${cmd[@]}" 
