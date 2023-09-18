NUM_TRAINERS=1
export CUDA_VISIBLE_DEVICES=1
# export NCCL_DEBUG=WARN
# CUDA_LAUNCH_BLOCKING=1 
cmd=(
    # --rdzv_backend=c10d
    # --rdzv_endpoint=localhost:20001
    # --standalone
    # --nnodes=1 
    # --nproc_per_node=$NUM_TRAINERS 
    ../main.py 
    --run_name=use-modulo-28
    # --cuda 
    --sim_type=v2
    --use_modulo
    # -N=5
    # -n=24
    # --curriculum
    --num_mem=32
    --num_inst=32
    -N=2
    --ary=10
    --num_train=300000
    --num_valid=5000
    --epochs=10000
    --log_interval=20 
    --save=../checkpoints
    --batch_size=2000
    --eval_batch_size=2000
    --emsize=128
    --nhid=512
    --nlayers=12
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
    --scheduler=plateau
    --patience=300
    --criterion=ce
    --label_smoothing=0.1
    --task=1
    --fix_set
    # --force-diff
    --weight_decay=0.01
    # --lr_finder
    # --sweep_config=../sweep.yaml 
    # --sweep 
    # --sweep_id=4c63qrbi
    --wandb 
    # --resume=../checkpoints/breezy-sweep-2/final-val-acc-0.9992.pt-epoch-4957.pt
    # --resume=../checkpoints/volcanic-oath-229/final-val-acc-0.9932-epoch-481.pt
    # --run_id=bmnigkxb
)

echo "Executing: "
printf "%q " "${cmd[@]}"
echo

python "${cmd[@]}" 

