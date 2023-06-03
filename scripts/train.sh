NUM_TRAINERS=1
export CUDA_VISIBLE_DEVICES=0

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
    --num_mem=16
    --num_inst=16
    -N=8
    --num_train=100000
    --num_valid=5000
    --epochs=1500
    --log_interval=20 
    --save=../checkpoints
    --batch_size=600
    --eval_batch_size=5000 
    --emsize=128
    --nhid=512 
    --nlayers=16
    --nhead=4 
    --dropout=0 
    --lr=5.28E-05
    # --lr=3.06E-05
    # --lr=4.25E-02
    # --clip=-1
    --grad_noise=0
    --block_diag=False
    --signed_mag=10 
    --optimizer=adam
    --scheduler=plateau
    --patience=200
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
