# LLaMA-7B, GaLore-Adam, 8 A100, 8 Node
export CUDA_VISIBLE_DEVICES=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_BUFFSIZE=1048576
export NCCL_MAX_NCHANNELS=2
torchrun --standalone --nnodes 1 --nproc_per_node 1 \
    --master_addr=localhost --master_port=12348 torchrun_main_c4_ddp.py \
    --model_config configs/llama_7b.json \
    --io_adam_lr 0.005 \
    --lr 0.005 \
    --galore_scale 0.25 \
    --rank 512 \
    --update_proj_gap 500 \
    --batch_size 8 \
    --total_batch_size 512 \
    --num_training_steps 150000 \
    --warmup_steps 15000 \
    --weight_decay 0 \
    --grad_clipping 1.0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer io_adam_lomo \
    --save_dir /prodcpfs/user/yiting/lowrank_optim/GaLore/results/pretrain/7b_ours_lomo_w_moment_lr0005_io_adam_lr_0005_bias_correct