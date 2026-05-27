# LLaMA-7B, GaLore-Adam, 8 A100, 8 Node
CUDA_VISIBLE_DEVICES=0
torchrun --standalone --nnodes 1 --nproc_per_node 1 torchrun_main_c4_ddp.py \
    --model_config configs/llama_7b.json \
    --lr 0.005 \
    --galore_scale 0.25 \
    --rank 1024 \
    --update_proj_gap 500 \
    --batch_size 8 \
    --total_batch_size 512 \
    --num_training_steps 150000 \
    --warmup_steps 15000 \
    --weight_decay 0 \
    --grad_clipping 1.0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer galore_adamw \
    --save_dir /prodcpfs/user/yiting/lowrank_optim/GaLore/results/pretrain/7b_galore