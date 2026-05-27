# LLaMA-60M, GaLore-Adam, 1 A100, 1 Node
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY='b99d23ec71c0f6189be80ec591c59d7f064b01dd'
torchrun --standalone --nproc_per_node 1  torchrun_main.py \
    --model_config configs/llama_60m.json \
    --lr 1e-3 \
    --galore_scale 0.25 \
    --rank 128 \
    --update_proj_gap 200 \
    --batch_size 300 \
    --total_batch_size 300 \
    --num_training_steps 30000 \
    --save_every 30000 \
    --warmup_steps 600 \
    --weight_decay 0.1 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --single_gpu \
    --optimizer io_adam \
    --save_dir ./results/pretrain/60M_galore_memory_alloc