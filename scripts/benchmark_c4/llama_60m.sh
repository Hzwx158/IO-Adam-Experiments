# LLaMA-60M, GaLore-Adam, 1 A100, 1 Node
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY='b99d23ec71c0f6189be80ec591c59d7f064b01dd'
torchrun --standalone --nproc_per_node 1  torchrun_main_c4_ddp_coupled_muon.py \
    --model_config configs/llama_60m.json \
    --lr 1e-3 \
    --rank 128 \
    --update_proj_gap 200 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --cos_eval_every 100 \
    --beta1 0.9 \
    --beta2 0.99 \
    --optimizer muon\
    --save_dir ./results/pretrain/60M_muon_baseline\
    --single_gpu \
    #--io_adam_lr 0.02 \