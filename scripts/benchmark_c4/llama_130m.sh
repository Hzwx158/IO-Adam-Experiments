# LLaMA-130M, GaLore-Adam, 1 A100, 1 Node
# conda activate /prodcpfs/user/yiting/miniconda3/envs/lowrank_optim
export CUDA_VISIBLE_DEVICES=5,7
export WANDB_API_KEY='b99d23ec71c0f6189be80ec591c59d7f064b01dd'
#export HF_ENDPOINT="https://hf-mirror.com"
torchrun --standalone --nproc_per_node 2  torchrun_main_c4_ddp_postscale_v6.py \
    --model_config configs/llama_130m.json \
    --lr 0.0005 \
    --galore_scale 0.25 \
    --rank 256 \
    --update_proj_gap 200 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --cos_eval_every 100 \
    --optimizer adam \
    --beta2 0.99 \
    --save_dir ./results/pretrain/130M_postscale_v6_free_scale