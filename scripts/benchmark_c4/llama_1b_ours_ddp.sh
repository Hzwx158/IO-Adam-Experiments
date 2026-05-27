# LLaMA-1B, GaLore-Adam, 8 A100, 1 Node
export WANDB_API_KEY='b99d23ec71c0f6189be80ec591c59d7f064b01dd'
CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.run --standalone --nproc_per_node 1 /prodcpfs/user/yiting/lowrank_optim/GaLore/torchrun_main_c4_ddp.py \
    --model_config /prodcpfs/user/yiting/lowrank_optim/GaLore/configs/llama_1b.json \
    --io_adam_lr 0.003 \
    --lr 0.003 \
    --galore_scale 0.25 \
    --rank 1024 \
    --update_proj_gap 200 \
    --batch_size 16 \
    --total_batch_size 512 \
    --num_training_steps 100000 \
    --warmup_steps 10000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer io_adam \
    --beta2 0.95 \
    --single_gpu \
    --save_dir /prodcpfs/user/yiting/lowrank_optim/GaLore/results/pretrain/1b_ours_lr3e-3_3e-3_beta2_095_rank_1024_single_gpu