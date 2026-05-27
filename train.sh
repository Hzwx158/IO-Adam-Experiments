# LLaMA-60M, GaLore-Adam, 1 A100, 1 Node

DS_l=(
    # "c4"
    "minipile"
)

Opt_l=(
    "io_adam"
    "adam"
)

for opt in ${Opt_l[@]}
do
    for ds in ${DS_l[@]}
    do
        torchrun --standalone --nproc_per_node 1 torchrun_main.py \
            --model_config configs/llama_60m.json \
            --lr 0.01 \
            --galore_scale 0.25 \
            --rank 128 \
            --update_proj_gap 200 \
            --batch_size 256 \
            --total_batch_size 512 \
            --num_training_steps 200 \
            --warmup_steps 1000 \
            --weight_decay 0 \
            --dtype bfloat16 \
            --eval_every 1000 \
            --optimizer $opt \
            --save_dir ./results/llama_60m_${opt}_${ds} \
            --ds $ds
            # --optimizer galore_adamw 
    done
done