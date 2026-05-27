for task in cola
do
    CUDA_VISIBLE_DEVICES=1 python run_glue.py \
    --model_name_or_path FacebookAI/roberta-base \
    --task_name $task \
    --enable_galore \
    --lora_all_modules \
    --max_length 512 \
    --seed=1234 \
    --lora_r 8 \
    --galore_scale 2 \
    --per_device_train_batch_size 32 \
    --update_proj_gap 500 \
    --learning_rate 1e-5 \
    --num_train_epochs 30 \
    --output_dir ./galore_results/ft/roberta_base_peak_alloc/${task}_r8
done