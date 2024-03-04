
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --mixed_precision fp16 \
    --num_processes 1 \
    --num_machines 1 \
    --num_cpu_threads_per_process 10 \
train.py \
    --collator_name classification \
    --model_name roberta \
    --pretrained_model_path /home/lyakhtin/repos/ctg/pretrained_models/roberta-base \
    --model_args 2 \
    --save_path /home/lyakhtin/repos/ctg/datasets/click_checkpoints/checkpoints_cls/bad \
    --train_data_path /home/lyakhtin/repos/ctg/datasets/click_checkpoints/data_cls/bad/train.txt \
    --max_input_length 192 \
    --seed 42 \
    --adafactor \
    --batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --num_epochs 1 \
    --warmup_steps 50
