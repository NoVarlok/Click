
cuda=0

for gamma in 02; do

CUDA_VISIBLE_DEVICES=${cuda} accelerate launch \
    --mixed_precision fp16 \
    --num_processes 1 \
    --num_machines 1 \
    --num_cpu_threads_per_process 10 \
train.py \
    --collator_name text2text_labels \
    --model_name blender_director_${gamma} \
    --pretrained_model_path /home/lyakhtin/repos/ctg/pretrained_models/blenderbot-90M \
    --save_path /home/lyakhtin/repos/ctg/datasets/click_checkpoints/checkpoints_bad/director_${gamma} \
    --train_data_path /home/lyakhtin/repos/ctg/datasets/click_checkpoints/data_bad/labels/train.txt \
    --max_input_length 128 \
    --max_decoder_input_length 32 \
    --seed 42 \
    --adafactor \
    --batch_size 64 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --num_epochs 3 \
    --warmup_steps 50

done
