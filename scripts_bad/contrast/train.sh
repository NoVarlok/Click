
cuda=0

for alpha in 20.0; do
for model in 05; do

CUDA_VISIBLE_DEVICES=${cuda} accelerate launch \
    --mixed_precision fp16 \
    --num_processes 1 \
    --num_machines 1 \
    --num_cpu_threads_per_process 10 \
train.py \
    --collator_name text2text_contrast \
    --model_name blender_contrast_${model} \
    --pretrained_model_path /home/lyakhtin/repos/ctg/pretrained_models/blenderbot-90M \
    --model_args ${alpha} \
    --save_path /home/lyakhtin/repos/ctg/datasets/click_checkpoints/checkpoints_bad/contrast_${model}/${alpha} \
    --train_data_path /home/lyakhtin/repos/ctg/datasets/click_checkpoints/data_bad/contrast/train.txt \
    --max_input_length 128 \
    --max_decoder_input_length 32 \
    --seed 42 \
    --adafactor \
    --batch_size 63 \
    --gradient_accumulation_steps 3 \
    --learning_rate 5e-5 \
    --num_epochs 2 \
    --warmup_steps 50

done
done
