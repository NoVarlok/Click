
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --mixed_precision no \
    --num_processes 1 \
    --num_machines 1 \
    --num_cpu_threads_per_process 10 \
eval_ppl_blender.py \
    --save_name self \
    --pretrained_model_path /home/lyakhtin/repos/ctg/pretrained_models/blenderbot-90M \
    --context_file /home/lyakhtin/repos/ctg/datasets/click_checkpoints/data_bad/blender/train.txt \
    --infer_data_paths /home/lyakhtin/repos/ctg/datasets/click_checkpoints/checkpoints_bad/blender/train \
    --max_input_length 128 \
    --max_decoder_input_length 32 \
    --seed 42 \
    --batch_size 400

