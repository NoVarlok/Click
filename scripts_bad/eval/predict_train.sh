
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --mixed_precision no \
    --num_processes 1 \
    --num_machines 1 \
    --num_cpu_threads_per_process 10 \
eval_bad.py \
    --collator_name classification \
    --model_name roberta \
    --pretrained_model_path /home/lyakhtin/repos/ctg/datasets/click_checkpoints/checkpoints_cls/bad \
    --model_args 2 \
    --context_file /home/lyakhtin/repos/ctg/datasets/click_checkpoints/data_bad/raw/train.txt \
    --infer_data_paths /home/lyakhtin/repos/ctg/datasets/click_checkpoints/checkpoints_bad/blender/train \
    --max_input_length 192 \
    --seed 42 \
    --batch_size 400
