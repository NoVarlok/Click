
paths=()
paths[${#paths[*]}]="/home/lyakhtin/repos/ctg/datasets/click_checkpoints/checkpoints_bad/ft/test"
paths[${#paths[*]}]="/home/lyakhtin/repos/ctg/datasets/click_checkpoints/checkpoints_bad/unlikelihood_01/test"
paths[${#paths[*]}]="/home/lyakhtin/repos/ctg/datasets/click_checkpoints/checkpoints_bad/gedi/test_20.0"
paths[${#paths[*]}]="/home/lyakhtin/repos/ctg/datasets/click_checkpoints/checkpoints_bad/dexperts/test_5.0"
paths[${#paths[*]}]="/home/lyakhtin/repos/ctg/datasets/click_checkpoints/checkpoints_bad/director_02/test_10.0"
paths[${#paths[*]}]="/home/lyakhtin/repos/ctg/datasets/click_checkpoints/checkpoints_bad/cringe_02/test"
paths[${#paths[*]}]="/home/lyakhtin/repos/ctg/datasets/click_checkpoints/checkpoints_bad/contrast_05/20.0/test"

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
    --context_file /home/lyakhtin/repos/ctg/datasets/click_checkpoints/data_bad/raw/test.txt \
    --infer_data_paths ${paths[*]} \
    --max_input_length 192 \
    --seed 42 \
    --batch_size 800
