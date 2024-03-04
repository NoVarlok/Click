
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --mixed_precision no \
    --num_processes 1 \
    --num_machines 1 \
    --num_cpu_threads_per_process 10 \
generate.py \
    --collator_name text2text \
    --model_name blender \
    --pretrained_model_path /home/lyakhtin/repos/ctg/pretrained_models/blenderbot-90M \
    --save_path /home/lyakhtin/repos/ctg/datasets/click_checkpoints/checkpoints_bad/blender \
    --infer_data_paths /home/lyakhtin/repos/ctg/datasets/click_checkpoints/data_bad/blender/train.txt \
    --infer_names train \
    --max_input_length 128 \
    --max_decoder_input_length 32 \
    --seed 0 \
    --lower \
    --max_length 32 \
    --min_length 5 \
    --batch_size 32 \
    --temperature 1 \
    --top_k 0 \
    --top_p 0.9 \
    --num_beams 1 \
    --num_return_sequences 20 \
    --length_penalty 1 \
    --repetition_penalty 1 \
    --no_repeat_ngram_size 0 \
    --encoder_no_repeat_ngram_size 0

