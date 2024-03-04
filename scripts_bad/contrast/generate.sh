
cuda=4

for model in 05; do
for alpha in 20.0; do

CUDA_VISIBLE_DEVICES=${cuda} accelerate launch \
    --mixed_precision no \
    --num_processes 1 \
    --num_machines 1 \
    --num_cpu_threads_per_process 10 \
generate.py \
    --collator_name text2text \
    --model_name blender \
    --pretrained_model_path /home/lyakhtin/repos/ctg/datasets/click_checkpoints/checkpoints_bad/contrast_${model}/${alpha} \
    --save_path /home/lyakhtin/repos/ctg/datasets/click_checkpoints/checkpoints_bad/contrast_${model}/${alpha} \
    --infer_data_paths /home/lyakhtin/repos/ctg/datasets/click_checkpoints/data_bad/blender/valid.txt /home/lyakhtin/repos/ctg/datasets/click_checkpoints/data_bad/blender/test.txt \
    --infer_names valid test \
    --max_input_length 128 \
    --max_decoder_input_length 32 \
    --seed 0 \
    --lower \
    --max_length 32 \
    --min_length 5 \
    --batch_size 6 \
    --temperature 1 \
    --top_k 0 \
    --top_p 0.9 \
    --num_beams 1 \
    --num_return_sequences 25 \
    --length_penalty 1 \
    --repetition_penalty 1 \
    --no_repeat_ngram_size 0 \
    --encoder_no_repeat_ngram_size 0

done
done
