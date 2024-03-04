
paths=()
paths[${#paths[*]}]="/home/lyakhtin/repos/ctg/datasets/click_checkpoints/checkpoints_bad/ft/test"
paths[${#paths[*]}]="/home/lyakhtin/repos/ctg/datasets/click_checkpoints/checkpoints_bad/unlikelihood_01/test"
paths[${#paths[*]}]="/home/lyakhtin/repos/ctg/datasets/click_checkpoints/checkpoints_bad/gedi/test_20.0"
paths[${#paths[*]}]="/home/lyakhtin/repos/ctg/datasets/click_checkpoints/checkpoints_bad/dexperts/test_5.0"
paths[${#paths[*]}]="/home/lyakhtin/repos/ctg/datasets/click_checkpoints/checkpoints_bad/director_02/test_10.0"
paths[${#paths[*]}]="/home/lyakhtin/repos/ctg/datasets/click_checkpoints/checkpoints_bad/cringe_02/test"
paths[${#paths[*]}]="/home/lyakhtin/repos/ctg/datasets/click_checkpoints/checkpoints_bad/contrast_05/20.0/test"

python results_bad.py --infer_data_paths ${paths[*]} --test
