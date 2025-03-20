TRAIN_TEST_SPLIT=worsttest

python -m debugpy --listen 5678 --wait-for-client $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=human_agent \
experiment_name=human_agent 