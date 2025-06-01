TRAIN_TEST_SPLIT=eval_random

python -m debugpy --listen 5678 --wait-for-client $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_deepseek.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=deepseek_agent \
worker=ray_distributed \
experiment_name=deepseek_agent_eval 
