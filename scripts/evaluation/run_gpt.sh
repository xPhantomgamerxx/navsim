TRAIN_TEST_SPLIT=eval_random
EXPERIMENT_NAME=${1:-gpt_agent_eval}

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_gpt.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=gpt_agent \
worker=ray_distributed \
experiment_name=$EXPERIMENT_NAME
