TRAIN_TEST_SPLIT=eval_challenging

python -m debugpy --listen 5678 --wait-for-client $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_gpt.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=gpt_agent \
worker=ray_distributed \
experiment_name=gpt_eval_finetune_challenging_rk4_integration 
