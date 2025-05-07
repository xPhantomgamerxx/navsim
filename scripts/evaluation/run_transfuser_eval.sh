TRAIN_TEST_SPLIT=eval_failures
CHECKPOINT=/home/ubuntu/project_ws/navsim/navsim/agents/transfuser/transfuser_seed_0.ckpt

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_transfuser_eval.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=transfuser_agent \
agent.checkpoint_path=$CHECKPOINT \
worker=ray_distributed_no_torch \
experiment_name=4.1_tr_improv_spcur \