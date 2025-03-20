TRAIN_TEST_SPLIT=navtest
CHECKPOINT=/home/ubuntu/project_ws/navsim/navsim/agents/transfuser/transfuser_seed_0.ckpt

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=transfuser_agent \
worker=ray_distributed_no_torch \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=transfuser_agent_eval 
