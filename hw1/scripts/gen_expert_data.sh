#!/bin/bash

N_TRAIN_ROLLOUT=20
N_TEST_ROLLOUT=5

echo "Preping train data for behavioral cloning..."
python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2\
	--num_rollouts=${N_TRAIN_ROLLOUT}
    #--render\
mv expert_data/Humanoid-v2.pkl expert_data/train

echo "Preping test data for behavioral cloning..."
python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2\
    --num_rollouts=${N_TEST_ROLLOUT}
    #--render\
mv expert_data/Humanoid-v2.pkl expert_data/test
