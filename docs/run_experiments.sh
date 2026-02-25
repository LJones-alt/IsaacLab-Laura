

## script to run experiments in the background 

task="Dev-IK-Rel-v1"
horizon=500
num_rollouts=100
run_file="scripts/imitation_learning/robomimic/play_ensemble_v05.py"
exp_type="lift_none"

##### this now runs the experiments

# ## ensemble 15
# ensemble_size=15
# seed=101
# exp_name="${task}_ensemble_${ensemble_size}_seed_${seed}_${exp_type}"
# ./isaaclab.sh -p $run_file --task $task --horizon $horizon --num_rollouts $num_rollouts --ensemble_size $ensemble_size --seed $seed --exp_name $exp_name --headless

# seed=107
# exp_name="${task}_ensemble_${ensemble_size}_seed_${seed}_${exp_type}"
# ./isaaclab.sh -p $run_file --task $task --horizon $horizon --num_rollouts $num_rollouts --ensemble_size $ensemble_size --seed $seed --exp_name $exp_name --headless

# seed=115
# exp_name="${task}_ensemble_${ensemble_size}_seed_${seed}_${exp_type}"
# ./isaaclab.sh -p $run_file --task $task --horizon $horizon --num_rollouts $num_rollouts --ensemble_size $ensemble_size --seed $seed --exp_name $exp_name --headless

## ensemble 10
ensemble_size=10
seed=101
exp_name="${task}_ensemble_${ensemble_size}_seed_${seed}_${exp_type}"
./isaaclab.sh -p $run_file --task $task --horizon $horizon --num_rollouts $num_rollouts --ensemble_size $ensemble_size --seed $seed --exp_name $exp_name --headless

seed=107
exp_name="${task}_ensemble_${ensemble_size}_seed_${seed}_${exp_type}"
./isaaclab.sh -p $run_file --task $task --horizon $horizon --num_rollouts $num_rollouts --ensemble_size $ensemble_size --seed $seed --exp_name $exp_name --headless

seed=115
exp_name="${task}_ensemble_${ensemble_size}_seed_${seed}_${exp_type}"
./isaaclab.sh -p $run_file --task $task --horizon $horizon --num_rollouts $num_rollouts --ensemble_size $ensemble_size --seed $seed --exp_name $exp_name --headless

## ensemble 5
ensemble_size=5
seed=101
exp_name="${task}_ensemble_${ensemble_size}_seed_${seed}_${exp_type}"
./isaaclab.sh -p $run_file --task $task --horizon $horizon --num_rollouts $num_rollouts --ensemble_size $ensemble_size --seed $seed --exp_name $exp_name --headless

seed=107
exp_name="${task}_ensemble_${ensemble_size}_seed_${seed}_${exp_type}"
./isaaclab.sh -p $run_file --task $task --horizon $horizon --num_rollouts $num_rollouts --ensemble_size $ensemble_size --seed $seed --exp_name $exp_name --headless

seed=115
exp_name="${task}_ensemble_${ensemble_size}_seed_${seed}_${exp_type}"
./isaaclab.sh -p $run_file --task $task --horizon $horizon --num_rollouts $num_rollouts --ensemble_size $ensemble_size --seed $seed --exp_name $exp_name --headless
