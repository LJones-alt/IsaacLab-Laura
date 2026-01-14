

## script to run experiments in the background 

task="Dev-IK-Rel-Place-v0"
horizon=5000
num_rollouts=100
run_file="scripts/imitation_learning/robomimic/play_ensemble_v05.py"
ensemble_size=15

##### this now runs the experiments

## ensemble 15
seed=101
exp_name="${task}_ensemble_${ensemble_size}_seed_${seed}_highdem"
./isaaclab.sh -p $run_file --task $task --horizon $horizon --num_rollouts $num_rollouts --ensemble_size $ensemble_size --seed $seed --exp_name $exp_name --headless

seed=107
exp_name="${task}_ensemble_${ensemble_size}_seed_${seed}_highdem"
./isaaclab.sh -p $run_file --task $task --horizon $horizon --num_rollouts $num_rollouts --ensemble_size $ensemble_size --seed $seed --exp_name $exp_name --headless

seed=115
exp_name="${task}_ensemble_${ensemble_size}_seed_${seed}_highdem"
./isaaclab.sh -p $run_file --task $task --horizon $horizon --num_rollouts $num_rollouts --ensemble_size $ensemble_size --seed $seed --exp_name $exp_name --headless

## ensemble 10 
ensemble_size=10
seed=101
exp_name="${task}_ensemble_${ensemble_size}_seed_${seed}_highdem"
./isaaclab.sh -p $run_file --task $task --horizon $horizon --num_rollouts $num_rollouts --ensemble_size $ensemble_size --seed $seed --exp_name $exp_name --headless

seed=107
exp_name="${task}_ensemble_${ensemble_size}_seed_${seed}_highdem"
./isaaclab.sh -p $run_file --task $task --horizon $horizon --num_rollouts $num_rollouts --ensemble_size $ensemble_size --seed $seed --exp_name $exp_name --headless

seed=115
exp_name="${task}_ensemble_${ensemble_size}_seed_${seed}_highdem"
./isaaclab.sh -p $run_file --task $task --horizon $horizon --num_rollouts $num_rollouts --ensemble_size $ensemble_size --seed $seed --exp_name $exp_name --headless

## ensemble 5 
ensemble_size=5
seed=101
exp_name="${task}_ensemble_${ensemble_size}_seed_${seed}_highdem"
./isaaclab.sh -p $run_file --task $task --horizon $horizon --num_rollouts $num_rollouts --ensemble_size $ensemble_size --seed $seed --exp_name $exp_name --headless

seed=107
exp_name="${task}_ensemble_${ensemble_size}_seed_${seed}_highdem"
./isaaclab.sh -p $run_file --task $task --horizon $horizon --num_rollouts $num_rollouts --ensemble_size $ensemble_size --seed $seed --exp_name $exp_name --headless

seed=115
exp_name="${task}_ensemble_${ensemble_size}_seed_${seed}_highdem"
./isaaclab.sh -p $run_file --task $task --horizon $horizon --num_rollouts $num_rollouts --ensemble_size $ensemble_size --seed $seed --exp_name $exp_name --headless

## ensemble 1
ensemble_size=1
seed=101
exp_name="${task}_ensemble_${ensemble_size}_seed_${seed}_highdem"
./isaaclab.sh -p $run_file --task $task --horizon $horizon --num_rollouts $num_rollouts --ensemble_size $ensemble_size --seed $seed --exp_name $exp_name --headless

seed=107
exp_name="${task}_ensemble_${ensemble_size}_seed_${seed}_highdem"
./isaaclab.sh -p $run_file --task $task --horizon $horizon --num_rollouts $num_rollouts --ensemble_size $ensemble_size --seed $seed --exp_name $exp_name --headless

seed=115
exp_name="${task}_ensemble_${ensemble_size}_seed_${seed}_highdem"
./isaaclab.sh -p $run_file --task $task --horizon $horizon --num_rollouts $num_rollouts --ensemble_size $ensemble_size --seed $seed --exp_name $exp_name --headless
