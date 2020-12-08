#!/bin/bash
# Job name:
#SBATCH --job-name=benchmark_sac_planning
#
# Account:
#SBATCH --account=fc_ntugame
#
# Partition:
#SBATCH --partition=savio2
#
# Request one node:
#SBATCH --nodes=1
#
# Request cores (24, for example)
#SBATCH --ntasks-per-node=2
#
#Request GPUs
#SBATCH --gres=gpu:0
#
#Request CPU
#SBATCH --cpus-per-task=8
#
# Wall clock limit:
#SBATCH --time=48:00:00
#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lucas_spangher@berkeley.edu
## Command(s) to run (example):
module load python/3.6
source /global/home/users/lucas_spangher/transactive_control/auto_keras_env/bin/activate
# vanilla

python StableBaselines.py sac --own_tb_log=fourier_test_1 --action_space=fourier --reward_function=lcr --planning_steps=0 --planning_model=Oracle --one_day=15 --num_players=10 --pricing_type=TOU --num_steps=10000 --test_planning_env=F --fourier_basis_size=1
python StableBaselines.py sac --own_tb_log=fourier_test_2 --action_space=fourier --reward_function=lcr --planning_steps=0 --planning_model=Oracle --one_day=15 --num_players=10 --pricing_type=TOU --num_steps=10000 --test_planning_env=F --fourier_basis_size=2
python StableBaselines.py sac --own_tb_log=fourier_test_3 --action_space=fourier --reward_function=lcr --planning_steps=0 --planning_model=Oracle --one_day=15 --num_players=10 --pricing_type=TOU --num_steps=10000 --test_planning_env=F --fourier_basis_size=3
python StableBaselines.py sac --own_tb_log=fourier_test_4 --action_space=fourier --reward_function=lcr --planning_steps=0 --planning_model=Oracle --one_day=15 --num_players=10 --pricing_type=TOU --num_steps=10000 --test_planning_env=F --fourier_basis_size=4
python StableBaselines.py sac --own_tb_log=fourier_test_5 --action_space=fourier --reward_function=lcr --planning_steps=0 --planning_model=Oracle --one_day=15 --num_players=10 --pricing_type=TOU --num_steps=10000 --test_planning_env=F --fourier_basis_size=5
python StableBaselines.py sac --own_tb_log=fourier_test_6 --action_space=fourier --reward_function=lcr --planning_steps=0 --planning_model=Oracle --one_day=15 --num_players=10 --pricing_type=TOU --num_steps=10000 --test_planning_env=F --fourier_basis_size=6
python StableBaselines.py sac --own_tb_log=fourier_test_7 --action_space=fourier --reward_function=lcr --planning_steps=0 --planning_model=Oracle --one_day=15 --num_players=10 --pricing_type=TOU --num_steps=10000 --test_planning_env=F --fourier_basis_size=7
python StableBaselines.py sac --own_tb_log=fourier_test_8 --action_space=fourier --reward_function=lcr --planning_steps=0 --planning_model=Oracle --one_day=15 --num_players=10 --pricing_type=TOU --num_steps=10000 --test_planning_env=F --fourier_basis_size=8
python StableBaselines.py sac --own_tb_log=fourier_test_9 --action_space=fourier --reward_function=lcr --planning_steps=0 --planning_model=Oracle --one_day=15 --num_players=10 --pricing_type=TOU --num_steps=10000 --test_planning_env=F --fourier_basis_size=9
python StableBaselines.py sac --own_tb_log=fourier_test_10 --action_space=fourier --reward_function=lcr --planning_steps=0 --planning_model=Oracle --one_day=15 --num_players=10 --pricing_type=TOU --num_steps=10000 --test_planning_env=F --fourier_basis_size=10
