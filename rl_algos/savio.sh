#!/bin/bash
# Job name:
#SBATCH --job-name=SAC_trialrun
#
# Account:
#SBATCH --account=fc_ntugame
#
# Partition:
#SBATCH --partition=savio2_gpu
#
# Request one node:
#SBATCH --nodes=1
#
# Request cores (24, for example)
#SBATCH --ntasks-per-node=2
#
#Request GPUs
#SBATCH --gres=gpu:1
#
# Wall clock limit:
#SBATCH --time=72:00:00
#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akashgokul@berkeley.edu
## Command(s) to run (example):
module load python
module load cuda/10.1
source activate myRAISEenv
python3 SoftActorCritic.py
