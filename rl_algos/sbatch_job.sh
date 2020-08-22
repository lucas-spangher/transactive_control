#!/bin/bash
# Job name:
#SBATCH --job-name=benchmark_sac_env
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
#SBATCH --time=96:00:00
#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akashgokul@berkeley.edu
## Command(s) to run (example):
module load python/3.6
source activate /global/scratch/akashgokul/conda/test
python3 StableBaselines.py sac