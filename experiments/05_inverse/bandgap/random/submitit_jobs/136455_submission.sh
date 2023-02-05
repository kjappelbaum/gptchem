#!/bin/bash

# Parameters
#SBATCH --array=0-99%32
#SBATCH --cpus-per-task=1
#SBATCH --error=/home/kevin/gptchem/experiments/05_inverse/bandgap/random/submitit_jobs/%A_%a_0_log.err
#SBATCH --gpus-per-node=0
#SBATCH --job-name=submitit
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/home/kevin/gptchem/experiments/05_inverse/bandgap/random/submitit_jobs/%A_%a_0_log.out
#SBATCH --partition=LocalQ
#SBATCH --signal=USR2@120
#SBATCH --time=60
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home/kevin/gptchem/experiments/05_inverse/bandgap/random/submitit_jobs/%A_%a_%t_log.out --error /home/kevin/gptchem/experiments/05_inverse/bandgap/random/submitit_jobs/%A_%a_%t_log.err /home/kevin/miniconda3/envs/gpt3/bin/python -u -m submitit.core._submit /home/kevin/gptchem/experiments/05_inverse/bandgap/random/submitit_jobs
