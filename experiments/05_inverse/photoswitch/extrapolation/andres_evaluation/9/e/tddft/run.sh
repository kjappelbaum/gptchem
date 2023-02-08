#!/bin/bash -l

#SBATCH --job-name mol1
#SBATCH --time 6:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 24
#SBATCH --mem 40000

module load gaussian/g16-C.01
g16 input.com > input.log
