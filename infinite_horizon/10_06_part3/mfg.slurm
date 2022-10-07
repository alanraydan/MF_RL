#!/bin/bash

#SBATCH --nodes=1 --ntasks-per-node=12

cd $SLURM_SUBMIT_DIR

python3 main_condensed.py params.json > test.out
