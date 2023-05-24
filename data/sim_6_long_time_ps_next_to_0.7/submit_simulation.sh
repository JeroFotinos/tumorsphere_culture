#!/bin/bash
#$ -q seq_long
#$ -N tumorsph_sim_6
#$ -o log/tumorsph_sim_6
#$ -e log/tumorsph_sim_6
#$ -l m_mem_free=200G
#$ -cwd
####$ -pe env_name 50 ### we request the number of cores


### Script to submit the simulation.
### You can execute it by doing:
### qsub ./submit_simulation.sh

source ~/miniconda3/bin/activate
conda activate
python -m tumorsphere.cli --prob-stem "0.68,0.70,0.72,0.74,0.76" --prob-diff "0" --realizations 10 --steps-per-realization 100 --rng-seed 1292317634567 --parallel-processes 50
