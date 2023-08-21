#!/bin/bash

## Nom du job
#$ -N tumorsphere_8
## JOB SEQUENTIEL LONG : temps réel < 45 jours
#$ -q seq_long
## Quantité de mémoire réservée
#$ -l m_mem_free=40G
## Environnement
#$ -cwd
#$ -o log/tumorsph_sim_9_job_8
#$ -e log/tumorsph_sim_9_job_8
#$ -j y

### Script to submit the simulation.
### You can execute it by doing:
### qsub ./submit_simulation.sh

### Executing the python script
source ~/miniconda3/bin/activate
conda activate
tumorsphere simulate --prob-stem "0.1,0.2,0.3,0.4,0.45,0.5,0.9,0.95" --prob-diff "0" --realizations 2 --steps-per-realization 60 --rng-seed 1292317634567 --parallel-processes 16 --ovito False --dat-files False
