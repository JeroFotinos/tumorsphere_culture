#!/bin/bash

## Nom du job
#$ -N tumorsphere
## JOB SEQUENTIEL LONG : temps réel < 45 jours
#$ -q seq_long
## Quantité de mémoire réservée
#$ -l m_mem_free=1.5G
## Environnement
#$ -cwd
#$ -o log/tumorsph_sim_8_job_1
#$ -e log/tumorsph_sim_8_job_1
#$ -j y

### Script to submit the simulation.
### You can execute it by doing:
### qsub ./submit_simulation.sh

### Executing the python script
source ~/miniconda3/bin/activate
conda activate
./sim_8_job_1.py