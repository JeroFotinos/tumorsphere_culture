#!/bin/bash

## Nom du job
#$ -N db_csv_sim_9
## Job séquentiel médium (seq_medium) : le temps de calcul est inférieur à 24 h en temps CPU.
#$ -q seq_medium
## Quantité de mémoire réservée
#$ -l m_mem_free=20G
## Environnement
#$ -cwd
#$ -o log/tumorsph_sim_9_job_1
#$ -e log/tumorsph_sim_9_job_1
#$ -j y

### Script to submit the simulation.
### You can execute it by doing:
### qsub ./submit_simulation.sh

### Executing the python script
source ~/miniconda3/bin/activate
conda activate
tumorsphere mergedbs --dbs-folder ./data --merging-path ./sim_9.db
tumorsphere makedf --db-path ./sim_9.db --csv-path ./sim_9.csv