#!/bin/bash

## Nom du job
#$ -N tumorsphere_1
## JOB SEQUENTIEL LONG : temps réel < 45 jours
#$ -q seq_long

## Quantité de mémoire réservée
#$ -l m_mem_free=1.5G 
## Esto es para todos los cultivos!
## Aumentar para asignar al menos 1G por cultivo.

## Environnement
#$ -cwd
#$ -o log/tumorsph_sim_8_job_1
#$ -e log/tumorsph_sim_8_job_1
#$ -j y

### Script to submit the simulation.
### You can execute it by doing:
### qsub ./submit_simulation.sh

### En el directorio donde se encuentre este archivo
### tiene que haber otro directorio llamado 'data'.
### Notar también que se asume la presencia de un directorio
### 'log' para la salida estándar y los errores.

### Executing the python script
source ~/miniconda3/bin/activate
conda activate
tumorsphere simulate --prob-stem "0.65,0.66" --prob-diff "0" --realizations 16 --steps-per-realization 60 --rng-seed 1292317634567 --parallel-processes 32 --ovito False --dat-files False
