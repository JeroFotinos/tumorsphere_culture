#!/bin/bash

### Correr con sbatch ./submit_simulation.sh
### Los template están pensados para ser usados por
### generate_and_submit_jobs_slurm.sh

#---------------------------------------------------

### Configuración del trabajo

### Nombre de la tarea
#SBATCH --job-name=tumorsphere_1

### Cola a usar (gpu, mono, multi)
#SBATCH --partition=multi

### Cantidad de nodos a usar
#SBATCH --nodes=1

### Cores a utilizar por nodo = procesos por nodo * cores por proceso
#SBATCH --ntasks-per-node=1
### Cores por proceso (para MPI+OpenMP)
#SBATCH --cpus-per-task=64

### Tiempo de ejecucion. Formato dias-horas:minutos.
### short:  <= 1 hora
### multi:  <= 2 días
#SBATCH --time 1-23:59

#---------------------------------------------------

# Script que se ejecuta al arrancar el trabajo

# Cargar el entorno del usuario incluyendo la funcionalidad de modules
# No tocar
. /etc/profile

# Lanzar el programa
source ~/miniconda3/bin/activate
conda activate
srun tumorsphere simulate --prob-stem "0.65,0.66" --prob-diff "0" --realizations 16 --steps-per-realization 60 --rng-seed 3104902187430912364 --parallel-processes 32 --sql True --dat-files True --ovito False --output-dir data_sim_13