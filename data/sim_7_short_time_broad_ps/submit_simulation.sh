#!/bin/bash

### correr con sbatch ./submit_simulation.sh

#---------------------------------------------------

### Configuración del trabajo

### Nombre de la tarea
#SBATCH --job-name=tumoresfera

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
#SBATCH --time 1-23:00

#---------------------------------------------------

# Script que se ejecuta al arrancar el trabajo

# Cargar el entorno del usuario incluyendo la funcionalidad de modules
# No tocar
. /etc/profile

# Lanzar el programa
source ~/miniconda3/bin/activate
conda activate
srun python -m tumorsphere.cli --prob-stem "0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7" --prob-diff "0" --realizations 8 --steps-per-realization 60 --rng-seed 1292317634567 --parallel-processes 64
