#!/bin/bash

### Correr con sbatch ./submit_simulation.sh
### En el directorio donde se encuentre este archivo
### tiene que haber otro directorio llamado 'data'.

#---------------------------------------------------

### Configuración del trabajo

### Nombre de la tarea
#SBATCH --job-name=tumoresfera_sim_15

### Cola a usar (gpu, mono, multi)
#SBATCH --partition=multi

### Cantidad de nodos a usar
#SBATCH --nodes=1

### Cores a utilizar por nodo = procesos por nodo * cores por proceso
#SBATCH --ntasks-per-node=1
### Cores por proceso (para MPI+OpenMP)
#SBATCH --cpus-per-task=24

### Tiempo de ejecucion. Formato dias-horas:minutos.
### short:  <= 1 hora
### multi:  <= 2 días
#SBATCH --time 0-6:00

#---------------------------------------------------

# Script que se ejecuta al arrancar el trabajo

# Cargar el entorno del usuario incluyendo la funcionalidad de modules
# No tocar
. /etc/profile

# Lanzar el programa
source ~/miniconda3/bin/activate
conda activate
srun tumorsphere simulate --prob-stem "0.77,0.78,0.79" --prob-diff "0" --realizations 24 --steps-per-realization 12 --rng-seed 1292317634567 --parallel-processes 24 --ovito False --dat-files False
