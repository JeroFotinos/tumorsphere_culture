#!/bin/bash

### Correr con sbatch ./merge_dbs_and_make_dfs.sh
### En el directorio donde se encuentre este archivo
### tiene que haber otro directorio llamado 'data'.

#---------------------------------------------------

### Configuración del trabajo

### Nombre de la tarea
#SBATCH --job-name=merge_dbs_and_make_dfs

### Cola a usar (gpu, mono, multi)
#SBATCH --partition=multi

### Cantidad de nodos a usar
#SBATCH --nodes=1

### Cores a utilizar por nodo = procesos por nodo * cores por proceso
#SBATCH --ntasks-per-node=1
### Cores por proceso (para MPI+OpenMP)
#SBATCH --cpus-per-task=1

### Tiempo de ejecucion. Formato dias-horas:minutos.
### short:  <= 1 hora
### multi:  <= 2 días
#SBATCH --time 1-00:00

#---------------------------------------------------

# Script que se ejecuta al arrancar el trabajo

# Cargar el entorno del usuario incluyendo la funcionalidad de modules
# No tocar
. /etc/profile

# Lanzar el programa
source ~/miniconda3/bin/activate
conda activate
srun tumorsphere mergedbs --dbs-folder ./data --merging-path ./sim_12.db
srun tumorsphere makedf --db-path ./sim_12.db --csv-path ./sim_12.csv
