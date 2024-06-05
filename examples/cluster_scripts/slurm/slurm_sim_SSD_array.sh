#!/bin/bash

### Run with sbatch ./submit_simulation_serafin_SSD_array.sh

#---------------------------------------------------

### Job configuration

#SBATCH --mail-type=ALL    
#SBATCH --mail-user=jerofoti@gmail.com

### Job name
#SBATCH --job-name=tumoresfera

### Queue to be used (gpu, mono, multi)
#SBATCH --partition=multi

### Amount of nodes to use
#SBATCH --nodes=1

### Cores per node = processes per node * cores per process
#SBATCH --ntasks-per-node=1
### Cores per process (para MPI+OpenMP)
#SBATCH --cpus-per-task=64

### Execution time. Format: days-hours:minutes
### short:  <= 1 hora
### multi:  <= 2 días
#SBATCH --time 0-4:30

### Array job configuration
#SBATCH --array=1-9


#---------------------------------------------------

# Script executed when the job starts

# Loading environment and modules (do not modify)
. /etc/profile

# Scratch directory for temporary files (unique for each array task)
scratch_dir="/scratch/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$scratch_dir"
echo "Created scratch_dir: $scratch_dir"

# Creating a new directory in the user's home to store the results (unique for each array task)
final_output_dir="$HOME/data_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$final_output_dir"
echo "Created final_output_dir: $final_output_dir"

# Mapping SLURM_ARRAY_TASK_ID to the desired prob-stem value
# prob_stem=$(echo "scale=1; 0.1 * ${SLURM_ARRAY_TASK_ID}" | bc)
prob_stem="0.$SLURM_ARRAY_TASK_ID"
echo "prob_stem: $prob_stem"

# Activate the conda environment
source ~/miniconda3/bin/activate
conda activate sim_17_venv

# Logging start time
echo "$(date +"%H:%M:%S"): Calling srun from $(hostname), array=$SLURM_ARRAY_TASK_ID"

# Call srun to execute the actual simulation
srun tumorsphere simulate --prob-stem "$prob_stem" --prob-diff "0" --realizations 64 --steps-per-realization 60 --rng-seed 1292317634567 --parallel-processes 64 --sql True --dat-files True --ovito False --output-dir "$scratch_dir"

# Save results after srun completes (copying from scratch to user's home)
echo "$(date +"%H:%M:%S"): Saving results"
sgather -rp "$scratch_dir" "$final_output_dir"
echo "$(date +"%H:%M:%S"): Results saved"
