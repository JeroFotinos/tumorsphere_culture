#!/bin/bash

### Run with sbatch ./merge_dbs_and_make_df_Slurm.sh

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
#SBATCH --time 1-23:00

#---------------------------------------------------

# Script executed when the job starts

# Loading environment and modules (do not modify)
. /etc/profile

# Directory with output files
base_dir="$HOME/data_205761.rome06.rome.ccad.unc.edu.ar"

# Directory with .db files
dbs_folder="$base_dir/db_files"

# Directory with .dat files
dat_folder="$base_dir/dat_files"

# Path and name of the merged `.db` file to append to or create
merging_path="$base_dir/sim_17.db"

# Path and name of the data base to read
db_path="$base_dir/sim_17.db"

# Path and name of the csv file to write
# csv_path="$base_dir/sim_17.csv"
csv_path="$HOME/tumoresfera/sim_17/sim_17.csv"

# Activate the conda environment
source ~/miniconda3/bin/activate
conda activate sim_17_venv

# Logging start time
echo "$(date +"%H:%M:%S"): Calling srun from $(hostname)"

# # Merging databases
# echo "$(date +"%H:%M:%S"): Merge databases"
# srun tumorsphere mergedbs --dbs-folder "$dbs_folder" --merging-path "$merging_path"
# echo "$(date +"%H:%M:%S"): Databases merged"


# Making the DataFrame from .dat files
echo "$(date +"%H:%M:%S"): Making the DataFrame from .dat files"
srun tumorsphere makedf --dat-folder "$dat_folder" --csv-path "$base_dir/sim_17_from_dat.csv" --dat-files True
echo "$(date +"%H:%M:%S"): DataFrame made"


# Making the DataFrame from the merged database
echo "$(date +"%H:%M:%S"): Making the DataFrame from the merged database"
srun tumorsphere makedf --db-path "$db_path" --csv-path "$csv_path" --dat-files False
echo "$(date +"%H:%M:%S"): DataFrame made"

