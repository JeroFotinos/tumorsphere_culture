#!/bin/bash
#$ -q seq_long
#$ -N val_nombre
#$ -o log/val_nombre
#$ -e log/val_nombre
#$ -l m_mem_free=1G
#$ -cwd


module load python


python3 val_nombre.py



