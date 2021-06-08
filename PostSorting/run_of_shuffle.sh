#!/bin/sh
#$ -cwd
#$ -l h_rt=00:29:00
#$ -l h_vmem=3G
#$ -pe sharedmem 1
#$ -P sbms_kg_grid_modelling

# Load Python
module load anaconda

# activate the virtual environment
source activate myenv

# append the python path
export PYTHONPATH="/home/s1228823/in_vivo_ephys_openephys"

# Run the program
python /home/s1228823/in_vivo_ephys_openephys/PostSorting/shuffle_analysis.py