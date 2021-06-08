#!/bin/sh
$ -cwd
$ -l h_rt=00:20:00
$ -l h_vmem=3G
$ -pe sharedmem 1
$ -P sbms_kg_grid_modelling

# Load Python
module load anaconda

# activate the virtual environment
source activate myenv

# append the python path
export PYTHONPATH="/home/s1228823/in_vivo_ephys_openephys"

# Run the program
python /mnt/datastore/Harry/test_recordings /home/s1228823/in_vivo_ephys_openephys/PostSorting/of_create_shuffles.py