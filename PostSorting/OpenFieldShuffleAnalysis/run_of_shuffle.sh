#!/bin/sh
#$ -cwd
#$ -l h_rt=47:59:00
#$ -l h_vmem=4G
#$ -pe sharedmem 1
#$ -P sbms_kg_grid_modelling

#load the environment modules
. /etc/profile.d/modules.sh

# Load Python
module load anaconda

# activate the virtual environment
source activate py38

# append the python path
export PYTHONPATH="/home/s1228823/in_vivo_ephys_openephys"

# Run the program
python /home/s1228823/in_vivo_ephys_openephys/PostSorting/OpenFieldShuffleAnalysis/shuffle_analysis.py