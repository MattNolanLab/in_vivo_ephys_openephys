"""Run the preprocessing code
"""

#%% imports

import pipieline_manager 
from PreClustering import pre_process_ephys_data
import Logger
import setting
import PreClustering.pre_process_ephys_data as pre_process_ephys_data
import PreClustering
import sys
import parameters
from collections import namedtuple
import file_utility
import os

#%% define input and output

if 'snakemake' not in locals():
    #Define some variable to run the script standalone
    input = namedtuple
    output = namedtuple
    input.recording_to_sort = 'testData/M1_D31_2018-11-01_12-28-25'
    output.spatial_firing = 'testData/processed/spatial_firing.pkl'
    output.noisy_clusters = 'testData/processed/noisy_clusters.pkl'
    output.raw_position_data = 'testData/processed/position_Data.pkl'
else:
    #in snakemake environment, the input and output will be provided by the workflow
    input = snakemake.input
    output = snakemake.output

#%% Determine the recording and output folder, initiate some book-keep stuffs

location_on_server = pipieline_manager.get_location_on_server(input.recording_to_sort )
tags = pipieline_manager.get_tags_parameter_file(input.recording_to_sort )

os.makedirs(input.recording_to_sort+'/log',exist_ok=True)
sys.stdout = Logger.Logger(input.recording_to_sort+'/log/sorting_log.txt')

prm = parameters.Parameters()
prm.set_date(input.recording_to_sort.rsplit('/', 2)[-2])
prm.set_filepath(input.recording_to_sort)
file_utility.set_continuous_data_path(prm)
PreClustering.dead_channels.get_dead_channel_ids(prm)  # read dead_channels.txt

