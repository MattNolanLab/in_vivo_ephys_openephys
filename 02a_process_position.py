# Process position data for open field recording

#%%
import pandas as pd
import testload
import PostSorting.post_process_sorted_data as ppsd
import PostSorting.open_field_sync_data as open_field_sync_data
from collections import namedtuple
import setting
from types import SimpleNamespace
import gc
import setting
import control_sorting_analysis
import hd_sampling_analysis

#%% define input and output
if 'snakemake' not in locals():
    #Define some variable to run the script standalone
    input = SimpleNamespace()
    output = SimpleNamespace()

    input.recording_to_sort = 'testData/M1_D27_2018-10-26_13-10-36_of/'

    sorterPrefix = input.recording_to_sort+'processed/'+setting.sorterName
    
    output.opto_pulse = input.recording_to_sort + 'processed/opto_pulse.pkl'
    output.hd_power_spectrum = input.recording_to_sort + 'processed/hd_power_spectrum.png'
    output.synced_spatial_data = input.recording_to_sort + 'processed/synced_spatial_data.hdf'
else:
    #in snakemake environment, the input and output will be provided by the workflow
    input = snakemake.input
    output = snakemake.output


#%% Read tags from folder
tags = control_sorting_analysis.get_tags_parameter_file(input.recording_to_sort)
unexpected_tag, interleaved_opto, delete_first_two_minutes, pixel_ratio = ppsd.process_running_parameter_tag(
    tags)


#%% Process opto channel
opto_on, opto_off, opto_tagging_start_index, is_found = ppsd.process_light_stimulation(input.recording_to_sort, output.opto_pulse)


#%% Process position data
spatial_data, position_was_found = ppsd.process_position_data(input.recording_to_sort, 'openfield')
hd_sampling_analysis.check_if_hd_sampling_was_high_enough(spatial_data,output.hd_power_spectrum)

#%% Synchronize with bonsai 
synced_spatial_data, total_length_sample_point, is_found = open_field_sync_data.process_sync_data(input.recording_to_sort, 
    spatial_data,opto_tagging_start_index)

#%% Save

synced_spatial_data.to_hdf( output.synced_spatial_data, 'synced_spatial_data',mode='w')
