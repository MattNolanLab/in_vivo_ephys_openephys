# Process position data for open field recording

#%%
import pandas as pd
import PostSorting.post_process_sorted_data as ppsd
import PostSorting.open_field_sync_data as open_field_sync_data
from collections import namedtuple
import setting
import gc
import control_sorting_analysis
import hd_sampling_analysis
import SnakeIOHelper
import PostSorting.open_field_light_data as open_field_light_data
from file_utility import get_tags_parameter_file

#%% define input and outpu
(sinput, soutput) = SnakeIOHelper.getSnake(locals(), 'workflow_of.smk', [setting.debug_folder+'/processed/opto_pulse.pkl'],
    'process_position')

#%% Read tags from folder
tags = get_tags_parameter_file(sinput.recording_to_sort)
unexpected_tag, interleaved_opto, delete_first_two_minutes, pixel_ratio = ppsd.process_running_parameter_tag(
    tags)
    
#%% Process opto channel
opto_on, opto_off, is_found, opto_tagging_start_index = open_field_light_data.process_opto_data(sinput.recording_to_sort)  # indices
if is_found:
    opto_data_frame = open_field_light_data.make_opto_data_frame(opto_on)
    opto_data_frame.to_pickle(soutput.opto_pulse)
else:
    df = pd.DataFrame()
    df.to_pickle(soutput.opto_pulse)

#%% Process position data
spatial_data, position_was_found = ppsd.process_position_data(sinput.recording_to_sort, 'openfield')
hd_sampling_analysis.check_if_hd_sampling_was_high_enough(spatial_data,soutput.hd_power_spectrum)

#%% Synchronize with bonsai 
synced_spatial_data, total_length_sample_point, ephys_sync_data, is_found = open_field_sync_data.process_sync_data(sinput.recording_to_sort, 
    spatial_data,opto_tagging_start_index)

#%% plot sync pulse to check
open_field_sync_data.plot_sync_pulse(synced_spatial_data, ephys_sync_data, soutput.sync_pulse)

#%% Save

synced_spatial_data.to_pickle(soutput.synced_spatial_data)
