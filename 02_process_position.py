#%%
import pandas as pd
import testload
from PostSorting.vr_sync_spatial_data import *
from PostSorting.vr_spatial_data import *
import PostSorting
from collections import namedtuple
import setting
from types import SimpleNamespace
#%% define input and output
if 'snakemake' not in locals():
    #Define some variable to run the script standalone
    input = SimpleNamespace()
    output = SimpleNamespace()

    input.recording_to_sort = 'testData/M1_D31_2018-11-01_12-28-25'
    
    sorterPrefix = input.recording_to_sort+'/processed/'+setting.sorterName
    
    output.trial_figure = input.recording_to_sort + '/processed/figures/trials.png'
    output.first_trial_ch = input.recording_to_sort + '/processed/figures/trials_type1.png'
    output.second_trial_ch = input.recording_to_sort + '/processed/figures/trials_type2.png'
    output.raw_position_data = input.recording_to_sort +'/processed/raw_position.hdf'
    output.processed_position_data = input.recording_to_sort +'/processed/processed_position.hdf'


else:
    #in snakemake environment, the input and output will be provided by the workflow
    input = snakemake.input
    output = snakemake.output


#%% Synchronize position data and calculate the trial-related data
raw_position_data = pd.DataFrame()

raw_position_data = calculate_track_location(raw_position_data, input.recording_to_sort)

raw_position_data = calculate_trial_numbers(raw_position_data)
PostSorting.vr_make_plots.plot_trials(raw_position_data['trial_number'], output.trial_figure)

(raw_position_data,first_ch,second_ch) = calculate_trial_types(raw_position_data, input.recording_to_sort)
PostSorting.vr_make_plots.plot_trial_channels(first_ch, second_ch, output.first_trial_ch, output.second_trial_ch)

raw_position_data = calculate_time(raw_position_data)
raw_position_data = calculate_instant_dwell_time(raw_position_data)
raw_position_data = calculate_instant_velocity(raw_position_data)
raw_position_data = get_avg_speed_200ms(raw_position_data)

#%% save data
raw_position_data.to_hdf(output.raw_position_data, 'raw_position_data', mode='w')

#%% bin the position data
processed_position_data = pd.DataFrame() # make dataframe for processed position data
processed_position_data = bin_data_over_trials(raw_position_data,processed_position_data)
processed_position_data = bin_data_trial_by_trial(raw_position_data,processed_position_data)
processed_position_data = calculate_total_trial_numbers(raw_position_data, processed_position_data)
processed_position_data = PostSorting.vr_stop_analysis.generate_stop_lists(raw_position_data, processed_position_data)

#%% save data
processed_position_data.to_hdf(output.processed_position_data,'processed_position_data', mode='w')

