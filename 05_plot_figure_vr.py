#%%

import PostSorting.vr_make_plots
import PostSorting.make_plots
import setting
import pandas as pd
from collections import namedtuple
from types import SimpleNamespace
import SnakeIOHelper
#%% define input and output

figure_folder ='/processed/figures'

if 'snakemake' not in locals():
    #Define some variable to run the script standalone
    input = SimpleNamespace()
    output = SimpleNamespace()

    input.recording_to_sort = 'testData/M1_D31_2018-11-01_12-28-25'
    input.raw_position = input.recording_to_sort + '/processed/raw_position.hdf'
    input.processed_position_data = input.recording_to_sort + '/processed/processed_position.hdf'
    input.spatial_firing_vr = input.recording_to_sort + '/processed/spatial_firing_vr.hdf'
    
    output.spike_histogram = input.recording_to_sort + figure_folder + '/behaviour/spike_histogram/'
    output.autocorrelogram = input.recording_to_sort + figure_folder + '/behaviour/autocorrelogram/'
    output.spike_trajectories = input.recording_to_sort + figure_folder + '/behaviour/spike_trajectories/'
    output.spike_rate =  input.recording_to_sort + figure_folder + '/behaviour/spike_rate/'
    output.result = input.recording_to_sort +'/processed/results.txt'

    SnakeIOHelper.makeFolders(output)
else:
    #in snakemake environment, the input and output will be provided by the workflow
    input = snakemake.input
    output = snakemake.output

#%% Load data
spike_data = pd.read_hdf(input.spatial_firing_vr)
raw_position_data =pd.read_hdf(input.raw_position)
processed_position_data = pd.read_hdf(input.processed_position_data)

#%%
# PostSorting.make_plots.plot_waveforms(spike_data, prm)
PostSorting.make_plots.plot_spike_histogram(spike_data, output.spike_histogram)

#%%
PostSorting.make_plots.plot_autocorrelograms(spike_data, output.autocorrelogram)

#%%
PostSorting.vr_make_plots.plot_spikes_on_track(spike_data,raw_position_data, processed_position_data, output.spike_trajectories, prefix='_movement')

#%%
PostSorting.vr_make_plots.plot_firing_rate_maps(spike_data, output.spike_rate, prefix='_all')

#%%
with open(output.result,'w') as f:
    f.write('Completed!')
    


#%%
