#%%
from matplotlib.pyplot import plot
import PostSorting.vr_make_plots
import PostSorting.make_plots
import setting
import pandas as pd
from collections import namedtuple
from types import SimpleNamespace
import SnakeIOHelper 

#%% define input and output
(sinput, soutput) = SnakeIOHelper.getSnake(locals(), 'workflow/workflow_vr.smk', [setting.debug_folder+'/processed/snakemake.done'],
    'plot_figures')
#%% Load data
spike_data = pd.read_pickle(sinput.spatial_firing_vr)
raw_position_data =pd.read_pickle(sinput.raw_position)
processed_position_data = pd.read_pickle(sinput.processed_position_data)

#%%
PostSorting.make_plots.plot_spike_histogram(spike_data, soutput.spike_histogram)
PostSorting.make_plots.plot_autocorrelograms(spike_data, soutput.autocorrelogram)
PostSorting.vr_make_plots.plot_spikes_on_track(spike_data, processed_position_data, soutput.spike_trajectories)
PostSorting.vr_make_plots.plot_firing_rate_maps(spike_data, processed_position_data, soutput.spike_rate, plot_sem=False)

# %%
