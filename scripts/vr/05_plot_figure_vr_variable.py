#%%
from matplotlib.pyplot import plot
import PostSorting.vr_make_plots
import PostSorting.make_plots
import settings
import pandas as pd
from collections import namedtuple
from types import SimpleNamespace
from utils import SnakeIOHelper
import PostSorting.open_field_make_plots as open_field_make_plots

#%% define input and output
(sinput, soutput) = SnakeIOHelper.getSnake(locals(), 'workflow/workflow_vr.smk', [settings.debug_folder+'/processed/snakemake.done'],
    'plot_figures')
    
#%% Load data
spike_data = pd.read_pickle(sinput.spatial_firing_vr)
raw_position_data =pd.read_pickle(sinput.raw_position)
processed_position_data = pd.read_pickle(sinput.processed_position_data)
try:
    track_length = spike_data.track_length[0]
except IndexError:
    # in case there is no cell
    track_length = 0

#%%
PostSorting.make_plots.plot_spike_histogram(spike_data, soutput.spike_histogram)
PostSorting.make_plots.plot_autocorrelograms(spike_data, soutput.autocorrelogram)
PostSorting.vr_make_plots.plot_spikes_on_track(spike_data, processed_position_data, soutput.spike_trajectories,track_length = track_length)
PostSorting.vr_make_plots.plot_firing_rate_maps(spike_data, processed_position_data, soutput.spike_rate, plot_sem=False, track_length = track_length)

# %% combine figures

folder_list = [
    sinput.waveform_figure_curated,
    soutput.autocorrelogram,
    soutput.spike_histogram,
    soutput.spike_trajectories,
    soutput.spike_rate
]

common_figures =[
    sinput.stop_raster,
    sinput.stop_histogram,
    sinput.speed_histogram,
]

open_field_make_plots.make_combined_figures_auto(folder_list, common_figures, [], soutput.combined,spike_data)

# %%
