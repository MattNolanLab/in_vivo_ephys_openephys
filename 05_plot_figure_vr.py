#%%
import PostSorting.vr_make_plots
import PostSorting.make_plots
import setting
import pandas as pd
from collections import namedtuple
from types import SimpleNamespace
import SnakeIOHelper 

#%% define input and output
# if 'snakemake' not in locals(): 
#     smk = getSnake('vr_workflow.smk',['testData/M1_D31_2018-11-01_12-28-25_short/processed/plot_figure_done.txt'],
#         'plot_figures' )
#     sinput = smk.input
#     soutput = smk.output
# else:
#     sinput = snakemake.input
#     soutput = snakemake.output


(sinput, soutput) = SnakeIOHelper.getSnake(locals(), 'vr_workflow.smk', [setting.debug_folder+'/processed/completed.txt'],
    'plot_figures')
#%% Load data
spike_data = pd.read_pickle(sinput.spatial_firing_vr)
raw_position_data =pd.read_pickle(sinput.raw_position)
processed_position_data = pd.read_pickle(sinput.processed_position_data)

#%%
# PostSorting.make_plots.plot_waveforms(spike_data, prm)
PostSorting.make_plots.plot_spike_histogram(spike_data, soutput.spike_histogram)
PostSorting.make_plots.plot_autocorrelograms(spike_data, soutput.autocorrelogram)
PostSorting.vr_make_plots.plot_spikes_on_track(spike_data,raw_position_data, processed_position_data, soutput.spike_trajectories, prefix='_movement')
PostSorting.vr_make_plots.plot_firing_rate_maps(spike_data, soutput.spike_rate, prefix='_all')
#%%
PostSorting.vr_make_plots.plot_convolved_rates_in_time(spike_data, soutput.convolved_rate)

#%%
with open(soutput.result,'w') as f:
    f.write('Completed!')
    