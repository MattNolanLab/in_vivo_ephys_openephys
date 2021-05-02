'''
Create binned firing rate and position info in time
'''

#%%
import scipy.signal as signal
import SnakeIOHelper
import settings
import pandas as pd 
from PostSorting import glmneuron
#%%

(sinput, soutput) = SnakeIOHelper.getSnake(locals(), 'workflow/workflow_vr.smk', [settings.debug_folder+'/processed/binned_firing.pkl'],
    'bin_firing')

# %%
raw_position = pd.read_pickle(sinput.raw_position)
spatial_firing_vr = pd.read_pickle(sinput.spatial_firing_vr)
# processed_position = pd.read_pickle(sinput.processed_position_data)
# %%

glmneuron.getSpikePopulationXR(spatial_firing_vr.firing_times, settings.sampling_rate)
# %%
