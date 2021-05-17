#%%
from PostSorting.load_firing_data import process_firing_times2
import PostSorting
import settings
from types import SimpleNamespace
from utils import SnakeIOHelper

#%%
(sinput, soutput) = SnakeIOHelper.getSnake(locals(), 'workflow/workflow_vr.smk', [settings.debug_folder+'/processed/spatial_firing.pkl'],
    'process_firings')

#%% process firing times
session_id = sinput.recording_to_sort.split('/')[-1]
spike_data = process_firing_times2(session_id, sinput.sorted_data_path, settings.session_type)

#%% save data
spike_data.to_pickle(soutput.spike_data)


# %%
