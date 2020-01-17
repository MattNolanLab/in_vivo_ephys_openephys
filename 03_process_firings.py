#%%
from PostSorting.load_firing_data import process_firing_times2
import PostSorting
import setting
from types import SimpleNamespace
import SnakeIOHelper 

#%%
(sinput, soutput) = SnakeIOHelper.getSnake(locals(), 'vr_workflow.smk', [setting.debug_folder+'/processed/spatial_firing.pkl'],
    'process_firings')

#%% process firing times
session_id = sinput.recording_to_sort.split('/')[-1]
spike_data = process_firing_times2(session_id, sinput.sorted_data_path, setting.session_type)

#%% save data
spike_data.to_pickle(soutput.spike_data)

