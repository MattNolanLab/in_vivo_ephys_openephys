# Process firing for open field data

#%%
from PostSorting.load_firing_data import process_firing_times2
import PostSorting
import setting
from types import SimpleNamespace
from SnakeIOHelper import getSnake

#%%
if 'snakemake' not in locals(): 
    #Run the the file from the root project directory
    smk = getSnake('op_workflow.smk',[setting.debug_folder+'/processed/spatial_firing.hdf'],
        'process_firings' )
    sinput = smk.input
    soutput = smk.output
else:
    sinput = snakemake.input
    soutput = snakemake.output


#%% process firing times
session_id = sinput.recording_to_sort.split('/')[-1]
spike_data = process_firing_times2(session_id, sinput.sorted_data_path, setting.session_type)

#%% save
spike_data.to_hdf(soutput.spatial_firing,'spatial_firing',mode='w')


#%%
