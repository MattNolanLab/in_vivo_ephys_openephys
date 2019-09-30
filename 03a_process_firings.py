# Process firing for open field data

#%%
from PostSorting.load_firing_data import process_firing_times2
import PostSorting
import setting
from types import SimpleNamespace

#%% define input and output
if 'snakemake' not in locals():
    #Define some variable to run the script standalone
    input = SimpleNamespace()
    output = SimpleNamespace()

    input.recording_to_sort = 'testData/M1_D27_2018-10-26_13-10-36_of/'
    input.sorted_data_path = input.recording_to_sort +'processed/' + setting.sorterName + '/sorter_curated_df.pkl'
    
    sorterPrefix = input.recording_to_sort+'/processed/'+setting.sorterName
    
    output.spatial_firing = input.recording_to_sort + '/processed/spatial_firing.hdf'


else:
    #in snakemake environment, the input and output will be provided by the workflow
    input = snakemake.input
    output = snakemake.output

#%% process firing times
session_id = input.recording_to_sort.split('/')[-2]
spike_data = process_firing_times2(session_id, input.sorted_data_path, setting.session_type)

#%% save
spike_data.to_hdf(output.spatial_firing,'spatial_firing',mode='w')



#%%
