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

    input.recording_to_sort = 'testData/M1_D31_2018-11-01_12-28-25'
    input.firing_data_path = input.recording_to_sort +'/processed/' + setting.sorterName + '/sorter_curated.pkl'
    
    sorterPrefix = input.recording_to_sort+'/processed/'+setting.sorterName
    
    output.trial_figure = input.recording_to_sort + '/processed/Figures/trials.png'
    output.first_trial_ch = input.recording_to_sort + '/processed/Figures/trials_type1.png'
    output.second_trial_ch = input.recording_to_sort + '/processed/Figures/trials_type2.png'
    output.spike_data = input.recording_to_sort + '/processed/spatial_firing.hdf'


else:
    #in snakemake environment, the input and output will be provided by the workflow
    input = snakemake.input
    output = snakemake.output

#%% process firing times
session_id = input.recording_to_sort.split('/')[-1]
spike_data = process_firing_times2(session_id, input.firing_data_path, setting.session_type)

#%% save data
spike_data.to_hdf(output.spike_data,'spike_data')


