import os
import setting

def getRecording2sort(base_folder):
    folder = []
    with os.scandir(base_folder) as it:
        for entry in it:
            if entry.is_dir():
                folder.append(entry.name)

    return folder

recordings = getRecording2sort('/home/ubuntu/to_sort/recordings')
sorterPrefix = '{recording}/processed/'+setting.sorterName

# rule all:
#     input:
#         result=expand('{recording}/complete',recording=recordings)

include: 'workflow_common.smk'


rule process_position:
    input:
        recording_to_sort = '{recording}'
    
    output:
        trial_figure = '{recording}/processed/figures/trials.png',
        first_trial_ch = '{recording}/processed/figures/trials_type1.png',
        second_trial_ch = '{recording}/processed/figures/trials_type2.png',
        raw_position_data = '{recording}/processed/raw_position.pkl',
        processed_position_data = '{recording}/processed/processed_position.pkl',
        stop_raster = '{recording}/processed/figures/behaviour/stop_raster.png',
        stop_histogram = '{recording}/processed/figures/behaviour/stop_histogram.png',
        speed_histogram = '{recording}/processed/figures/behaviour/speed_histogram.png'
    script:
        '02_process_position.py'


rule process_firings:
    input:
        recording_to_sort = '{recording}',
        sorted_data_path = '{recording}/processed/'+setting.sorterName+'/sorter_curated_df.pkl'
    output:
        spike_data ='{recording}/processed/spatial_firing.pkl'
    script:
        '03_process_firings.py'

rule process_expt:
    input:
        spatial_firing = '{recording}/processed/spatial_firing.pkl',
        raw_position = '{recording}/processed/raw_position.pkl',
        processed_position_data = '{recording}/processed/processed_position.pkl'
    output:
        spatial_firing_vr = '{recording}/processed/spatial_firing_vr.pkl',
        cluster_spike_plot = directory('{recording}/processed/figures/spike_number/'),
        spike_data = directory('{recording}/processed/figures/spike_data/'),
    script:
        '04_process_expt_vr.py'


rule plot_figures:
    input:
        raw_position = '{recording}/processed/raw_position.pkl',
        processed_position_data =  '{recording}/processed/processed_position.pkl',
        spatial_firing_vr = '{recording}/processed/spatial_firing_vr.pkl'
    
    output:
        spike_histogram = directory('{recording}/processed/figures/spike_histogram/'),
        autocorrelogram = directory('{recording}/processed/figures/autocorrelogram/'),
        spike_trajectories = directory('{recording}/processed/figures/spike_trajectories/'),
        spike_rate =  directory('{recording}/processed/figures/spike_rate/'),
        convolved_rate = directory('{recording}/processed/figures/ConvolvedRates_InTime/'),
        result = touch('{recording}/processed/snakemake.done')
    script:
        '05_plot_figure_vr.py'

