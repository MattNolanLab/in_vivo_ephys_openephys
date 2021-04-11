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
        raw_position_data = '{recording}/processed/raw_position.pkl',
        processed_position_data = '{recording}/processed/processed_position.pkl',
        stop_raster = '{recording}/processed/figures/behaviour/stop_raster.png',
        trial_type_plot_folder = directory('{recording}/processed/figures/trials'),
        stop_histogram = '{recording}/processed/figures/behaviour/stop_histogram.png',
        speed_histogram = '{recording}/processed/figures/behaviour/speed_histogram.png',
        speed_plot = '{recording}/processed/figures/behaviour/speed.png',
        mean_speed_plot = '{recording}/processed/figures/behaviour/speed_mean.png',
        raw_position_plot = '{recording}/processed/figures/position.png',
        speed_heat_map = '{recording}/processed/figures/behaviour/speed_heat_map.png'
    script:
        '../scripts/vr/02_process_position.py'


rule process_firings:
    input:
        recording_to_sort = '{recording}',
        sorted_data_path = '{recording}/processed/'+setting.sorterName+'/sorter_curated_df.pkl'
    output:
        spike_data ='{recording}/processed/spatial_firing.pkl'
    script:
        '../scripts/vr/03_process_firings.py'

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
        '../scripts/vr/04_process_expt_vr.py'


rule plot_figures:
    input:
        raw_position = '{recording}/processed/raw_position.pkl',
        processed_position_data =  '{recording}/processed/processed_position.pkl',
        spatial_firing_vr = '{recording}/processed/spatial_firing_vr.pkl',
        waveform_figure_curated = sorterPrefix + '/waveform/curated/',
        stop_histogram = '{recording}/processed/figures/behaviour/stop_histogram.png',
        speed_histogram = '{recording}/processed/figures/behaviour/speed_histogram.png',
        stop_raster = '{recording}/processed/figures/behaviour/stop_raster.png'
    output:
        spike_histogram = directory('{recording}/processed/figures/spike_histogram/'),
        autocorrelogram = directory('{recording}/processed/figures/autocorrelogram/'),
        spike_trajectories = directory('{recording}/processed/figures/spike_trajectories/'),
        spike_rate =  directory('{recording}/processed/figures/spike_rate/'),
        combined = directory('{recording}/processed/figures/combined/'),
        result = touch('{recording}/processed/snakemake.done')
    script:
        '../scripts/vr/05_plot_figure_vr.py'

