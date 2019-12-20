import os
import setting

def getRecording2sort(base_folder):
    folder = []
    with os.scandir(base_folder) as it:
        for entry in it:
            if entry.is_dir():
                folder.append(entry.path)

    return folder

recordings = getRecording2sort('testData')
sorterPrefix = '{recording}/processed/'+setting.sorterName

rule all:
    input:
        result=expand('{recording}/processed/results.txt',recording=recordings)

rule sort_spikes:
    input:
        probe_file = 'sorting_files/tetrode_16.prb',
        sort_param = 'sorting_files/params.json',
        tetrode_geom = 'sorting_files/geom_all_tetrodes_original.csv',
        recording_to_sort = '{recording}',
        dead_channel = '{recording}/dead_channels.txt'
    
    output:
        sorter = sorterPrefix +'/sorter.pkl',
        sorter_df = sorterPrefix +'/sorter_df.pkl',
        sorter_curated = sorterPrefix +'/sorter_curated.pkl',
        sorter_curated_df = sorterPrefix +'/sorter_curated_df.pkl',
        waveform_figure = directory(sorterPrefix + '/waveform/')
    script:
        '01_sorting.py'


rule process_position:
    input:
        recording_to_sort = '{recording}'
    
    output:
        trial_figure = '{recording}/processed/figures/trials.png',
        first_trial_ch = '{recording}/processed/figures/trials_type1.png',
        second_trial_ch = '{recording}/processed/figures/trials_type2.png',
        raw_position_data = '{recording}/processed/raw_position.hdf',
        processed_position_data = '{recording}/processed/processed_position.hdf',
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
        spike_data ='{recording}/processed/spatial_firing.hdf'
    script:
        'scripts/03_process_firings.py'

rule process_expt:
    input:
        spatial_firing = '{recording}/processed/spatial_firing.hdf',
        raw_position = '{recording}/processed/raw_position.hdf',
        processed_position_data = '{recording}/processed/processed_position.hdf'
    output:
        spatial_firing_vr = '{recording}/processed/spatial_firing_vr.hdf',
        cluster_spike_plot = directory('{recording}/processed/figures/spike_number/'),
        spike_data = directory('{recording}/processed/figures/spike_data/'),
    script:
        '04_process_expt_vr.py'


rule plot_figures:
    input:
        raw_position = '{recording}/processed/raw_position.hdf',
        processed_position_data =  '{recording}/processed/processed_position.hdf',
        spatial_firing_vr = '{recording}/processed/spatial_firing_vr.hdf'
    
    output:
        spike_histogram = directory('{recording}/processed/figures/spike_histogram/'),
        autocorrelogram = directory('{recording}/processed/figures/autocorrelogram/'),
        spike_trajectories = directory('{recording}/processed/figures/spike_trajectories/'),
        spike_rate =  directory('{recording}/processed/figures/spike_rate/'),
        convolved_rate = directory('{recording}/processed/figures/ConvolvedRates_InTime/'),
        result = '{recording}/processed/completed.txt'
    script:
        '05_plot_figure_vr.py'