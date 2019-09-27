import os

def getRecording2sort(base_folder):
    folder = []
    with os.scandir(base_folder) as it:
        for entry in it:
            if entry.is_dir():
                folder.append(entry.path)

    return folder

recordings = getRecording2sort('testData')

rule all:
    input:
        result=expand('{recording}/processed/results.txt',recording=recordings)

rule plot_figures:
    input:
        raw_position = '{recording}/processed/raw_position.hdf',
        processed_position_data =  '{recording}/processed/processed_position.hdf',
        spatial_firing_vr = '{recording}/processed/spatial_firing_vr.hdf'
    
    output:
        spike_histogram = '{recording}/processed/figures/behaviour/spike_histogram/',
        autocorrelogram = '{recording}/processed/behaviour/autocorrelogram/',
        spike_trajectories = '{recording}/processed//behaviour/spike_trajectories/',
        spike_rate =  '{recording}/processed//behaviour/spike_rate/',
        result = '{recording}/processed/results.txt'
    script:
        '05_plot_figure_vr.py'
    
    