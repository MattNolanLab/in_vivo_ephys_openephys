import os
import setting

def getRecording2sort(base_folder):
    folder = []
    with os.scandir(base_folder) as it:
        for entry in it:
            if entry.is_dir():
                folder.append(entry.path)

    return folder

def optionalFile(path):
    print(path)
    if os.path.exists(path):
        return path
    else:
        return [None]

recordings = getRecording2sort('/home/ubuntu/to_sort/recordings')
figure_prefix = '{recording}/processed/figures'
sorterPrefix = '{recording}/processed/'+setting.sorterName


rule all:
    input:
        result=expand('{recording}/processed/snakemake.done',recording=recordings)


include: 'workflow_common.smk'

rule process_position:
    input:
        recording_to_sort = '{recording}'
    output:    
        opto_pulse = '{recording}/processed/opto_pulse.pkl',
        synced_spatial_data = '{recording}/processed/synced_spatial_data.pkl',
        sync_pulse = '{recording}/processed/sync_pulse.png'
    script:
        '../scripts/openfield/02a_process_position.py'

rule process_firings:
    input:
        recording_to_sort = '{recording}',
        sorted_data_path = '{recording}/processed/'+setting.sorterName+'/sorter_curated_df.pkl'
    output:
        spatial_firing = '{recording}/processed/spatial_firing.pkl'
    script:
        '../scripts/openfield/03a_process_firings.py'

rule process_expt:
    input:
        spatial_firing = '{recording}/processed/spatial_firing.pkl',
        position = '{recording}/processed/synced_spatial_data.pkl'
    output:
        spatial_firing_of = '{recording}/processed/spatial_firing_of.pkl',
        position_heat_map = '{recording}/processed/position_heat_map.pkl',
        hd_histogram = '{recording}/processed/hd_histogram.pkl',
        hd_csv = directory('{recording}/processed/firing_fields/')
    script:
        '../scripts/openfield/04a_process_expt_openfield.py'


rule plot_figures:
    input:
        spatial_firing = '{recording}/processed/spatial_firing_of.pkl',
        position = '{recording}/processed/synced_spatial_data.pkl',
        position_heat_map = '{recording}/processed/position_heat_map.pkl',
        hd_histogram = '{recording}/processed/hd_histogram.pkl',
        waveform_figure_curated = sorterPrefix + '/waveform/curated/',
    output:
        spike_histogram = directory(figure_prefix+'/spike_histogram/'),
        autocorrelogram = directory(figure_prefix+'/autocorrelogram/'),
        convolved_rate = directory(figure_prefix+'/ConvolvedRates_InTime/'),
        speed_histogram = directory(figure_prefix+'/speed_histogram/'),
        firing_rate_vs_speed = directory(figure_prefix+'/firing_rate_vs_speed/'),
        firing_scatter = directory(figure_prefix+'/firing_scatters/'),
        rate_maps = directory(figure_prefix+'/rate_maps/'),
        rate_map_autocorrelogram = directory(figure_prefix+'/rate_map_autocorrelogram/'),
        head_direction_plots_2d = directory(figure_prefix+'/head_direction_plots_2d/'),
        head_direction_plots_polar = directory(figure_prefix+'/head_direction_plots_polar/'),
        firing_field_plots =  directory(figure_prefix+'/firing_field_plots/'),
        firing_field_head_direction = directory(figure_prefix+'/firing_field_head_direction/'),
        firing_field_head_direction_raw = directory(figure_prefix+'/firing_field_head_direction_raw/'),
        firing_fields_coloured_spikes = directory(figure_prefix+'/firing_fields_coloured_spikes/'),
        combined = directory(figure_prefix+'/combined/'),
        coverage_map = figure_prefix+'/session/heatmap.png',
        result = touch('{recording}/processed/snakemake.done')
    script:
        '../scripts/openfield/05a_plot_figure_openfield.py'