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
sorterPrefix = '{recording}/processed/'+setting.sorterName
figure_prefix = '{recording}/processed/figures'


rule all:
    input:
        result=expand('{recording}/processed/snakemake.done',recording=recordings)


include: 'workflow_common.smk'

rule curate_clusters:
    input:
        sorter_df = sorterPrefix +'/sorter_df.pkl',
        recording_info = '{recording}/processed/recording_info.pkl'
    output:
        sorter_curated_df = sorterPrefix +'/sorter_curated_df.pkl',
        waveform_figure = directory(sorterPrefix + '/waveform/')
    script:
            '01b_curation.py'



rule process_position:
    input:
        recording_to_sort = '{recording}'
    output:    
        opto_pulse = '{recording}/processed/opto_pulse.pkl',
        hd_power_spectrum = '{recording}/processed/hd_power_spectrum.png',
        synced_spatial_data = '{recording}/processed/synced_spatial_data.pkl',
        sync_pulse = '{recording}/processed/sync_pulse.png'
    script:
        '02a_process_position.py'

rule process_firings:
    input:
        recording_to_sort = '{recording}',
        sorted_data_path = '{recording}/processed/'+setting.sorterName+'/sorter_curated_df.pkl'
    output:
        spatial_firing = '{recording}/processed/spatial_firing.pkl'
    script:
        '03a_process_firings.py'

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
        '04a_process_expt_openfield.py'


rule plot_figures:
    input:
        spatial_firing = '{recording}/processed/spatial_firing_of.pkl',
        position = '{recording}/processed/synced_spatial_data.pkl',
        position_heat_map = '{recording}/processed/position_heat_map.pkl',
        hd_histogram = '{recording}/processed/hd_histogram.pkl'
    output:
        spike_histogram = directory(figure_prefix+'/spike_histogram/'),
        autocorrelogram = directory(figure_prefix+'/autocorrelogram/'),
        # spike_trajectories = directory(figure_prefix+'/spike_trajectories/'),
        # spike_rate =   directory(figure_prefix+'/spike_rate/'),
        convolved_rate = directory(figure_prefix+'/ConvolvedRates_InTime/'),
        firing_properties = directory(figure_prefix+'/firing_properties/'),
        firing_scatter = directory(figure_prefix+'/firing_scatters/'),
        session = directory(figure_prefix+'/session/'),
        rate_maps = directory(figure_prefix+'/rate_maps/'),
        rate_map_autocorrelogram = directory(figure_prefix+'/rate_map_autocorrelogram/'),
        head_direction_plots_2d = directory(figure_prefix+'/head_direction_plots_2d/'),
        head_direction_plots_polar = directory(figure_prefix+'/head_direction_plots_polar/'),
        firing_field_plots =  directory(figure_prefix+'/firing_field_plots/'),
        firing_fields_coloured_spikes = directory(figure_prefix+'/firing_fields_coloured_spikes/'),
        combined = directory(figure_prefix+'/combined/'),
        result = touch('{recording}/processed/snakemake.done')
    script:
        '05a_plot_figure_openfield.py'