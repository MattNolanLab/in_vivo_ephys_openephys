
sorterPrefix = '{recording}/processed/'+setting.sorterName

rule sort_spikes:
    input:
        probe_file = 'sorting_files/tetrode_16.prb',
        sort_param = 'sorting_files/params.json',
        tetrode_geom = 'sorting_files/geom_all_tetrodes_original.csv',
        recording_to_sort = '{recording}',
        # parameter_file = '{recording}/parameters.yaml'
        # dead_channel = '{recording}/dead_channels.txt'
    threads: workflow.cores
    output:
        sorter = sorterPrefix +'/sorter.pkl',
        sorter_df = sorterPrefix +'/sorter_df.pkl',
        recording_info = '{recording}/processed/recording_info.pkl'
    script:
        '../scripts/01_sorting.py'


rule curate_clusters:
    input:
        sorter_df = sorterPrefix +'/sorter_df.pkl',
        recording_info = '{recording}/processed/recording_info.pkl'
    output:
        sorter_curated_df = sorterPrefix +'/sorter_curated_df.pkl',
        waveform_figure_all = directory(sorterPrefix + '/waveform/all/'),
        waveform_figure_curated =directory(sorterPrefix + '/waveform/curated/')
    script:
        '../scripts/01b_curation.py'
