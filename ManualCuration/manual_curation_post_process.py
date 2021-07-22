import control_sorting_analysis
import ManualCuration.manual_curation_settings
import numpy as np
import pandas as pd
import shutil


def load_phy_output(recording_local):
    print('Loading phy output...')
    path_to_phy_output = recording_local + '/MountainSort/phy/'
    spike_times = np.load(path_to_phy_output + 'spike_times.npy')
    spike_clusters = np.load(path_to_phy_output + 'spike_clusters.npy')
    cluster_group = pd.read_csv(path_to_phy_output + 'cluster_group.tsv', sep="\t")
    print('Clusters loaded:')
    print(cluster_group)
    if cluster_group.group.str.contains('unsorted').sum() > 0:
        print('WARNING: There are unsorted clusters in this file. '
              'You should mark each cluster as good, noise or MUA during curation. You should be running this script '
              'after doing the manual curation in phy. '
              'https://phy.readthedocs.io/en/latest/clustering/')
    return spike_times, spike_clusters, cluster_group


def split_spike_times_for_clusters(spike_times, spike_clusters):
    spatial_firing_combined = pd.DataFrame()
    cluster_ids = np.unique(spike_clusters)
    firing_times = []
    for cluster in cluster_ids:
        firing_times.append(spike_times[spike_clusters == cluster])

    spatial_firing_combined['cluster_id'] = cluster_ids
    spatial_firing_combined['firing_times'] = firing_times
    return spatial_firing_combined


def get_spatial_firing_for_stitch_points(spatial_firing_combined, stitch_point_1, stitch_point_2):
    spatial_firing = pd.DataFrame()
    firing_times = []
    for cluster_id, cluster in spatial_firing_combined.iterrows():
        firing_times_all = cluster.firing_times.flatten()
        bigger_than_previous_stitch = firing_times_all > stitch_point_1
        smaller_than_next_stitch = firing_times_all < stitch_point_2
        in_between = bigger_than_previous_stitch & smaller_than_next_stitch
        firing_times.append(firing_times_all[in_between])
    spatial_firing['cluster_id'] = spatial_firing_combined.cluster_id
    spatial_firing['manual_cluster_group'] = spatial_firing_combined.manual_cluster_group
    spatial_firing['firing_times'] = firing_times
    spatial_firing['primary_channel'] = spatial_firing_combined.primary_channel
    return spatial_firing


def save_data_for_recording(spatial_firing, recording_path, beginning_of_server_path):
    spatial_firing['session_id'] = recording_path.split('/')[-2]
    good_clusters = spatial_firing[spatial_firing.manual_cluster_group == 'good']
    noise_clusters = spatial_firing[spatial_firing.manual_cluster_group != 'good']
    good_clusters.to_pickle(
        beginning_of_server_path + recording_path + '/MountainSort/DataFrames/spatial_firing_curated.pkl')
    noise_clusters.to_pickle(
        beginning_of_server_path + recording_path + '/MountainSort/DataFrames/spatial_firing_curated_noise.pkl')
    print('The curated data is saved here: ' + beginning_of_server_path + recording_path + '/MountainSort/DataFrames/spatial_firing_curated.pkl')


def split_and_save_on_server(recording_local, recording_server, spatial_firing_combined, stitch_points):
    beginning_of_server_path = '/' + '/'.join(recording_server.split('/')[1:3]) + '/'
    end_of_path = '/' + '/'.join(recording_server.split('/')[3:]) + '/'
    tags = control_sorting_analysis.get_tags_parameter_file(recording_local)
    paired_recordings = control_sorting_analysis.check_for_paired(tags)
    stitch_point_1 = stitch_points.iloc[0][0]
    spatial_firing_first = get_spatial_firing_for_stitch_points(spatial_firing_combined, 0, stitch_point_1)
    save_data_for_recording(spatial_firing_first, end_of_path, beginning_of_server_path)
    for recording_index, paired_recording in enumerate(paired_recordings):
        stitch_point_1 = stitch_points.iloc[recording_index][0]
        stitch_point_2 = stitch_points.iloc[recording_index + 1][0]
        spatial_firing = get_spatial_firing_for_stitch_points(spatial_firing_combined, stitch_point_1, stitch_point_2)
        save_data_for_recording(spatial_firing, paired_recording, beginning_of_server_path)


def get_sum_of_waveforms_for_channel(ch, ephys_data, times):
    sum_amp = 0
    for time in range(len(times)):
        wave = ephys_data[ch][times[time] - 15:times[time] + 15]
        sum_amp += np.sum(np.abs(wave))
    return sum_amp


def find_primary_channel_for_each_cluster(recording_local, spatial_firing, num_channels=16):
    primary_channels = []
    ephys_data_flat = np.fromfile(recording_local + '/MountainSort/phy/recording.dat')
    ephys_data = [ephys_data_flat[idx::num_channels] for idx in range(num_channels)]
    for cluster_id, cluster in spatial_firing.iterrows():
        random_firing_times = np.random.choice(cluster.firing_times.flatten(), size=500, replace=True)
        highest_ch = 0
        highest_amp = 0
        for ch in range(num_channels):
            sum_amplitude = get_sum_of_waveforms_for_channel(ch, ephys_data, random_firing_times)
            if sum_amplitude > highest_amp:
                highest_ch = ch
                highest_amp = sum_amplitude
        primary_channels.append(highest_ch + 1)
    spatial_firing['primary_channel'] = primary_channels
    return spatial_firing


def post_process_manually_curated_data(recording_server, recording_local):
    spike_times, spike_clusters, cluster_group = load_phy_output(recording_local)
    stitch_points = pd.read_csv(recording_local + '/stitch_points.csv', header=None)
    spatial_firing_combined = split_spike_times_for_clusters(spike_times, spike_clusters)
    spatial_firing_combined['manual_cluster_group'] = cluster_group.group
    spatial_firing_combined = find_primary_channel_for_each_cluster(recording_local, spatial_firing_combined)
    split_and_save_on_server(recording_local, recording_server, spatial_firing_combined, stitch_points)
    shutil.rmtree('/'.join(recording_local.split('/')[:-1]))


def main():
    recording_server = ManualCuration.manual_curation_settings.get_recording_path_datastore()
    recording_local = ManualCuration.manual_curation_settings.get_local_recording_path()
    print('This script will process the manually curated phy outputs and upload them to datastore: ' + recording_server)
    post_process_manually_curated_data(recording_server, recording_local)


if __name__ == '__main__':
    main()