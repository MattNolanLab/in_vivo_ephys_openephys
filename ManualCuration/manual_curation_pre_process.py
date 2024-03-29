import pandas as pd
import numpy as np
import spikeinterface.toolkit as st
import os
import spikeinterface.extractors as se
import OpenEphys
import shutil
import ManualCuration.manual_curation_settings
import control_sorting_analysis
import glob
import multiprocessing
from joblib import Parallel, delayed
from PreClustering import pre_process_ephys_data


def copy_recording_to_sort_to_local(path_server, path_local):
    print('Copying this recording: ' + path_server)
    if os.path.exists(path_server) is False:
        print('This folder does not exist on the server:')
        print(path_server)
        return False
    if not os.path.exists(path_local):
        os.mkdir(path_local)
        num_cores = multiprocessing.cpu_count()
        Parallel(n_jobs=num_cores)(delayed(control_sorting_analysis.copy_file)(filename, path_local) for filename in glob.glob(os.path.join(path_server, '*.*')))

        spatial_firing_path = path_server + '/MountainSort/DataFrames/spatial_firing.pkl'
        if os.path.isfile(spatial_firing_path) is True:
            if not os.path.isdir(path_local + '/MountainSort/DataFrames/'):
                os.makedirs(path_local + '/MountainSort/DataFrames/')
            shutil.copy(spatial_firing_path, path_local + '/MountainSort/DataFrames/spatial_firing.pkl')
    else:
        print('There is already a folder with this name on the computer running the analysis, I will not overwrite it. '
              'Delete (rm -r foldername on terminal) and rerun if you want it overwritten.')


def spatial_firing2label(spatial_firing):
    times = []
    labels = []
    for cluster_id in np.unique(spatial_firing["cluster_id"]):
        cluster_spatial_firing = spatial_firing[(spatial_firing["cluster_id"] == cluster_id)]
        cluster_times = list(cluster_spatial_firing["firing_times"].iloc[0])
        cluster_labels = list(cluster_id*np.ones(len(cluster_times)))

        times.extend(cluster_times)
        labels.extend(cluster_labels)
    return np.array(times), np.array(labels)


def load_OpenEphysRecording(folder, data_file_prefix, num_tetrodes):
    signal = []
    for i in range(num_tetrodes*4):
        fname = folder+'/'+data_file_prefix+str(i+1)+'.continuous'
        x = OpenEphys.loadContinuousFast(fname)['data']
        if i==0:
            #preallocate array on first run
            signal = np.zeros((num_tetrodes*4,x.shape[0]))
        signal[i,:] = x
    return signal


def getDeadChannel(deadChannelFile):
    deadChannels=None
    if os.path.exists(deadChannelFile):
        with open(deadChannelFile,'r') as f:
            deadChannels = [int(s) for s in f.readlines()]
    return deadChannels


def get_geom():
    geom = np.zeros((16, 2))
    geom[0] = [0, 200]
    geom[1] = [25, 200.1]
    geom[2] = [25, 200.2]
    geom[3] = [0, 200.3]
    geom[4] = [0, 200.4]
    geom[5] = [25, 200.5]
    geom[6] = [25, 200.6]
    geom[7] = [0, 200.7]
    geom[8] = [0, 200.8]
    geom[9] = [25, 200.9]
    geom[10] = [25, 201]
    geom[11] = [0, 201.1]
    geom[12] = [0, 201.2]
    geom[13] = [25, 201.3]
    geom[14] = [25, 201.4]
    geom[15] = [0, 201.5]
    return geom


def create_phy(recording, spatial_firing, output_folder, sampling_rate=30000):
    times, labels = spatial_firing2label(spatial_firing)
    signal = load_OpenEphysRecording(recording, data_file_prefix='100_CH', num_tetrodes=4)
    dead_channel_path = recording + '/dead_channels.txt'
    bad_channel = getDeadChannel(dead_channel_path)
    geom = get_geom()
    recording = se.NumpyRecordingExtractor(signal, sampling_rate, geom)
    recording = st.preprocessing.remove_bad_channels(recording, bad_channel_ids=bad_channel)
    recording = st.preprocessing.bandpass_filter(recording, freq_min=300, freq_max=6000)
    recording = st.preprocessing.whiten(recording)
    recording = se.CacheRecordingExtractor(recording)
    # reconstruct a sorting extractor
    sorting = se.NumpySortingExtractor()
    sorting.set_times_labels(times=times, labels=labels)
    sorting.set_sampling_frequency(sampling_frequency=sampling_rate)
    st.postprocessing.export_to_phy(recording, sorting, output_folder=output_folder,
                                    copy_binary=True, ms_before=0.5, ms_after=0.5)
    print("I have created the phy output for ", recording)


def make_phy_input_for_recording(local_path_to_recording):
    """
    :param local_path_to_recording: this is the path to the recording. local means that the analysis will be done here
    (the data was copied here) so this could be your computer or an instance in the cloud
    """
    spatial_firing_path = local_path_to_recording + '/MountainSort/DataFrames/spatial_firing_manual.pkl'
    spatial_firing = pd.read_pickle(spatial_firing_path)
    # spatial_firing = spatial_firing[spatial_firing.number_of_spikes > 5]

    # this folder gets overwritten by spikeinterface every time it runs
    output_folder = local_path_to_recording + '/MountainSort/phy/'
    create_phy(local_path_to_recording, spatial_firing, output_folder, sampling_rate=30000)


def get_list_of_paired_recordings_local(recording_local):
    tags = control_sorting_analysis.get_tags_parameter_file(recording_local)
    paired_recordings = control_sorting_analysis.check_for_paired(tags)
    main_local_folder = '/'.join(recording_local.split('/')[:-1]) + '/'
    paired_local = []
    for paired_recording in paired_recordings:
        paired_folder_name = paired_recording.split('/')[-1]
        paired_local.append(main_local_folder + paired_folder_name)
    return paired_local


def copy_recordings_to_local(recording_local, recording_server):
    main_local_folder = '/'.join(recording_local.split('/')[:-1]) + '/'
    beginning_of_server_path = '/'.join(recording_server.split('/')[:3]) + '/'
    if not os.path.exists(recording_local):
        shutil.copytree(recording_server, recording_local)
    tags = control_sorting_analysis.get_tags_parameter_file(recording_local)
    paired_recordings = control_sorting_analysis.check_for_paired(tags)
    if paired_recordings is not None:
        print('There are some recordings sorted together with this recording. These will be copied too. '
              + str(paired_recordings))
        for paired_recording in paired_recordings:
            path_server = beginning_of_server_path + paired_recording
            end_of_paired_path = paired_recording.split('/')[-1]
            path_local = main_local_folder + end_of_paired_path
            copy_recording_to_sort_to_local(path_server, path_local)


def get_list_of_clusters_across_all_recordings(recording_local, paired_recordings):
    df_path = '/MountainSort/DataFrames/spatial_firing.pkl'

    spatial_firing = pd.read_pickle(recording_local + df_path)
    spatial_firing = spatial_firing[['cluster_id', 'firing_times']]  # only keep the columns we need
    all_clusters = spatial_firing.cluster_id.values.tolist()
    for index, paired_recording in enumerate(paired_recordings):
        # concatenate firing times from paired recordings to cluster
        paired_df = paired_recording + df_path
        spatial_firing_paired = pd.read_pickle(paired_df)
        cluster_ids = spatial_firing_paired.cluster_id
        all_clusters.extend(cluster_ids.tolist())
    all_unique_clusters = np.unique(all_clusters)
    return all_unique_clusters


def add_opto_times_to_list(spatial_firing, cluster_index, firing_times_cluster, stitchpoint=None):
    opto_times = spatial_firing[spatial_firing.cluster_id == cluster_index].firing_times_opto
    if len(opto_times) > 0:
        opto_times = opto_times.tolist()[0].tolist()
        if stitchpoint is not None:
            opto_times = [x + stitchpoint for x in opto_times]
        firing_times_cluster.extend(opto_times)
    return firing_times_cluster


def make_combined_spatial_firing_df(recording_local, paired_recordings, stitch_points):
    opto_session = False
    df_path = '/MountainSort/DataFrames/spatial_firing.pkl'
    spatial_firing_combined = pd.DataFrame()
    spatial_firing = pd.read_pickle(recording_local + df_path)
    if 'firing_times_opto' in spatial_firing:
        spatial_firing = spatial_firing[['cluster_id', 'firing_times', 'firing_times_opto']]
        opto_session = True
    else:
        spatial_firing = spatial_firing[['cluster_id', 'firing_times']]  # only keep the columns we need
    combined_firing_times = []
    cluster_ids_combined = []
    cluster_ids_all_recordings = get_list_of_clusters_across_all_recordings(recording_local, paired_recordings)
    for cluster_index in cluster_ids_all_recordings:
        firing_times_cluster = spatial_firing[spatial_firing.cluster_id == cluster_index].firing_times
        if len(firing_times_cluster) > 0:
            firing_times_cluster = firing_times_cluster.tolist()[0].tolist()
        else:
            firing_times_cluster = []
        if opto_session:
            firing_times_cluster = add_opto_times_to_list(spatial_firing, cluster_index, firing_times_cluster)
        for index, paired_recording in enumerate(paired_recordings):
            # concatenate firing times from paired recordings to cluster
            paired_df = paired_recording + df_path
            spatial_firing_paired = pd.read_pickle(paired_df)
            paired_cluster_times = spatial_firing_paired[spatial_firing_paired.cluster_id == cluster_index].firing_times
            paired_cluster_times += stitch_points[index]
            if len(paired_cluster_times) > 0:
                paired_cluster_times_list = paired_cluster_times.iloc[0].tolist()
                firing_times_cluster.extend(paired_cluster_times_list)
            if 'firing_times_opto' in spatial_firing_paired:
                firing_times_cluster = add_opto_times_to_list(spatial_firing_paired, cluster_index, firing_times_cluster, stitchpoint=stitch_points[index])
        combined_firing_times.append(firing_times_cluster)
        cluster_ids_combined.append(cluster_index)

    spatial_firing_combined['cluster_id'] = cluster_ids_combined
    spatial_firing_combined['firing_times'] = combined_firing_times
    spatial_firing_combined.to_pickle(recording_local + '/MountainSort/DataFrames/spatial_firing_manual.pkl')


def delete_unused_raw_data(paired_recordings):
    for recording in paired_recordings:
        files_in_directory = os.listdir(recording)
        filtered_files = [file for file in files_in_directory if file.endswith(".continuous")]
        for file in filtered_files:
            path_to_file = os.path.join(recording, file)
            os.remove(path_to_file)


def pre_process_recording_for_manual_curation(recording_server, recording_local):
    copy_recordings_to_local(recording_local, recording_server)
    paired_recordings = get_list_of_paired_recordings_local(recording_local)
    # this will concatenate all the recordings that were copied
    recording_local, stitch_points = pre_process_ephys_data.stitch_recordings(recording_local, paired_recordings)
    np.savetxt(recording_local + '/stitch_points.csv', stitch_points, delimiter=',')   # test
    make_combined_spatial_firing_df(recording_local, paired_recordings, stitch_points)
    # call phy for the combined data
    delete_unused_raw_data(paired_recordings)
    make_phy_input_for_recording(recording_local)


def main():
    recording_server = ManualCuration.manual_curation_settings.get_recording_path_datastore()
    recording_local = ManualCuration.manual_curation_settings.get_local_recording_path()
    print('This script will make the phy input files for this recording: ' + recording_server)
    pre_process_recording_for_manual_curation(recording_server, recording_local)


if __name__ == '__main__':
    main()