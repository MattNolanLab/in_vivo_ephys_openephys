import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.toolkit as st
import os
import spikeinterface.extractors as se
import OpenEphys
import shutil
import ManualCuration.manual_curation_settings


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


def create_phy(recording, spatial_firing, output_folder, sampling_rate=30000):
    signal = load_OpenEphysRecording(recording, data_file_prefix='100_CH', num_tetrodes=4)
    dead_channel_path = recording +'/dead_channels.txt'
    bad_channel = getDeadChannel(dead_channel_path)
    tetrode_geom = '/home/ubuntu/to_sort/sorting_files/geom_all_tetrodes_original.csv'
    geom = pd.read_csv(tetrode_geom,header=None).values
    recording = se.NumpyRecordingExtractor(signal, sampling_rate, geom)
    recording = st.preprocessing.remove_bad_channels(recording, bad_channel_ids=bad_channel)
    recording = st.preprocessing.bandpass_filter(recording, freq_min=300, freq_max=6000)
    recording = st.preprocessing.whiten(recording)
    recording = se.CacheRecordingExtractor(recording)
    # reconstruct a sorting extractor
    times, labels = spatial_firing2label(spatial_firing)
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
    spatial_firing_path = local_path_to_recording + '/MountainSort/DataFrames/spatial_firing.pkl'
    spatial_firing = pd.read_pickle(spatial_firing_path)
    # spatial_firing = spatial_firing[spatial_firing.number_of_spikes > 5]

    # this folder gets overwritten by spikeinterface every time it runs
    output_folder = local_path_to_recording + '/MountainSort/phy/'
    create_phy(local_path_to_recording, spatial_firing, output_folder, sampling_rate=30000)


def pre_process_recording_for_manual_curation(recording_server, recording_local):
    # todo check if it is there and copy if not. also make ubuntu/manual if needed
    if not os.path.exists(recording_local):
        shutil.copytree(recording_server, recording_local)
    # todo check parameters and also copy any paired recordings
    # make concatenated recording that has continuous data, dead channels and spatial firing
    # call phy for the combined data
    make_phy_input_for_recording(recording_local)


def main():
    recording_server = ManualCuration.manual_curation_settings.get_recording_path_datastore()
    recording_local = ManualCuration.manual_curation_settings.get_local_recording_path()
    print('This script will make the phy input files for this recording: ' + recording_server)
    pre_process_recording_for_manual_curation(recording_server, recording_local)
    # add a post manual curation function including these steps
    # (manual sort)
    # save phy output
    # read phy output and split firing times back and save as spatial_firing_curated
    # save output back on server (copy manual spatial firing back)
    ## change pipeline so it loads manual spatial firing if it exists (?)


if __name__ == '__main__':
    main()