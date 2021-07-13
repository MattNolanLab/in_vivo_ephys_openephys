import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.toolkit as st
import os
import spikeinterface.extractors as se
import OpenEphys
import shutil


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


def manually_curate_sorting_results(recording, output):
    spatial_firing_path = recording + '/MountainSort/DataFrames/spatial_firing.pkl'
    spatial_firing = pd.read_pickle(spatial_firing_path)
    # spatial_firing = spatial_firing[spatial_firing.number_of_spikes > 5]

    output_folder = output + '/MountainSort/phy/'
    create_phy(recording, spatial_firing, output_folder, sampling_rate=30000)


def pre_process_recording_for_manual_curation(recording_server, recording_local):
    # todo check if it is there and copy if not. also make ubuntu/manual if needed
    # shutil.copytree(recording_server, recording_local)
    # todo check parameters and also copy any paired recordings
    # make concatenated recording that has continuous data, dead channels and spatial firing
    # call phy for the combined data
    manually_curate_sorting_results(recording_local, recording_local)


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    recording_name = 'M3_2021-05-26_14-19-02_of'
    exp_folder = 'Klara/CA1_to_deep_MEC_in_vivo/analysis_test_manual/'
    recording_server = "/mnt/datastore/" + exp_folder + recording_name
    recording_local = "/home/ubuntu/manual/" + recording_name

    pre_process_recording_for_manual_curation(recording_server, recording_local)
    # add a post manual curation function including these steps
    # (manual sort)
    # save phy output
    # read phy output and split firing times back and save as spatial_firing_curated
    # save output back on server (copy manual spatial firing back)

    ## change pipeline so it loads manual spatial firing if it exists (?)


if __name__ == '__main__':
    main()