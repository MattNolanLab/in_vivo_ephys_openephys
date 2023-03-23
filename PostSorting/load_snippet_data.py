import os
import mdaio
import numpy as np
import settings
import pandas as pd
import spikeinterface as si
import PreClustering.dead_channels
import matplotlib.pylab as plt
from probeinterface import Probe, ProbeGroup
from probeinterface.plotting import plot_probe
from probeinterface import get_probe
from scipy.spatial import distance
from file_utility import *
from spikeinterfaceHelper import get_probe_dataframe

def extract_random_snippets(filtered_data, firing_times, tetrode, number_of_snippets, dead_channels):
    if len(dead_channels) != 0:
        for dead_ch in range(len(dead_channels[0])):
            to_insert = np.zeros(len(filtered_data[0]))
            filtered_data = np.insert(filtered_data, int(dead_channels[0][dead_ch]) - 1, to_insert, 0)

    if len(firing_times)<50:
        random_indices = np.arange(0, len(firing_times))
    else:
        random_indices = np.ceil(np.random.uniform(16, len(firing_times)-16, number_of_snippets)).astype(int)
    snippets = np.zeros((4, 30, number_of_snippets))

    channels = [(tetrode-1)*4, (tetrode-1)*4 + 1, (tetrode-1)*4 + 2, (tetrode-1)*4 + 3]

    for index, event in enumerate(random_indices):
        snippets_indices = np.arange(firing_times[event]-10, firing_times[event]+20, 1).astype(int)
        snippets[:, :, index] = filtered_data[channels[0]:channels[3]+1, snippets_indices]
    # plt.plot(snippets[3,:,:]) # example ch plot
    return snippets

def extract_all_snippets(filtered_data, firing_times, tetrode, dead_channels):
    if len(dead_channels) != 0:
        for dead_ch in range(len(dead_channels[0])):
            to_insert = np.zeros(len(filtered_data[0]))
            filtered_data = np.insert(filtered_data, int(dead_channels[0][dead_ch]) - 1, to_insert, 0)

    if len(firing_times)<50:
        all_indices = np.arange(0, len(firing_times))
    else:
        all_indices = np.arange(16, len(firing_times)-16)
    snippets = np.zeros((4, 30, len(all_indices)))

    channels = [(tetrode-1)*4, (tetrode-1)*4 + 1, (tetrode-1)*4 + 2, (tetrode-1)*4 + 3]

    for index, event in enumerate(all_indices):
        snippets_indices = np.arange(firing_times[event]-10, firing_times[event]+20, 1).astype(int)
        snippets[:, :, index] = filtered_data[channels[0]:channels[3]+1, snippets_indices]
    # plt.plot(snippets[3,:,:]) # example ch plot
    return snippets

def get_snippet_method(SorterInstance=None):
    if SorterInstance is not None:
        return "from_spike_interface"
    else:
        return "from_mda"

def get_n_closest_waveforms(waveforms, number_of_channels, primary_channel, probe_id, shank_id, n=16):
    probe_df = get_probe_dataframe(number_of_channels)
    shank_df = probe_df[(probe_df["probe_index"] == int(probe_id)) & (probe_df["shank_ids"] == int(shank_id))]
    shank_df = shank_df.sort_values(by="channel", ascending=True)
    shank_df = shank_df.reset_index()
    # primary channel is the index of largest template waveform in the shank plus 1
    primary_x = shank_df["x"].iloc[primary_channel]
    primary_y = shank_df["y"].iloc[primary_channel]

    channel_indices = []
    channel_distances = []
    for i, channel in shank_df.iterrows():
        channel_x = channel["x"]
        channel_y = channel["y"]
        dst = distance.euclidean((primary_x, primary_y), (channel_x, channel_y))
        channel_distances.append(dst)
        channel_indices.append(i)
    channel_distances = np.array(channel_distances)
    channel_indices = np.array(channel_indices)
    closest_channel_indices = channel_indices[np.argsort(channel_distances)]
    closest_n_as_indices = closest_channel_indices[:n]
    return waveforms[closest_n_as_indices, :, :]

def get_snippets(firing_data, file_path, sorter_name, dead_channels, random_snippets=True, method="from_mda"):
    if 'random_snippets' in firing_data:
        return firing_data
    print('I will get some random snippets now for each cluster.')

    snippets_all_clusters = []
    if found_SorterInstance():
        for cluster, cluster_id in enumerate(firing_data.cluster_id):
            primary_channel = firing_data[firing_data["cluster_id"] == cluster_id]["primary_channel"].iloc[0]
            number_of_channels = firing_data[firing_data["cluster_id"] == cluster_id]["number_of_channels"].iloc[0]
            probe_id = firing_data[firing_data["cluster_id"] == cluster_id]["probe_id"].iloc[0]
            shank_id = firing_data[firing_data["cluster_id"] == cluster_id]["shank_id"].iloc[0]

            waveforms = np.load(settings.temp_storage_path+"/waveform_arrays/waveforms_"+str(int(cluster_id))+".npy")
            waveforms = np.swapaxes(waveforms, 0, 2)
            snippets = get_n_closest_waveforms(waveforms, number_of_channels, primary_channel, probe_id, shank_id)
            snippets_all_clusters.append(snippets)
        firing_data["primary_channel"] = 1 # all waveforms are sorted according to the spatial proximity to the primary channel

    elif method == "from_mda":
        filtered_data_path = file_path + '/Electrophysiology/' + sorter_name + '/filt.mda'
        if os.path.exists(filtered_data_path):
            filtered_data = mdaio.readmda(filtered_data_path)
            for cluster, cluster_id in enumerate(firing_data.cluster_id):
                tetrode = np.asarray(firing_data[firing_data.cluster_id == cluster_id].tetrode)[0]
                firing_times = np.asarray(firing_data[firing_data.cluster_id == cluster_id].firing_times)[0]
                if random_snippets is True:
                    snippets = extract_random_snippets(filtered_data, firing_times, tetrode, 50, dead_channels)
                else:
                    snippets = extract_all_snippets(filtered_data, firing_times, tetrode, dead_channels)
                snippets_all_clusters.append(snippets)

    if random_snippets is True:
        firing_data['random_snippets'] = snippets_all_clusters
    else:
        firing_data['all_snippets'] = snippets_all_clusters
    #plt.plot(firing_data.random_snippets[4][3,:,:])
    return firing_data