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

def get_n_closest_waveforms(waveforms, primary_channel, number_of_channels, n=4):
    assert n < number_of_channels

    if number_of_channels == 16: # presume tetrodes
        geom = pd.read_csv(settings.tetrode_geom_path, header=None).values
        probe = Probe(ndim=2, si_units='um')
        probe.set_contacts(positions=geom, shapes='circle', shape_params={'radius': 5})
        probe.set_device_channel_indices(np.arange(number_of_channels))
        probe_df = probe.to_dataframe()
        probe_df["channel"] = np.arange(1,16+1)

    else: # presume cambridge neurotech P1 probes
        assert number_of_channels%64==0

        probegroup = ProbeGroup()
        n_probes = int(number_of_channels/64)
        for i in range(n_probes):
            probe = get_probe('cambridgeneurotech', 'ASSY-236-P-1')
            probe.move([i*2000, 0]) # move the probes far away from eachother
            probe.set_device_channel_indices(np.arange(64)+(64*i))
            probe.set_contact_ids(np.array(probe.to_dataframe()["contact_ids"].values, dtype=np.int64)+(64*i))
            probegroup.add_probe(probe)
        probe_df = probegroup.to_dataframe()
        probe_df["channel"] = np.arange(1,number_of_channels+1)

    primary_x = probe_df[probe_df["channel"] == primary_channel]["x"].iloc[0]
    primary_y = probe_df[probe_df["channel"] == primary_channel]["y"].iloc[0]

    channel_ids = []
    channel_distances = []
    for i, channel in probe_df.iterrows():
        channel_id = channel["channel"]
        channel_x = channel["x"]
        channel_y = channel["y"]
        dst = distance.euclidean((primary_x, primary_y), (channel_x, channel_y))
        channel_distances.append(dst)
        channel_ids.append(channel_id)
    channel_distances = np.array(channel_distances)
    channel_ids = np.array(channel_ids)
    closest_channel_ids = channel_ids[np.argsort(channel_distances)]
    closest_n = closest_channel_ids[:n]
    closest_n_as_indices = closest_n-1

    # closest n includes primary channel and n-1 of the closest contacts
    return waveforms[:,:, closest_n_as_indices]

def get_snippets(firing_data, file_path, sorter_name, dead_channels, random_snippets=True, method="from_mda"):
    if 'random_snippets' in firing_data:
        return firing_data
    print('I will get some random snippets now for each cluster.')

    snippets_all_clusters = []
    if method == "from_mda":
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

    elif method == "from_spike_interface":
        for cluster, cluster_id in enumerate(firing_data.cluster_id):
            primary_channel = firing_data[firing_data["cluster_id"] == cluster_id]["primary_channel"].iloc[0]
            number_of_channels = firing_data[firing_data["cluster_id"] == cluster_id]["number_of_channels"].iloc[0]

            waveforms = np.load(settings.temp_storage_path+"/waveform_arrays/waveforms_"+str(cluster_id)+".npy")
            snippets = get_n_closest_waveforms(waveforms, primary_channel, number_of_channels, n=4)
            snippets_all_clusters.append(snippets)

    if random_snippets is True:
        firing_data['random_snippets'] = snippets_all_clusters
    else:
        firing_data['all_snippets'] = snippets_all_clusters
    #plt.plot(firing_data.random_snippets[4][3,:,:])
    return firing_data