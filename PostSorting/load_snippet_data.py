import os
import mdaio
import numpy as np


def extract_random_snippets(filtered_data, firing_times, tetrode, number_of_snippets):
    random_indices = np.ceil(np.random.uniform(0, len(firing_times), number_of_snippets)).astype(int)
    snippets = np.zeros((4, 30, number_of_snippets))

    channels = [(tetrode-1)*4, (tetrode-1)*4 + 1, (tetrode-1)*4 + 2, (tetrode-1)*4 + 3]

    for index, event in enumerate(random_indices):
        snippets_indices = np.arange(firing_times[event]-15, firing_times[event]+15, 1).astype(int)
        snippets[:, :, index] = filtered_data[channels[0]:channels[3]+1, snippets_indices]

    return snippets


def get_snippets(firing_data, prm):
    print('I will get some random snippets now for each cluster.')
    file_path = prm.get_local_recording_folder_path()
    filtered_data_path = []
    if prm.get_is_windows():
        filtered_data_path = file_path + '\\Electrophysiology\\Spike_sorting\\all_tetrodes\\data\\filt.mda'

    if prm.get_is_ubuntu():
        filtered_data_path = file_path + '/Electrophysiology/Spike_sorting/all_tetrodes/data/filt.mda'

    snippets_all_clusters = []
    if os.path.exists(filtered_data_path):
        filtered_data = mdaio.readmda(filtered_data_path)
        for cluster in range(len(firing_data)):
            cluster = firing_data.cluster_id.values[cluster] - 1
            firing_times = firing_data.firing_times[cluster]
            snippets = extract_random_snippets(filtered_data, firing_times, firing_data.tetrode[cluster], 50)
            snippets_all_clusters.append(snippets)
    firing_data['random_snippets'] = snippets_all_clusters
    return firing_data