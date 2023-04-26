
import mdaio
import numpy as np
import PreClustering.dead_channels
import matplotlib.pylab as plt
from scipy.spatial import distance
import settings
from file_utility import *
from PostSorting.load_snippet_data import get_n_closest_waveforms
import PostSorting.load_snippet_data


def get_opto_snippets(firing_data, local_recording_folder, sorter_name, dead_channels, random_snippets=True, column_name='snippets_opto', firing_times_column='firing_times_opto', method='from_mda'):
    """
    Get snippets / action potentials from the filtered data from during opto tagging.

    """
    if column_name in firing_data:
        return firing_data
    print('I will get some random snippets from the opto-tagging part now for each cluster for this column: ' + column_name)

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

    elif method == 'from_mda':
        filtered_data_path = local_recording_folder + '/Electrophysiology' + sorter_name + '/filt.mda'
        if os.path.exists(filtered_data_path):
            filtered_data = mdaio.readmda(filtered_data_path)
            for cluster, cluster_id in enumerate(firing_data.cluster_id):
                tetrode = np.asarray(firing_data[firing_data.cluster_id == cluster_id].tetrode)[0]
                firing_times = np.asarray(firing_data[firing_data.cluster_id == cluster_id][firing_times_column])[0]
                firing_times = np.array(firing_times)
                firing_times = firing_times[~np.isnan(firing_times)]  # this can happen in some types of opto data

                if len(firing_times) > 0:
                    if random_snippets is True:
                        snippets = PostSorting.load_snippet_data.extract_random_snippets(filtered_data, firing_times, tetrode, 50, dead_channels)
                    else:
                        snippets = PostSorting.load_snippet_data.extract_all_snippets(filtered_data, firing_times, tetrode, dead_channels)
                    snippets_all_clusters.append(snippets)
                else:
                    snippets_all_clusters.append([])

    if random_snippets is True:
        random_column_name = 'random_' + column_name
        firing_data[random_column_name] = snippets_all_clusters
    else:
        firing_data[column_name] = snippets_all_clusters
    return firing_data