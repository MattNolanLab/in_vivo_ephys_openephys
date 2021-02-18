import os
import mdaio
import numpy as np
import PreClustering.dead_channels
import matplotlib.pylab as plt
import PostSorting.load_snippet_data


def get_opto_snippets(firing_data, prm, random_snippets=True, column_name='snippets_opto', firing_times_column='firing_times_opto'):
    if column_name in firing_data:
        return firing_data
    print('I will get some random snippets from the opto-tagging part now for each cluster.')
    file_path = prm.get_local_recording_folder_path()
    filtered_data_path = []

    filtered_data_path = file_path + '/Electrophysiology' + prm.get_sorter_name() + '/filt.mda'

    snippets_all_clusters = []
    if os.path.exists(filtered_data_path):
        filtered_data = mdaio.readmda(filtered_data_path)
        if prm.stitchpoint is not None and prm.paired_order == "first":
            filtered_data = filtered_data[:, prm.stitchpoint:]

        for cluster, cluster_id in enumerate(firing_data.cluster_id):
            tetrode = np.asarray(firing_data[firing_data.cluster_id == cluster_id].tetrode)[0]
            firing_times = np.asarray(firing_data[firing_data.cluster_id == cluster_id][firing_times_column])[0]

            if random_snippets is True:
                snippets = PostSorting.load_snippet_data.extract_random_snippets(filtered_data, firing_times, tetrode, 50, prm)
            else:
                snippets = PostSorting.load_snippet_data.extract_all_snippets(filtered_data, firing_times, tetrode, prm)
            snippets_all_clusters.append(snippets)

    if random_snippets is True:
        random_column_name = 'random_' + column_name
        firing_data[random_column_name] = snippets_all_clusters
    else:
        firing_data[column_name] = snippets_all_clusters
    # plt.plot(firing_data.random_snippets[4][3,:,:])
    return firing_data