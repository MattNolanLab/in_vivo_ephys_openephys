import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd
import pynapple as nap
import seaborn as sns


def identify_interneurons(recording_path, sampling_rate=30000, binsize=0.02):
    path_to_spatial_firing = recording_path + "/MountainSort/DataFrames/spatial_firing.pkl"
    if os.path.exists(path_to_spatial_firing):
        # plot all combinations
        spatial_firing = pd.read_pickle(path_to_spatial_firing)
        tetrode_ids = spatial_firing.tetrode.unique()
        for tetrode in tetrode_ids:
            neurons_on_tetrode = spatial_firing[spatial_firing.tetrode == tetrode]
            number_of_neurons = len(neurons_on_tetrode)
            if number_of_neurons > 1:
                cluster_ids = neurons_on_tetrode.cluster_id.values
                fig, axs = plt.subplots(number_of_neurons, number_of_neurons)
                for cluster1_index, cluster1 in enumerate(cluster_ids):
                    for cluster2_index, cluster2 in enumerate(cluster_ids):
                        neuron_1_times = np.array((spatial_firing[spatial_firing.cluster_id == cluster1].firing_times.values / sampling_rate)[0])
                        neuron_2_times = np.array((spatial_firing[spatial_firing.cluster_id == cluster2].firing_times.values / sampling_rate)[0])
                        cross_corr, xt = nap.cross_correlogram(neuron_1_times, neuron_2_times, binsize=binsize, windowsize=0.5)

                        axs[cluster1_index, cluster2_index].bar(xt, cross_corr, binsize)
                        sns.despine(top=True, right=True, left=False, bottom=False)
            # plt.title('Tetrode ' + str(tetrode), fontsize=16)
            plt.show()
                        # plt.xlabel("Time (us)")
                        # plt.ylabel("Cross-correlation", fontsize=1)
                        # plt.savefig()  # save to new folder with cross corrs

    # todo make combined plot with all cross-corrs for session


def process_recordings(experiment_folder):
    recording_list = []
    recording_list.extend([f.path for f in os.scandir(experiment_folder) if f.is_dir()])
    for recording in recording_list:
        identify_interneurons(recording)


def main():
    experiment_folder = "/mnt/datastore/Klara/CA1_to_deep_MEC_in_vivo/"
    process_recordings(experiment_folder)
    recording_path = "/mnt/datastore/Klara/CA1_to_deep_MEC_in_vivo/M3_2021-06-16_14-10-45_of/"
    identify_interneurons(recording_path)


if __name__ == '__main__':
    main()
