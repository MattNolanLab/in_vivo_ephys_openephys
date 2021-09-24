import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import settings
import os
import sys
import traceback
#import warnings
#warnings.filterwarnings("ignore")

def plot_shuffle_distributions(recordings_folder_to_process):
    recording_list = [f.path for f in os.scandir(recordings_folder_to_process) if f.is_dir()]

    for recording_path in recording_list:
        print("processing ", recording_path)
        output_path = recording_path+'/'+settings.sorterName

        shuffle=pd.DataFrame()
        if os.path.isdir(recording_path+r"/MountainSort/DataFrames/shuffles"):
            shuffle_list = [f.path for f in os.scandir(recording_path+r"/MountainSort/DataFrames/shuffles") if f.is_file()]
            # remove shuffle.pkl if one is found, this is a deprecated pkl.
            if os.path.isfile(recording_path+r"/MountainSort/DataFrames/shuffles/shuffle.pkl"):
                shuffle_list.remove(recording_path+r"/MountainSort/DataFrames/shuffles/shuffle.pkl")

            for i in range(len(shuffle_list)):
                cluster_shuffle = pd.read_pickle(shuffle_list[i])
                shuffle = pd.concat([shuffle, cluster_shuffle], ignore_index=False)
            print("I have found a shuffled dataframe")

            if os.path.isfile(recording_path+r"/MountainSort/DataFrames/spatial_firing.pkl"):
                spatial_firing = pd.read_pickle(recording_path+r"/MountainSort/DataFrames/spatial_firing.pkl")

                if len(spatial_firing)>0:
                    print("cluster IDs in shuffle df: ", np.unique(shuffle.cluster_id))
                    print("cluster IDs in spatial df: ", np.unique(shuffle.cluster_id))
                    print("There are", len(shuffle)/len(spatial_firing), "shuffles per cell")

                    print('plotting shuffle distibutions...')
                    save_path = output_path + '/Figures/shuffle distributions'
                    if os.path.exists(save_path) is False:
                        os.makedirs(save_path)

                    for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
                        cluster_spatial_firing = spatial_firing[(spatial_firing["cluster_id"] == cluster_id)]
                        cluster_shuffle_df = shuffle[(shuffle.cluster_id == cluster_id)] # dataframe for that cluster
                        print("For cluster", cluster_id, " there are ", len(cluster_shuffle_df), " shuffles")

                        speed_scores = np.array(cluster_shuffle_df["speed_score"])
                        hd_scores = np.array(cluster_shuffle_df["hd_score"])
                        rayleigh_scores = np.array(cluster_shuffle_df["rayleigh_score"])
                        spatial_information_scores = np.array(cluster_shuffle_df["spatial_information_score"])
                        grid_scores = np.array(cluster_shuffle_df["grid_score"])
                        border_scores = np.array(cluster_shuffle_df["border_score"])
                        half_session_scores = np.array(cluster_shuffle_df["rate_map_correlation_first_vs_second_half"])

                        hd_threshold = cluster_spatial_firing["hd_threshold"].iloc[0]
                        rayleigh_threshold = cluster_spatial_firing["rayleigh_threshold"].iloc[0]
                        spatial_threshold = cluster_spatial_firing['spatial_threshold'].iloc[0]
                        grid_threshold = cluster_spatial_firing['grid_threshold'].iloc[0]
                        border_threshold = cluster_spatial_firing['border_threshold'].iloc[0]
                        half_session_threshold = cluster_spatial_firing['half_session_threshold'].iloc[0]
                        hd_score = cluster_spatial_firing["hd_score"].iloc[0]
                        rayleigh_score = cluster_spatial_firing['rayleigh_score'].iloc[0]
                        grid_score = cluster_spatial_firing['grid_score'].iloc[0]
                        border_score = cluster_spatial_firing['border_score'].iloc[0]
                        half_session_score = cluster_spatial_firing['rate_map_correlation_first_vs_second_half'].iloc[0]
                        spatial_information_score = cluster_spatial_firing['spatial_information_score'].iloc[0]

                        n, bins, patches = plt.hist(speed_scores, 50, density=True, facecolor='b', alpha=0.75)
                        plt.savefig(save_path + '/' + spatial_firing.session_id.iloc[cluster_index] + '_speed_shuffle_dist_Cluster_' + str(cluster_id) + '.png', dpi=200)
                        plt.close()
                        plt.cla()

                        n, bins, patches = plt.hist(hd_scores, 50, density=True, facecolor='b', alpha=0.75)
                        plt.vlines(hd_score, ymin=0, ymax=max(n), colors='k', linestyles='solid')
                        plt.vlines(hd_threshold, ymin=0, ymax=max(n), colors='k', linestyles='dashed')
                        plt.savefig(save_path + '/' + spatial_firing.session_id.iloc[cluster_index] + '_hd_shuffle_dist_Cluster_' + str(cluster_id) + '.png', dpi=200)
                        plt.close()
                        plt.cla()

                        n, bins, patches = plt.hist(rayleigh_scores, 50, density=True, facecolor='b', alpha=0.75)
                        plt.vlines(rayleigh_score, ymin=0, ymax=max(n), colors='k', linestyles='solid')
                        plt.vlines(rayleigh_threshold, ymin=0, ymax=max(n), colors='k', linestyles='dashed')
                        plt.savefig(save_path + '/' + spatial_firing.session_id.iloc[cluster_index] + '_rayleigh_shuffle_dist_Cluster_' + str(cluster_id) + '.png', dpi=200)
                        plt.close()
                        plt.cla()

                        n, bins, patches = plt.hist(grid_scores, 50, density=True, facecolor='b', alpha=0.75)
                        plt.vlines(grid_score, ymin=0, ymax=max(n), colors='k', linestyles='solid')
                        plt.vlines(grid_threshold, ymin=0, ymax=max(n), colors='k', linestyles='dashed')
                        plt.savefig(save_path + '/' + spatial_firing.session_id.iloc[cluster_index] + '_grid_shuffle_dist_Cluster_' + str(cluster_id) + '.png', dpi=200)
                        plt.close()
                        plt.cla()

                        n, bins, patches = plt.hist(border_scores, 50, density=True, facecolor='b', alpha=0.75)
                        plt.vlines(border_score, ymin=0, ymax=max(n), colors='k', linestyles='solid')
                        plt.vlines(border_threshold, ymin=0, ymax=max(n), colors='k', linestyles='dashed')
                        plt.savefig(save_path + '/' + spatial_firing.session_id.iloc[cluster_index] + '_border_shuffle_dist_Cluster_' + str(cluster_id) + '.png', dpi=200)
                        plt.close()
                        plt.cla()

                        n, bins, patches = plt.hist(half_session_scores, 50, density=True, facecolor='b', alpha=0.75)
                        plt.vlines(half_session_score, ymin=0, ymax=max(n), colors='k', linestyles='solid')
                        plt.vlines(half_session_threshold, ymin=0, ymax=max(n), colors='k', linestyles='dashed')
                        plt.savefig(save_path + '/' + spatial_firing.session_id.iloc[cluster_index] + '_half_session_stability_shuffle_dist_Cluster_' + str(cluster_id) + '.png', dpi=200)
                        plt.close()
                        plt.cla()

                        n, bins, patches = plt.hist(spatial_information_scores, 50, density=True, facecolor='b', alpha=0.75)
                        plt.vlines(spatial_information_score, ymin=0, ymax=max(n), colors='k', linestyles='solid')
                        plt.vlines(spatial_threshold, ymin=0, ymax=max(n), colors='k', linestyles='dashed')
                        plt.savefig(save_path + '/' + spatial_firing.session_id.iloc[cluster_index] + '_spatial_information_shuffle_dist_Cluster_' + str(cluster_id) + '.png', dpi=200)
                        plt.close()
                        plt.cla()

                else:
                    print("There are no cells in this recordings")
            else:
                print("No spatial firing could be found")

        print("completed", recording_path)

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    recordings_folder_to_process = r"/mnt/datastore/Harry/Cohort8_may2021/of"
    plot_shuffle_distributions(recordings_folder_to_process)
    print("look now")

if __name__ == '__main__':
    main()