import pandas as pd
import numpy as np
import os

def add_shuffled_cutoffs(recordings_folder_to_process):

    recording_list = [f.path for f in os.scandir(recordings_folder_to_process) if f.is_dir()]

    for recording_path in recording_list:
        print("processing ", recording_path)
        if os.path.isfile(recording_path+r"/MountainSort/DataFrames/shuffles/shuffle.pkl"):
            print("I have found a shuffled dataframe")
            shuffle = pd.read_pickle(recording_path+r"/MountainSort/DataFrames/shuffles/shuffle.pkl")
            spatial_firing = pd.read_pickle(recording_path+r"/MountainSort/DataFrames/spatial_firing.pkl")

            print("There are", int(len(shuffle)/len(spatial_firing)), "shuffles per cell")

            speed_threshold_poss = []
            speed_threshold_negs = []
            hd_thresholds = []
            rayleigh_thresholds = []
            spatial_thresholds = []
            grid_thresholds = []
            border_thresholds = []

            for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
                cluster_shuffle_df = shuffle[(shuffle.cluster_id == cluster_id)] # dataframe for that cluster

                # calculate the 95th percentile threshold for individual clusters
                # calculations based on z values please see https://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_probability/bs704_probability10.html
                adjusted_speed_threshold_pos = np.nanmean(cluster_shuffle_df["speed_score"]) + (np.nanstd(cluster_shuffle_df["speed_score"])*+1.960) # two tailed
                adjusted_speed_threshold_neg = np.nanmean(cluster_shuffle_df["speed_score"]) + (np.nanstd(cluster_shuffle_df["speed_score"])*-1.960) # two tailed
                adjusted_hd_threshold = np.nanmean(cluster_shuffle_df["hd_score"]) + (np.nanstd(cluster_shuffle_df["hd_score"])*1.645) # one tailed
                adjusted_rayleigh_threshold = np.nanmean(cluster_shuffle_df["rayleigh_score"]) + (np.nanstd(cluster_shuffle_df["rayleigh_score"])*-1.645) # one tailed
                adjusted_spatial_threshold = np.nanmean(cluster_shuffle_df["spatial_information_score"]) + (np.nanstd(cluster_shuffle_df["spatial_information_score"])*1.645) # one tailed
                adjusted_grid_threshold = np.nanmean(cluster_shuffle_df["grid_score"]) + (np.nanstd(cluster_shuffle_df["grid_score"])*1.645) # one tailed
                adjusted_border_threshold = np.nanmean(cluster_shuffle_df["border_score"]) + (np.nanstd(cluster_shuffle_df["border_score"])*1.645) # one tailed

                speed_threshold_poss.append(adjusted_speed_threshold_pos)
                speed_threshold_negs.append(adjusted_speed_threshold_neg)
                hd_thresholds.append(adjusted_hd_threshold)
                rayleigh_thresholds.append(adjusted_rayleigh_threshold)
                spatial_thresholds.append(adjusted_spatial_threshold)
                grid_thresholds.append(adjusted_grid_threshold)
                border_thresholds.append(adjusted_border_threshold)

            spatial_firing["speed_threshold_pos"] = speed_threshold_poss
            spatial_firing["speed_threshold_neg"] = speed_threshold_negs
            spatial_firing["hd_threshold"] = hd_thresholds
            spatial_firing["rayleigh_threshold"] = rayleigh_thresholds
            spatial_firing["spatial_threshold"] = spatial_thresholds
            spatial_firing["grid_threshold"] = grid_thresholds
            spatial_firing["border_threshold"] = border_thresholds

            spatial_firing.to_pickle(recording_path+r"/MountainSort/DataFrames/spatial_firing.pkl")

def add_spatial_classifier_based_on_cutoffs(recordings_folder_to_process):
    recording_list = [f.path for f in os.scandir(recordings_folder_to_process) if f.is_dir()]

    for recording_path in recording_list:
        print("processing ", recording_path)
        if os.path.isfile(recording_path+r"/MountainSort/DataFrames/spatial_firing.pkl"):
            spatial_firing = pd.read_pickle(recording_path+r"/MountainSort/DataFrames/spatial_firing.pkl")

            grid_cells = []
            border_cells = []
            hd_cells = []
            hd_cells_rayleigh = []
            spatial_cells = []
            speed_cells = []

            for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
                cluster_spatial_firing = spatial_firing[(spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster

                if (cluster_spatial_firing["grid_score"].iloc[0] > cluster_spatial_firing["grid_threshold"].iloc[0]):
                    grid_cell = True
                else:
                    grid_cell = False

                if (cluster_spatial_firing["border_score"].iloc[0] > cluster_spatial_firing["border_threshold"].iloc[0]):
                    border_cell = True
                else:
                    border_cell = False

                if (cluster_spatial_firing["hd_score"].iloc[0] > cluster_spatial_firing["hd_threshold"].iloc[0]):
                    hd_cell = True
                else:
                    hd_cell = False

                if (cluster_spatial_firing["rayleigh_score"].iloc[0] < cluster_spatial_firing["rayleigh_threshold"].iloc[0]):
                    hd_cell_rayleigh = True
                else:
                    hd_cell_rayleigh = False

                if (cluster_spatial_firing["spatial_information_score"].iloc[0] > cluster_spatial_firing["spatial_threshold"].iloc[0]):
                    spatial_cell = True
                else:
                    spatial_cell = False

                if (cluster_spatial_firing["speed_score"].iloc[0] > cluster_spatial_firing["speed_threshold_pos"].iloc[0]):
                    speed_cell = True
                elif (cluster_spatial_firing["speed_score"].iloc[0] < cluster_spatial_firing["speed_threshold_neg"].iloc[0]):
                    speed_cell = True
                else:
                    speed_cell = False

                grid_cells.append(grid_cell)
                border_cells.append(border_cell)
                hd_cells.append(hd_cell)
                hd_cells_rayleigh.append(hd_cell_rayleigh)
                spatial_cells.append(spatial_cell)
                speed_cells.append(speed_cell)

            spatial_firing["grid_cell"] = grid_cells
            spatial_firing["border_cell"] = border_cells
            spatial_firing["hd_cell"] = hd_cells
            spatial_firing["hd_cell_rayleigh"] = hd_cells_rayleigh
            spatial_firing["spatial_cell"] = spatial_cells
            spatial_firing["speed_cell"] = speed_cells

            spatial_firing.to_pickle(recording_path+r"/MountainSort/DataFrames/spatial_firing.pkl")

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    recordings_folder_to_process = r"/mnt/datastore/Harry/test_recordings"
    add_shuffled_cutoffs(recordings_folder_to_process)
    add_spatial_classifier_based_on_cutoffs(recordings_folder_to_process)

    print("look now")

if __name__ == '__main__':
    main()