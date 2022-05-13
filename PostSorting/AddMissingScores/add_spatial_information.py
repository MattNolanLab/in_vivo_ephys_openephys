import numpy as np
import pandas as pd
import os
import PostSorting.open_field_firing_maps
import sys
sys.setrecursionlimit(100000)


def add_spatial_information_to_spatial_firing(recording_to_process):
    print('I will add spatial information scores for all cells in this folder: ' + recording_to_process)
    path_to_spatial_firing = recording_to_process+"/MountainSort/DataFrames/spatial_firing.pkl"
    if os.path.exists(path_to_spatial_firing):
        spatial_firing = pd.read_pickle(path_to_spatial_firing)

        print(path_to_spatial_firing)
        print(len(spatial_firing))
        try:

            if not "spatial_information_score" in list(spatial_firing):
                position_heatmap = np.load(recording_to_process + "/MountainSort/DataFrames/position_heat_map.npy")
                spatial_firing = PostSorting.open_field_firing_maps.calculate_spatial_information(spatial_firing, position_heatmap)
                spatial_firing.to_pickle(recording_to_process+"/MountainSort/DataFrames/spatial_firing.pkl")
        except:
            print("stop here")


def process_recordings(recording_list):
    for recording in recording_list:
        add_spatial_information_to_spatial_firing(recording)
    print("all recordings processed")


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    # get list of all recordings in the recordings folder
    recording_list = []
    #recording_list.extend([f.path for f in os.scandir("/mnt/datastore/Klara/CA1_to_deep_MEC_in_vivo/") if f.is_dir()])
    #recording_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort8_may2021/vr/") if f.is_dir()])
    #recording_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort7_october2020/of/") if f.is_dir()])
    #recording_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort7_october2020/vr/") if f.is_dir()])
    #recording_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort6_july2020/of/") if f.is_dir()])
    #recording_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort6_july2020/vr/") if f.is_dir()])

    process_recordings(recording_list)
    print("look now")


if __name__ == '__main__':
    main()