import mdaio
import numpy as np
import os
import pandas as pd


def get_firing_info(file_path):
    firing_times_path = file_path + '\\Spike_sorting\\all_tetrodes\\data\\firings.mda'
    units_list = None
    firing_info = None
    if os.path.exists(firing_times_path):
        firing_info = mdaio.readmda(firing_times_path)
        units_list = np.unique(firing_info[2])
    return units_list, firing_info


def process_firing_times(recording_to_process):
    session_id = recording_to_process.split('/')[-1]
    units_list, firing_info = get_firing_info(recording_to_process)
    cluster_ids = firing_info[2]
    firing_times = firing_info[1]
    primary_channel = firing_info[0]
    firing_data = pd.DataFrame(columns=['session_id', 'cluster_id', 'tetrode', 'primary_channel', 'firing_times'])
    for cluster in units_list:
        cluster_firings = firing_times[cluster_ids == cluster]
        channel_detected = primary_channel[cluster_ids == cluster][0]
        tetrode = int((channel_detected-1)/4 + 1)
        ch = int((channel_detected - 1) % 4 + 1)
        firing_data = firing_data.append({
            "session_id": session_id,
            "cluster_id":  int(cluster),
            "tetrode": tetrode,
            "primary_channel": ch,
            "firing_times": cluster_firings
        }, ignore_index=True)
    return firing_data


def create_firing_data_frame(recording_to_process):
    spike_data = None
    spike_data = process_firing_times(recording_to_process)
    return spike_data

