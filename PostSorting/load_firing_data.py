import mdaio
import numpy as np
import os
import pandas as pd
import PreClustering.dead_channels
import data_frame_utility
import pickle
import setting

def get_firing_info(firing_times_path):
    units_list = None
    firing_info = None
    if os.path.exists(firing_times_path):
        firing_info = mdaio.readmda(firing_times_path)
        units_list = np.unique(firing_info[2])
    else:
        print('I could not find the MountainSort output [firing.mda] file.')
    return units_list, firing_info


# if the recording has dead channels, detected channels need to be shifted to get read channel ids
def correct_detected_ch_for_dead_channels(dead_channels, primary_channels):
    for dead_channel in dead_channels:
        indices_to_add_to = np.where(primary_channels >= dead_channel)
        primary_channels[indices_to_add_to] += 1
    return primary_channels


def correct_for_dead_channels(primary_channels, prm):
    PreClustering.dead_channels.get_dead_channel_ids(prm)
    dead_channels = prm.get_dead_channels()
    if len(dead_channels) != 0:
        dead_channels = list(map(int, dead_channels[0]))
        primary_channels = correct_detected_ch_for_dead_channels(dead_channels, primary_channels)
    return primary_channels

def process_firing_times2(session_id, sorter_data_path, session_type):
    #Read from sorter and create a dataframe to store the experiments values

    sorter = pickle.load(open(sorter_data_path,'rb'))
    cluster_ids = sorter.get_unit_ids()
    
    if session_type == 'openfield' and prm.get_opto_tagging_start_index() is not None:
        #TODO implement the openfield processing
        pass
    else:
        #TODO: should be implement as a list of dataframe instead
        firing_data = data_frame_utility.df_empty(['session_id', 'cluster_id', 'tetrode', 
            'primary_channel', 'firing_times', 'trial_number', 'trial_type', 'number_of_spikes', 'mean_firing_rate'], 
            dtypes=[str, np.uint8, np.uint8, np.uint8, np.uint64, np.uint8, np.uint16, np.float, np.float])

        for id in cluster_ids:
            cluster_firings = sorter.get_unit_spike_train(id)
            ch = sorter.get_unit_property(id,'max_channel')
            tetrode  = ch//setting.num_tetrodes
            num_spikes = sorter.get_unit_property(id,'number_of_spikes')
            mean_rate = sorter.get_unit_property(id, 'mean_firing_rate')

            firing_data = firing_data.append({
                    "session_id": session_id,
                    "cluster_id":  int(id),
                    "tetrode": tetrode,
                    "primary_channel": ch,
                    "firing_times": cluster_firings,
                    'number_of_spikes': num_spikes,
                    'mean_firing_rate': mean_rate
                }, ignore_index=True)

    return firing_data

def process_firing_times(session_id,firing_data_path, session_type):
    #TODO probably easier to use the sorterextractor object directly
    session_id = session_id
    units_list, firing_info = get_firing_info(firing_data_path)
    cluster_ids = firing_info[2]
    firing_times = firing_info[1]
    primary_channel = firing_info[0]
    # primary_channel = correct_for_dead_channels(primary_channel, prm)
    if session_type == 'openfield' and prm.get_opto_tagging_start_index() is not None:
        firing_data = data_frame_utility.df_empty(['session_id', 'cluster_id', 'tetrode', 'primary_channel', 'firing_times', 'firing_times_opto'], dtypes=[str, np.uint8, np.uint8, np.uint8, np.uint64, np.uint64])
        for cluster in units_list:
            cluster_firings_all = firing_times[cluster_ids == cluster]
            cluster_firings = np.take(cluster_firings_all, np.where(cluster_firings_all < prm.get_opto_tagging_start_index())[0])
            cluster_firings_opto = np.take(cluster_firings_all, np.where(cluster_firings_all >= prm.get_opto_tagging_start_index())[0])
            channel_detected = primary_channel[cluster_ids == cluster][0]
            tetrode = int((channel_detected-1)/4 + 1)
            ch = int((channel_detected - 1) % 4 + 1)
            firing_data = firing_data.append({
                "session_id": session_id,
                "cluster_id":  int(cluster),
                "tetrode": tetrode,
                "primary_channel": ch,
                "firing_times": cluster_firings,
                "firing_times_opto": cluster_firings_opto
            }, ignore_index=True)
    else:
        firing_data = data_frame_utility.df_empty(['session_id', 'cluster_id', 'tetrode', 'primary_channel', 'firing_times', 'trial_number', 'trial_type'], dtypes=[str, np.uint8, np.uint8, np.uint8, np.uint64, np.uint8, np.uint16])
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


def create_firing_data_frame(recording_to_process, session_type, prm):
    spike_data = None
    spike_data = process_firing_times(recording_to_process, session_type, prm)
    return spike_data

