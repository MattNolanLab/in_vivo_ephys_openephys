import matplotlib.pylab as plt
import numpy as np
from scipy.stats.stats import pearsonr
import PostSorting.open_field_firing_maps
import pandas as pd
import open_ephys_IO
import os
from scipy import signal
from Edmond.Concatenate_from_server import *
from Edmond.summarise_experiment import *

def load_ephys_channel(recording_folder, ephys_channel, prm):
    print('Extracting ephys data')
    file_path = recording_folder + '/' + ephys_channel
    if os.path.exists(file_path):
        channel_data = open_ephys_IO.get_data_continuous(prm, file_path)
    else:
        print('Movement data was not found.')
    return channel_data

def process_lfp(recording_folder, prm):
    print("I am now processing the lfp")
    ephys_channels_list = prm.get_ephys_channels()

    lfp_df = pd.DataFrame()

    frequencies = []
    power_spectra = []
    channels = []

    for i in range(len(ephys_channels_list)):
        ephys_channel = ephys_channels_list[i]
        ephys_channel_data = load_ephys_channel(recording_folder, ephys_channel, prm)

        f, power_spectrum_channel = signal.welch(ephys_channel_data, fs=1000, nperseg=10000, scaling='spectrum')
        frequencies.append(f)
        power_spectra.append(power_spectrum_channel)
        channels.append(ephys_channel)

    lfp_df["channel"] = channels
    lfp_df["frequencies"] = frequencies
    lfp_df["power_spectra"] = power_spectra

    return lfp_df


def process_folder(recordings_base_folder, save_path=None):
    recording_paths = get_recording_paths([], recordings_base_folder)
    all_days_df = pd.DataFrame()
    all_days_df = add_full_session_id(all_days_df, recording_paths)
    all_days_df = add_session_identifiers(all_days_df)

    lfp_path = "/MountainSort/DataFrames/lfp_data.pkl"
    for path in recording_paths:
        data_frame_path = path+lfp_path

        print('Processing ' + data_frame_path)
        if os.path.exists(data_frame_path):
            try:
                print('I found a spatial data frame, processing ' + data_frame_path)
                lfp = pd.read_pickle(data_frame_path)
                # do something

                all_days_df = pd.concat([all_days_df, tmp_df], ignore_index=True)
                print('spatial firing data extracted from frame successfully')

            except Exception as ex:
                print('This is what Python says happened:')
                print(ex)
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_tb(exc_traceback)
                print('something went wrong, the recording might be missing dataframes!')

        else:
            print("I couldn't find a spatial firing dataframe")

    if save_path is not None:
        all_days_df.to_pickle(save_path+"/All_mice_lfp.pkl")
    print("completed all in list")
    return

def main():
    print("------------------------")

    recordings_base_folder = "/mnt/datastore/Harry/Cohort7_october2020/vr"
    process_folder(recordings_base_folder, save_path ="/mnt/datastore/Harry/Cohort7_october2020/summary/")

    recordings_base_folder = "/mnt/datastore/Harry/Cohort6_july2020/vr"
    process_folder(recordings_base_folder, save_path ="/mnt/datastore/Harry/Cohort6_july2020/summary/")

    print("look here")

if __name__ == '__main__':
    main()