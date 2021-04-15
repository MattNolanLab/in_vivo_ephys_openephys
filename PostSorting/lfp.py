import matplotlib.pylab as plt
import numpy as np
from scipy.stats.stats import pearsonr
import PostSorting.open_field_firing_maps
import pandas as pd
import open_ephys_IO
import os
from scipy import signal

def load_ephys_channel(recording_folder, ephys_channel, prm):
    print('Extracting ephys data')
    file_path = recording_folder + '/' + ephys_channel
    if os.path.exists(file_path):
        channel_data = open_ephys_IO.get_data_continuous(file_path)
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

def main():
    print("------------------------")


    print("look here")

if __name__ == '__main__':
    main()