'''
Functions for converting open ephys data to mountainsort's mda format
'''


import os

import PreClustering.make_sorting_database
import numpy as np
import open_ephys_IO

import file_utility
import mdaio


def pad_event(waveforms, channel, samples_per_spike, event):
    padded_event = np.hstack((waveforms[event, :, channel], np.zeros(samples_per_spike)))
    return padded_event


def get_padded_array(waveforms, samples_per_spike):
    number_of_events = waveforms.shape[0]
    all_padded = np.zeros((4, samples_per_spike*2*number_of_events))

    for channel in range(4):
        padded_events_ch = np.zeros((number_of_events, samples_per_spike*2))
        for event in range(number_of_events):
            padded_event = pad_event(waveforms, channel, samples_per_spike, event)
            padded_events_ch[event, :] = padded_event
        padded_events_ch = padded_events_ch.flatten('C')
        all_padded[channel, :] = padded_events_ch

    too_big_indices = np.where(all_padded > 30000)
    too_small_indices = np.where(all_padded < -30000)

    all_padded[too_big_indices] = 30000
    all_padded[too_small_indices] = -30000

    all_padded = all_padded * (-1)*10000000

    all_padded = np.asarray(all_padded, dtype='int16')

    return all_padded


def get_peak_indices(waveforms, samples_per_spike):
    number_of_events = waveforms.shape[0]
    waveforms2d = np.zeros((number_of_events, 4*samples_per_spike))

    for event in range(number_of_events):
        waveforms2d[event, :] = waveforms[event, :, :].flatten()

    peak_indices_all_ch = np.argmax(waveforms2d, 1)
    peak_indices_in_wave = np.floor(peak_indices_all_ch/4)
    peak_indices_in_wave = np.asarray(peak_indices_in_wave, dtype='int16')
    peak_indices_all_events = peak_indices_in_wave + np.arange(0, number_of_events)*40
    peak_indices = np.array(peak_indices_all_events, dtype='int32')
    return peak_indices  # add event number to this, it needs the absolute index not 0-40


def convert_spk_to_mda(prm):
    file_utility.create_folder_structure(prm)
    folder_path = prm.get_filepath()
    spike_data_path = prm.get_spike_path() + '\\'
    number_of_tetrodes = prm.get_num_tetrodes()
    samples_per_spike = prm.get_waveform_size()

    if os.path.isfile(spike_data_path + 't1_' + prm.get_date() + '\\raw.nt1.mda') is False:
        file_utility.create_ephys_folder_structure(prm)

        for tetrode in range(number_of_tetrodes):
            file_path = folder_path + 'TT' + str(tetrode) + '.spikes'
            waveforms, timestamps = open_ephys_IO.get_data_spike(folder_path, file_path, 'TT' + str(tetrode + 1))
            np.save(spike_data_path + 't' + str(tetrode + 1) + '_' + prm.get_date() + '\\TT' + str(tetrode + 1) + '_timestamps', timestamps)  # todo: this is shifted by 10 seconds relative to light and location!

            padded_array = get_padded_array(waveforms, samples_per_spike)

            mdaio.writemda16i(padded_array, spike_data_path + 't' + str(tetrode + 1) + '_' + prm.get_date() + '\\raw.nt' + str(tetrode + 1) + '.mda')
            peak_indices = get_peak_indices(waveforms, samples_per_spike)
            mdaio.writemda32i(peak_indices, spike_data_path + 't' + str(tetrode + 1) + '_' + prm.get_date() + '\\event_times.nt' + str(tetrode + 1) + '.mda')

            mdaio.writemda32(timestamps, spike_data_path + 't' + str(tetrode + 1) + '_' + prm.get_date() + '\\timestamps.nt' + str(tetrode + 1) + '.mda')


def convert_continuous_to_mda(prm):
    file_utility.create_folder_structure(prm)
    # make_sorting_database.create_sorting_folder_structure(prm)
    number_of_tetrodes = prm.get_num_tetrodes()
    folder_path = prm.get_filepath()
    spike_data_path = prm.get_spike_path() + '\\'
    continuous_file_name = prm.get_continuous_file_name()

    if os.path.isfile(spike_data_path + 't1_' + prm.get_date() + '\\raw.mda') is False:
        file_utility.create_ephys_folder_structure(prm)

    for tetrode in range(number_of_tetrodes):
        channel_data_all = []
        for channel in range(4):
            file_path = folder_path + continuous_file_name + str(tetrode*4 + channel + 1) + '.continuous'
            channel_data = open_ephys_IO.get_data_continuous(file_path)
            channel_data_all.append(channel_data)

        recording_length = len(channel_data_all[0])
        channels_tetrode = np.zeros((4, recording_length))

        for ch in range(4):
            channels_tetrode[ch, :] = channel_data_all[ch]
        mdaio.writemda16i(channels_tetrode, spike_data_path + 't' + str(tetrode + 1) + '_' + prm.get_date() + '_continuous\\data\\raw.mda')


def convert_all_tetrodes_to_mda(prm):
    file_utility.create_folder_structure(prm)
    PreClustering.make_sorting_database.organize_files_for_ms(prm)
    number_of_tetrodes = prm.get_num_tetrodes()
    folder_path = prm.get_filepath()
    spike_data_path = prm.get_spike_path() + '\\'

    path = spike_data_path + 'all_tetrodes\\data\\raw.mda'

    file_path = folder_path + '100_CH' + str(1) + '.continuous'
    first_ch = open_ephys_IO.get_data_continuous(file_path)
    recording_length = len(first_ch)
    channels_all = np.zeros((number_of_tetrodes*4, recording_length))
    channels_all[0, :] = first_ch


    for channel in range(15):
        file_path = folder_path + '100_CH' + str(channel + 2) + '.continuous'
        channel_data = open_ephys_IO.get_data_continuous(file_path)
        channels_all[channel + 1, :] = channel_data

    mdaio.writemda16i(channels_all, path)












