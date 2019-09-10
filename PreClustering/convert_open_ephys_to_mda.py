'''
Functions for converting open ephys data to mountainsort's mda format

'''

import glob
import os

import PreClustering.dead_channels
import PreClustering.make_sorting_database
import numpy as np
import open_ephys_IO

import file_utility
import mdaio


def convert_continuous_to_mda(prm):
    '''
    Convert continous data from open ephys to mda format
    '''
    file_utility.create_folder_structure(prm)
    file_utility.folders_for_separate_tetrodes(prm)
    number_of_tetrodes = prm.get_num_tetrodes()
    folder_path = prm.get_filepath()
    spike_data_path = prm.get_spike_path() + '/'
    continuous_file_name = prm.get_continuous_file_name()
    continuous_file_name_end = prm.get_continuous_file_name_end()

    raw_mda_file_path = file_utility.get_raw_mda_path_separate_tetrodes(prm)

    for tetrode in range(number_of_tetrodes):
        live_channels = PreClustering.dead_channels.get_list_of_live_channels(prm, tetrode)
        number_of_live_ch_in_tetrode = 0
        if os.path.isfile(spike_data_path + 't' + str(tetrode + 1) + raw_mda_file_path) is False:
            channel_data_all = []
            if len(live_channels) >= 2:
                # search for the live channels in each tetrode and load them

                for channel in range(4):
                    if (channel + 1) in live_channels:
                        number_of_live_ch_in_tetrode += 1
                        file_path = folder_path + continuous_file_name + str(tetrode*4 + channel + 1) + continuous_file_name_end + '.continuous'
                        channel_data = open_ephys_IO.get_data_continuous(prm, file_path)
                        channel_data_all.append(channel_data)

                recording_length = len(channel_data_all[0])
                #TODO: there should be a more efficient way to implement this
                channels_tetrode = np.zeros((number_of_live_ch_in_tetrode, recording_length))
                for ch in range(number_of_live_ch_in_tetrode):
                    channels_tetrode[ch, :] = channel_data_all[ch]
                    
                #write to the bag MDA file
                mdaio.writemda16i(channels_tetrode, spike_data_path + 't' + str(tetrode + 1) + raw_mda_file_path)
            else:
                print('The number of live channels is fewer than 2 in this tetrode so I will not sort it.')

        else:
            print('This tetrode is already converted to mda, I will move on and check the next one. ' + spike_data_path + 't' + str(tetrode + 1) + '\\data\\raw.mda')


def try_to_figure_out_non_default_file_names(folder_path, ch_num):
    beginning = glob.glob(folder_path + '*.continuous')[0].split('/')[-1].split('_')[0]
    end = glob.glob(folder_path + '*.continuous')[0].split('/')[-1].split('CH')[-1].split('.')[0].split('_')[1:]
    if len(end) == 2:
        file_path = folder_path + beginning + '_CH' + str(ch_num) + '_' + end[0] + '_' + end[1] + '.continuous'
    else:
        file_path = folder_path + beginning + '_CH' + str(ch_num) + '_' + end[0] + '.continuous'
    return file_path


# this is for putting all tetrodes in the same mda file
def convert_all_tetrodes_to_mda(prm):
    raw_mda_path = file_utility.get_raw_mda_path_all_channels(prm)
    if os.path.isfile(raw_mda_path) is False:
        file_utility.create_folder_structure(prm)
        PreClustering.make_sorting_database.create_sorting_folder_structure(prm)
        folder_path = prm.get_filepath()
        continuous_file_name = prm.get_continuous_file_name()
        continuous_file_name_end = prm.get_continuous_file_name_end()

        path = raw_mda_path

        file_path = folder_path + continuous_file_name + str(1) + continuous_file_name_end + '.continuous'
        if os.path.exists(file_path):
            first_ch = open_ephys_IO.get_data_continuous(prm, file_path)
        else:
            file_path = try_to_figure_out_non_default_file_names(folder_path, 1)

            first_ch = open_ephys_IO.get_data_continuous(prm, file_path)

        live_channels = PreClustering.dead_channels.get_list_of_live_channels_all_tetrodes(prm)
        number_of_live_channels = len(live_channels)

        recording_length = len(first_ch)
        channels_all = np.zeros((number_of_live_channels, recording_length))

        live_ch_counter = 0
        for channel in range(16):
            if (channel + 1) in live_channels:
                file_path = folder_path + continuous_file_name + str(channel + 1) + continuous_file_name_end + '.continuous'
                if os.path.exists(file_path):
                    channel_data = open_ephys_IO.get_data_continuous(prm, file_path)
                else:
                    file_path = try_to_figure_out_non_default_file_names(folder_path, channel + 1)
                    channel_data = open_ephys_IO.get_data_continuous(prm, file_path)

                channels_all[live_ch_counter, :] = channel_data
                live_ch_counter += 1

        mdaio.writemda16i(channels_all, path)
    else:
        print('The mda file that contains all channels is already in Electrophysiology/Spike_sorting/all_tetrodes/data.'
              ' You  need to delete it if you want me to make it again.')












