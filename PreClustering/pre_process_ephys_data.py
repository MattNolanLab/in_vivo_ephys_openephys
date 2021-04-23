import PreClustering.dead_channels
import PreClustering.make_sorting_database
import PreClustering.parameters
import numpy as np
import OpenEphys
import os

import file_utility
from PreClustering import convert_open_ephys_to_mda

prm = PreClustering.parameters.Parameters()


def init_params():
    prm.set_sampling_rate(30000)
    prm.set_num_tetrodes(4)
    prm.set_movement_ch('100_ADC2.continuous')
    prm.set_opto_ch('100_ADC3.continuous')
    # file_utility.init_data_file_names(prm, '105_CH', '_0')  # old files
    file_utility.init_data_file_names(prm, '100_CH', '')  # currently used (2018)
    prm.set_waveform_size(40)

    # These are not exclusive, both can be True for the same recording - that way it'll be sorted twice
    prm.set_is_tetrode_by_tetrode(False)  # set to True if you want the spike sorting to be done tetrode by tetrode
    prm.set_is_all_tetrodes_together(True)  # set to True if you want the spike sorting done on all tetrodes combined


def split_back(recording_to_sort, stitch_point):
    """
    :param recording_to_sort: Path to recording #1 that is sorted together with other recordings
    :param stitch_point: length of recordings
    :return: the path (same as input parameter) and the total number of time steps in the combined data
    """
    print('I will split the data that was sorted together. It might take a while.')
    dir = [f.path for f in os.scandir(recording_to_sort)]
    first_stitch_point = stitch_point[0]
    n_timestamps = 0
    for filepath in dir:
        filename = filepath.split("/")[-1]

        if filename.startswith(prm.get_continuous_file_name()):
            ch = OpenEphys.loadContinuous(recording_to_sort + '/' + filename)

            # this calculates total sample length of recordings
            if n_timestamps == 0:
                n_timestamps = len(ch["data"])

            ch['data'] = ch['data'][:first_stitch_point]
            ch['timestamps'] = ch['timestamps'][:first_stitch_point]
            ch['recordingNumber'] = ch['recordingNumber'][:first_stitch_point]
            OpenEphys.writeContinuousFile(filepath, ch['header'], ch['timestamps'], ch['data'], ch['recordingNumber'])
    
    return recording_to_sort, n_timestamps


def stitch_recordings(recording_to_sort, paired_recordings):
    """
    Load continuous data from multiple recordings, concatenate the arrays and write new continuous files.
    :param recording_to_sort: path to recording #1
    :param paired_recordings: path list of recordings to sort together with recording #1
    :return: combined recording and time points where a new recording started
    """
    print('I will stitch these recordings together now so they can be sorted together. It might take a while.')
    init_params()
    file_utility.set_continuous_data_path(prm)

    directory_list = [f.path for f in os.scandir(recording_to_sort)]
    stitch_points = []
    added_first_stitch = False
    added_paired_stitch = False
    for filepath in directory_list:
        filename = filepath.split("/")[-1]
        if filename.startswith(prm.get_continuous_file_name()):
            ch = OpenEphys.loadContinuous(recording_to_sort + '/' + filename)
            if not added_first_stitch:
                length_of_recording = len(ch['data'])
                stitch_points.append(length_of_recording)
                added_first_stitch = True
            for recording in paired_recordings:
                ch_p = OpenEphys.loadContinuous(recording + '/' + filename)
                ch['data'] = np.append(ch['data'], ch_p['data'])
                ch['timestamps'] = np.append(ch['timestamps'], ch_p['timestamps'])
                ch['recordingNumber'] = np.append(ch['recordingNumber'], ch_p['recordingNumber'])
                if not added_paired_stitch:
                    length_of_other_recording = len(ch_p['data'])
                    previous_stitch = stitch_points[-1]
                    stitch_points.append(previous_stitch + length_of_other_recording)
            added_paired_stitch = True
            OpenEphys.writeContinuousFile(filepath, ch['header'], ch['timestamps'], ch['data'], ch['recordingNumber'])

    return recording_to_sort, stitch_points


# Prepares input for running spike sorting for the recording.
def process_a_dir(dir_name):
    print('')
    print('I am pre-processing data in {} before spike sorting.'.format(dir_name))
    prm.set_date(dir_name.rsplit('/', 2)[-2])

    prm.set_filepath(dir_name)
    file_utility.set_continuous_data_path(prm)

    PreClustering.dead_channels.get_dead_channel_ids(prm)  # read dead_channels.txt
    file_utility.create_folder_structure(prm)

    if prm.get_is_tetrode_by_tetrode() is True:
        print('------------------------------------------')
        print('I am making one mda file for each tetrode.')
        print('------------------------------------------')
        PreClustering.make_sorting_database.create_sorting_folder_structure_separate_tetrodes(prm)
        convert_open_ephys_to_mda.convert_continuous_to_mda(prm)
        print('All 4 tetrodes were converted to separate mda files.')
        print('*****************************************************')

    if prm.get_is_all_tetrodes_together() is True:
        print('-------------------------------------------------------------------------')
        print('I am converting all channels into one mda file. This will take some time.')
        print('-------------------------------------------------------------------------')
        PreClustering.make_sorting_database.create_sorting_folder_structure(prm)
        convert_open_ephys_to_mda.convert_all_tetrodes_to_mda(prm)
        print('The big mda file is created, it is in Electrophysiology' + prm.get_spike_sorter())
        print('***************************************************************************************')


def pre_process_data(dir_name, sorter_name='MountainSort'):
    init_params()
    prm.set_spike_sorter(sorter_name)
    process_a_dir(dir_name + '/')


# call main when you only want to run this through a folder with recordings without the rest of the pipeline
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    init_params()
    #recording_folder = 'C:/Users/s1466507/Documents/Ephys/recordings/M0_2017-12-14_15-00-13_of'
    #pre_process_data(recording_folder)

    recording_folder = r"C:\Users\44756\Desktop\test_recordings_waveform_matching\M2_D3_2019-03-06_13-35-15"
    paired_folder =    r"C:\Users\44756\Desktop\test_recordings_waveform_matching\M2_D3_2019-03-06_15-24-38"

    stitch_recordings(recording_folder, paired_folder)


if __name__ == '__main__':
    main()
