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
    dir = [f.path for f in os.scandir(recording_to_sort)]

    for filepath in dir:
        filename = filepath.split("/")[-1]

        if filename.startswith(prm.get_continuous_file_name()):
            ch = OpenEphys.loadContinuous(recording_to_sort + '/' + filename)
            ch['data'] = ch['data'][:stitch_point]
            ch['timestamps'] = ch['timestamps'][:stitch_point]
            ch['recordingNumber'] = ch['recordingNumber'][:stitch_point]
            OpenEphys.writeContinuousFile(filepath, ch['header'], ch['timestamps'], ch['data'], ch['recordingNumber'])

    return recording_to_sort

def stitch_recordings(recording_to_sort, paired_recording):
    init_params()
    file_utility.set_continuous_data_path(prm)

    dir = [f.path for f in os.scandir(recording_to_sort)]

    for filepath in dir:
        filename = filepath.split("/")[-1]

        if filename.startswith(prm.get_continuous_file_name()):
            ch = OpenEphys.loadContinuous(recording_to_sort + '/' + filename)
            ch_p = OpenEphys.loadContinuous(paired_recording + '/' + filename)
            stitch_point = len(ch['data'])
            ch['data'] = np.append(ch['data'], ch_p['data'])
            ch['timestamps'] = np.append(ch['timestamps'], ch_p['timestamps'])
            ch['recordingNumber'] = np.append(ch['recordingNumber'], ch_p['recordingNumber'])

            OpenEphys.writeContinuousFile(filepath, ch['header'], ch['timestamps'], ch['data'], ch['recordingNumber'])

    return recording_to_sort, stitch_point

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
