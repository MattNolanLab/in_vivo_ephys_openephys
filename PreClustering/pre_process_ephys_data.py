import PreClustering.dead_channels
import PreClustering.make_sorting_database
import parameters
from scipy.signal import butter,filtfilt
from tqdm import tqdm
import file_utility
import numpy as np
from PreClustering import convert_open_ephys_to_mda

prm = parameters.Parameters()


def filterRecording(recording, sampling_freq, lp_freq=300,hp_freq=6000,order=3):
    fn = sampling_freq / 2.
    band = np.array([lp_freq, hp_freq]) / fn

    b, a = butter(order, band, btype='bandpass')

    if not (np.all(np.abs(np.roots(a)) < 1) and np.all(np.abs(np.roots(a)) < 1)):
        raise ValueError('Filter is not stable')
    
    for i in tqdm(range(recording._timeseries.shape[0])):
        recording._timeseries[i,:] = filtfilt(b,a,recording._timeseries[i,:])

    return recording

def init_params():
    
    # file_utility.init_data_file_names(prm, '105_CH', '_0')  # old files
    file_utility.init_data_file_names(prm, '100_CH', '')  # currently used (2018)

    return prm


# Prepares input for running spike sorting for the recording.
def process_a_dir(prm,dir_name):
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
    recording_folder = 'C:/Users/s1466507/Documents/Ephys/recordings/M5_2018-03-06_15-34-44_of'
    pre_process_data(recording_folder)


if __name__ == '__main__':
    main()