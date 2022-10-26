import os
import numpy as np


def create_folder_for_output(prm):
    output_path = prm.get_output_path()
    new_folder = output_path + '/Interleaved'
    if os.path.exists(output_path + '/DataFrames') is False:
        os.makedirs(output_path + '/DataFrames')
    if os.path.exists(output_path + '/Figures_stim') is False:
        os.makedirs(output_path + '/Figures_stim')
    if os.path.exists(output_path + '/Figures_control') is False:
        os.makedirs(output_path + '/Figures_control')

    #TODO output output paths for plotting functions?


def find_stimulation_interval(opto_pulses, stimulation_frequency, sampling_rate):
    time_between_pulses = sampling_rate/stimulation_frequency  # interval between pulses in sampling points
    opto_end_times = np.take(opto_pulses, np.where(np.diff(opto_pulses)[0] > time_between_pulses)[0]) # start of stimulation
    opto_start_times_from_second = np.take(opto_pulses, np.where(np.diff(opto_pulses)[0] > time_between_pulses)[0] + 1)
    opto_start_times = np.append(opto_pulses[0][0], opto_start_times_from_second)  # end of stimulation
    stimulation_length = (opto_end_times[0] - opto_start_times[0]) / sampling_rate  # in seconds
    interval_between_stimulations = (opto_start_times[1] - opto_end_times[0])/sampling_rate  # in seconds

    return opto_end_times, opto_start_times, stimulation_length, interval_between_stimulations


def analyse_interleaved_stimulation(spatial_firing, opto_pulses, stimulation_frequency, prm):
    create_folder_for_output(prm)
    opto_on, opto_off, stimulation_length, stimulation_interval = find_stimulation_interval(opto_pulses, prm.sampling_rate)

    # chop data frames and re-index (?)
    # create open-field plots for stim
    # create open-field plots for control
    # create new df containing key values for each condition
    # save df as .pkl
    pass