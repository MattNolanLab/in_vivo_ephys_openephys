import open_ephys_IO
import os
import numpy as np
import pandas as pd
from scipy import stats
import PostSorting.parameters
import PostSorting.load_snippet_data_opto

import PostSorting.open_field_make_plots
# import PostSorting.SALT

ignore_opto = False  # one cannot simply ignore the opto


def load_opto_data(recording_to_process, prm):
    is_found = False
    opto_data = None
    print('loading opto channel...')
    file_path = recording_to_process + '/' + prm.get_opto_channel()
    if os.path.exists(file_path):
        opto_data = open_ephys_IO.get_data_continuous(prm, file_path)
        is_found = True
    else:
        print('Opto data was not found.')
    return opto_data, is_found


def get_ons_and_offs(opto_data):
    # opto_on = np.where(opto_data > np.min(opto_data) + 10 * np.std(opto_data))
    # opto_off = np.where(opto_data <= np.min(opto_data) + 10 * np.std(opto_data))
    mode = stats.mode(opto_data[::30000])[0][0]
    opto_on = np.where(opto_data > 0.2 + mode)
    opto_off = np.where(opto_data <= 0.2 + mode)
    return opto_on, opto_off


def process_opto_data(recording_to_process, prm):
    opto_on = opto_off = None
    opto_data, is_found = load_opto_data(recording_to_process, prm)
    if is_found:
        opto_on, opto_off = get_ons_and_offs(opto_data)
        if not np.asarray(opto_on).size:
            prm.set_opto_tagging_start_index(None)
            is_found = None
        else:
            first_opto_pulse_index = min(opto_on[0])
            prm.set_opto_tagging_start_index(first_opto_pulse_index)

    else:
        prm.set_opto_tagging_start_index(None)

    return opto_on, opto_off, is_found


def make_opto_data_frame(opto_on: tuple) -> pd.DataFrame:
    opto_data_frame = pd.DataFrame()
    opto_end_times = np.take(opto_on, np.where(np.diff(opto_on)[0] > 1))
    opto_start_times_from_second = np.take(opto_on, np.where(np.diff(opto_on)[0] > 1)[0] + 1)
    opto_start_times = np.append(opto_on[0][0], opto_start_times_from_second)
    opto_data_frame['opto_start_times'] = opto_start_times
    opto_end_times = np.append(opto_end_times, opto_on[0][-1])
    opto_data_frame['opto_end_times'] = opto_end_times
    return opto_data_frame


def check_parity_of_window_size(window_size_ms):
    if window_size_ms % 2 != 0:
        print("Window size must be divisible by 2")
        assert window_size_ms % 2 == 0


def get_on_pulse_times(prm):
    path_to_pulses = prm.get_output_path() + '/DataFrames/opto_pulses.pkl'
    pulses = pd.read_pickle(path_to_pulses)
    on_pulses = pulses.opto_start_times
    return on_pulses


def get_firing_times(cell):
    if 'firing_times_opto' in cell:
        firing_times = np.append(cell.firing_times, cell.firing_times_opto)
    else:
        firing_times = cell.firing_times
    return firing_times


def find_spike_positions_in_window(pulse, firing_times, window_size_sampling_rate):
    spikes_in_window_binary = np.zeros(window_size_sampling_rate)
    window_start = int(pulse - window_size_sampling_rate / 2)
    window_end = int(pulse + window_size_sampling_rate / 2)
    spikes_in_window_indices = np.where((firing_times > window_start) & (firing_times < window_end))
    spike_times = np.take(firing_times, spikes_in_window_indices)[0]
    position_of_spikes = spike_times.astype(int) - window_start
    spikes_in_window_binary[position_of_spikes] = 1
    return spikes_in_window_binary


def make_df_to_append_for_pulse(session_id, cluster_id, spikes_in_window_binary, window_size_sampling_rate):
    columns = np.append(['session_id', 'cluster_id'], range(window_size_sampling_rate))
    df_row = np.append([session_id, cluster_id], spikes_in_window_binary.astype(int))
    df_to_append = pd.DataFrame([(df_row)], columns=columns)
    return df_to_append


def get_peristumulus_opto_data(window_size_ms, prm):
    print('Get data for peristimulus array.')
    check_parity_of_window_size(window_size_ms)
    on_pulses = get_on_pulse_times(prm)  # these are the start times of the pulses
    sampling_rate = prm.get_sampling_rate()
    window_size_sampling_rate = int(sampling_rate/1000 * window_size_ms)
    return on_pulses, window_size_sampling_rate


def make_peristimulus_df(spatial_firing, on_pulses, window_size_sampling_rate, prm):
    print('Make peristimulus data frame.')
    peristimulus_spikes_path = prm.get_output_path() + '/DataFrames/peristimulus_spikes.pkl'
    columns = np.append(['session_id', 'cluster_id'], range(window_size_sampling_rate))
    peristimulus_spikes = pd.DataFrame(columns=columns)

    for index, cell in spatial_firing.iterrows():
        session_id = cell.session_id
        cluster_id = cell.cluster_id
        if len(on_pulses) >= 500:
            on_pulses = on_pulses[-500:]  # only look at last 500 to make sure it's just the opto tagging
        for pulse in on_pulses:
            firing_times = get_firing_times(cell)
            spikes_in_window_binary = find_spike_positions_in_window(pulse, firing_times, window_size_sampling_rate)
            df_to_append = make_df_to_append_for_pulse(session_id, cluster_id, spikes_in_window_binary, window_size_sampling_rate)
            peristimulus_spikes = peristimulus_spikes.append(df_to_append)
    peristimulus_spikes.to_pickle(peristimulus_spikes_path)
    return peristimulus_spikes


def create_baseline_and_test_epochs(peristimulus_spikes):
    pass


def add_first_spike_times_after_stimulation(spatial_firing, on_pulses, sampling_rate=30000):
    # Identifies first spike firing times and latencies and makes columns ('spike_times_after_opto' and 'latencies')
    print('I will find the first spikes after the light for each opto stimulation pulse.')
    first_spikes_times = []
    latencies = []
    for cluster_index, cluster in spatial_firing.iterrows():
        firing_times = cluster.firing_times_opto
        first_spikes_times_cell = []
        latencies_cell = []
        for pulse in on_pulses:
            if len(firing_times[firing_times > pulse]) > 0:
                first_spike_after_pulse = firing_times[firing_times > pulse][0]
                latency = first_spike_after_pulse - pulse
                if latency > sampling_rate / 100:
                    latency = np.nan  # the spike is more than 10 ms after the pulse
            else:
                first_spike_after_pulse = np.nan
                latency = np.nan
            first_spikes_times_cell.append(first_spike_after_pulse)
            latencies_cell.append(latency)
        first_spikes_times.append(first_spikes_times_cell)
        latencies.append(latencies_cell)
    spatial_firing['spike_times_after_opto'] = first_spikes_times
    spatial_firing['opto_latencies'] = latencies
    return spatial_firing


def analyse_latencies(spatial_firing, prm):
    print('Analyse latencies.')
    sampling_rate = prm.get_sampling_rate()
    latencies_mean_ms = []
    latencies_sd_ms = []

    for cluster_index, cluster in spatial_firing.iterrows():
        mean = np.nanmean(cluster.opto_latencies) / sampling_rate * 1000
        sd = np.nanstd(cluster.opto_latencies) / sampling_rate * 1000
        latencies_mean_ms.append(mean)
        latencies_sd_ms.append(sd)
    spatial_firing['opto_latencies_mean_ms'] = latencies_mean_ms
    spatial_firing['opto_latencies_sd_ms'] = latencies_sd_ms
    return spatial_firing


def process_spikes_around_light(spatial_firing, prm, window_size_ms=40):
    print('I will process opto data.')
    on_pulses, window_size_sampling_rate = get_peristumulus_opto_data(window_size_ms, prm)
    peristimulus_spikes = make_peristimulus_df(spatial_firing, on_pulses, window_size_sampling_rate, prm)
    spatial_firing = add_first_spike_times_after_stimulation(spatial_firing, on_pulses)
    spatial_firing = analyse_latencies(spatial_firing, prm)
    spatial_firing = PostSorting.load_snippet_data_opto.get_opto_snippets(spatial_firing, prm, random_snippets=True, column_name='first_spike_snippets_opto', firing_times_column='spike_times_after_opto')
    # plt.plot((peristimulus_spikes.iloc[:, 2:].astype(int)).sum().rolling(50).sum())
    # baseline, test = create_baseline_and_test_epochs(peristimulus_spikes)
    # latencies, p_values, I_values = salt(baseline_trials, test_trials, winsize=0.01 * pq.s, latency_step=0.01 * pq.s)

    return spatial_firing


def main():
    # recording_folder = '/Users/briannavandrey/Documents/recordings'
    recording_folder = 'C:/Users/s1466507/Documents/Work/opto/M2_2021-02-17_18-07-42_of'
    # C:\Users\s1466507\Documents\Work\opto\M2_2021-02-17_18-07-42_of\MountainSort\DataFrames
    prm = PostSorting.parameters.Parameters()
    prm.set_output_path(recording_folder + '/MountainSort')
    prm.set_sampling_rate(30000)
    spikes_path = prm.get_output_path() + '/DataFrames/spatial_firing.pkl'
    spikes = pd.read_pickle(spikes_path)
    process_spikes_around_light(spikes, prm)


if __name__ == '__main__':
    main()



