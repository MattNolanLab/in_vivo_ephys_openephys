from fractions import Fraction
import open_ephys_IO
import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy import signal


def load_sync_data_ephys(recording_to_process, prm):
    is_found = False
    sync_data = None
    print('loading sync channel...')
    file_path = recording_to_process + '/' + prm.get_sync_channel()
    if os.path.exists(file_path):
        sync_data = open_ephys_IO.get_data_continuous(prm, file_path)
        is_found = True
    else:
        print('Opto data was not found.')
    return sync_data, is_found


def get_video_sync_on_and_off_times(spatial_data):
    threshold = np.median(spatial_data['syncLED']) + 2 * np.std(spatial_data['syncLED'])
    spatial_data['sync_pulse_on'] = spatial_data['syncLED'] > threshold
    spatial_data['sync_pulse_on_diff'] = spatial_data['sync_pulse_on'].diff()
    return spatial_data


def get_ephys_sync_on_and_off_times(sync_data_ephys, prm):
    sync_data_ephys['on_index'] = sync_data_ephys['sync_pulse'] > 0.5
    sync_data_ephys['on_index_diff'] = sync_data_ephys['on_index'].diff()  # true when light turns on
    sync_data_ephys['time'] = sync_data_ephys.index / prm.get_sampling_rate()
    return sync_data_ephys


def check_correlation(sync_data_ephys, spatial_data):
    # plt.plot(sync_data_ephys['sync_pulse'], color='black')
    # plt.plot(spatial_data['syncLED'], color='cyan')

    ephys_on = sync_data_ephys['on_index_diff'] == 1
    on_times_ephys = sync_data_ephys[ephys_on].time
    video_sync_on = spatial_data['sync_pulse_on_diff'] == 1
    on_times_video = spatial_data[video_sync_on].time_seconds

    plt.plot(sync_data_ephys[ephys_on].time)
    lag = np.argmax(np.correlate(spatial_data['time_seconds'], sync_data_ephys_downsampled.time, "full"))
    on_times_ephys = np.roll(on_times_ephys, shift=int(np.ceil(lag)))
    plt.plot(on_times_ephys, on_times_video)

    plt.plot(spatial_data['sync_pulse_on_diff'])

    correlate = np.correlate(sync_data_ephys['sync_pulse'], spatial_data['syncLED'], "full")
    plt.plot(correlate, color='red')
    plt.show()


def correlate_signals(sync_data_ephys, spatial_data):
    print('I will synchronize the position and ephys data by shifting the position to match the ephys.')
    avg_sampling_rate_bonsai = float(1 / spatial_data['time_seconds'].diff().mean())
    avg_sampling_rate_open_ephys = float(1 / sync_data_ephys['time'].diff().mean())

    fraction = Fraction(avg_sampling_rate_bonsai/avg_sampling_rate_open_ephys).limit_denominator()  # rational fraction of sampling rates
    sync_data_ephys['time_downsample'] = sync_data_ephys['time'][::int(avg_sampling_rate_open_ephys/avg_sampling_rate_bonsai)]
    sync_data_ephys_downsampled = sync_data_ephys[pd.notnull(sync_data_ephys['time_downsample'])]
    lag = np.argmax(np.correlate(spatial_data['time_seconds'], sync_data_ephys['time_downsampled'], "full"))
    print(sync_data_ephys.head())



def process_sync_data(recording_to_process, prm, spatial_data):
    sync_data, is_found = load_sync_data_ephys(recording_to_process, prm)
    sync_data_ephys = pd.DataFrame(sync_data)
    sync_data_ephys.columns = ['sync_pulse']
    sync_data_ephys = get_ephys_sync_on_and_off_times(sync_data_ephys, prm)
    spatial_data = get_video_sync_on_and_off_times(spatial_data)
    correlate_signals(sync_data_ephys, spatial_data)

    check_correlation(sync_data_ephys, spatial_data)
    return sync_data, is_found