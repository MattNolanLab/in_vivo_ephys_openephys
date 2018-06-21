from __future__ import division
from fractions import Fraction
import open_ephys_IO
import os
import numpy as np
import math_utility
import pandas as pd
import matplotlib.pylab as plt
import scipy.signal
import resampy



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


def reduce_noise(pulses, threshold):
    to_replace = np.where(pulses < threshold)
    np.put(pulses, to_replace, 0)
    return pulses


def correlate_signals(sync_data_ephys, spatial_data):
    print('I will synchronize the position and ephys data by shifting the position to match the ephys.')
    avg_sampling_rate_bonsai = float(1 / spatial_data['time_seconds'].diff().mean())
    avg_sampling_rate_open_ephys = float(1 / sync_data_ephys['time'].diff().mean())
    sampling_rate_rate = avg_sampling_rate_open_ephys/avg_sampling_rate_bonsai
    # sr_fraction = Fraction(sampling_rate_rate).limit_denominator(max_denominator=1000)
    # sync_data_ephys['time_downsample'] = resampy.resample(sync_data_ephys['time'].values, avg_sampling_rate_open_ephys, avg_sampling_rate_open_ephys)
    length = int(len(sync_data_ephys['time']) / sampling_rate_rate)
    indices = (np.arange(length) * sampling_rate_rate).astype(int)
    sync_data_ephys_downsampled = sync_data_ephys['time'][indices]
    sync_data_ephys_downsampled['sync_pulse'] = sync_data_ephys['sync_pulse'][indices]
    sync_data_ephys_downsampled['time'] = sync_data_ephys['time'][indices]

    bonsai = spatial_data['syncLED'].values
    oe = sync_data_ephys_downsampled.sync_pulse.values

    bonsai = reduce_noise(bonsai, np.median(bonsai) + 4 * np.std(bonsai))
    oe = reduce_noise(oe, 0.5)

    if len(bonsai) < len(oe):
        bonsai = np.pad(bonsai, (0, len(oe)-len(bonsai)), 'constant')
    if len(oe) < len(bonsai):
        oe = np.pad(oe, (0, len(bonsai)-len(oe)), 'constant')

    # corr = scipy.signal.fftconvolve(bonsai, oe[::-1])
    corr = np.correlate(bonsai, oe, "full")
    lag = (np.argmax(corr) - (corr.size + 1)/2)/avg_sampling_rate_bonsai
    spatial_data['synced_time'] = spatial_data.time_seconds - lag

    plt.plot(sync_data_ephys_downsampled['time'].values, sync_data_ephys_downsampled['sync_pulse'].values*2000)
    plt.plot(spatial_data.synced_time, spatial_data['syncLED'])
    print(lag)


def process_sync_data(recording_to_process, prm, spatial_data):
    sync_data, is_found = load_sync_data_ephys(recording_to_process, prm)
    sync_data_ephys = pd.DataFrame(sync_data)
    sync_data_ephys.columns = ['sync_pulse']
    sync_data_ephys = get_ephys_sync_on_and_off_times(sync_data_ephys, prm)
    spatial_data = get_video_sync_on_and_off_times(spatial_data)
    correlate_signals(sync_data_ephys, spatial_data)

    return sync_data, is_found