import pandas as pd
import numpy as np
from scipy import stats




def test_if_reward_zone_ramp(cluster_index,spike_data):
    firing_rate_beaconed = np.array(spike_data.at[cluster_index, 'avg_spike_per_bin_b'])
    firing_rate_nonbeaconed = np.array(spike_data.at[cluster_index, 'avg_spike_per_bin_nb'])
    firing_rate_probe = np.array(spike_data.at[cluster_index, 'avg_spike_per_bin_p'])

    return spike_data


def test_if_track_ramp(cluster_index,spike_data):
    firing_rate_beaconed = np.array(spike_data.at[cluster_index, 'avg_spike_per_bin_b'])
    firing_rate_nonbeaconed = np.array(spike_data.at[cluster_index, 'avg_spike_per_bin_nb'])
    firing_rate_probe = np.array(spike_data.at[cluster_index, 'avg_spike_per_bin_p'])

    bins=np.arange(0,200,1)

    slope,intercept,r_value, p_value, std_err = stats.linregress(bins,firing_rate_beaconed) #linear_regression
    print('-------------------------------------------------')
    print('beaconed firing rate, lingress:' + 'slope:' + str(slope) + 'intercept:' + str(intercept), + 'r_value:' + str(r_value) + 'p_value' + str(p_value))
    slope,intercept,r_value, p_value, std_err = stats.linregress(bins,firing_rate_nonbeaconed) #linear_regression
    print('-------------------------------------------------')
    print('nonbeaconed firing rate, lingress:' + 'slope:' + str(slope) + 'intercept:' + str(intercept), + 'r_value:' + str(r_value) + 'p_value' + str(p_value))
    slope,intercept,r_value, p_value, std_err = stats.linregress(bins,firing_rate_probe) #linear_regression
    print('-------------------------------------------------')
    print('probe firing rate, lingress:' + 'slope:' + str(slope) + 'intercept:' + str(intercept), + 'r_value:' + str(r_value) + 'p_value' + str(p_value))

    return spike_data


def analyse_ramp_firing(spike_data):
    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1

        spike_data = test_if_track_ramp(cluster_index,spike_data)

    return spike_data
