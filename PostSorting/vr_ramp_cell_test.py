import pandas as pd
import numpy as np
from scipy import stats
import PostSorting.vr_make_plots


def calculate_regression_line(x,y):
    slope,intercept,r_value, p_value, std_err = stats.linregress(x,y) #linear_regression
    return slope, intercept,r_value, p_value, std_err


def test_if_reward_zone_ramp(prm,cluster_index,spike_data):
    firing_rate_beaconed = np.array(spike_data.at[cluster_index, 'avg_spike_per_bin_b'])
    firing_rate_nonbeaconed = np.array(spike_data.at[cluster_index, 'avg_spike_per_bin_nb'])
    firing_rate_probe = np.array(spike_data.at[cluster_index, 'avg_spike_per_bin_p'])
    bins=np.arange(0,100,1)

    beaconed_start = firing_rate_beaconed[:100]
    beaconed_end = firing_rate_beaconed[100:]
    nonbeaconed_start = firing_rate_nonbeaconed[:100]
    nonbeaconed_end = firing_rate_nonbeaconed[100:]
    probe_start = firing_rate_probe[:100]
    probe_end = firing_rate_probe[100:]

    slope,intercept,r_value, p_value, std_err = calculate_regression_line(bins[30:90],beaconed_start[30:90])
    print('--------------------------------------------------------------------------------------------------')
    print('Analysing ', str(cluster_index), ' now....... : region specific analysis')
    print('beaconed firing rate in outbound, lingress: slope:', str(slope), ', intercept:', str(intercept), ', r_value:', str(r_value), ', p_value', str(p_value))
    PostSorting.vr_make_plots.plot_firing_rate_vs_distance_regression(prm, cluster_index,spike_data, bins[30:90],beaconed_start[30:90], slope,intercept,r_value, p_value, prefix='RZ_beaconed')

    slope,intercept,r_value, p_value, std_err = calculate_regression_line(bins[30:90],nonbeaconed_start[30:90])
    print('--------------------------------------------------------------------------------------------------')
    print('nonbeaconed firing rate in outbound, lingress: slope:', str(slope), ', intercept:', str(intercept), ', r_value:', str(r_value), ', p_value', str(p_value))
    PostSorting.vr_make_plots.plot_firing_rate_vs_distance_regression(prm, cluster_index,spike_data, bins[30:90],nonbeaconed_start[30:90], slope,intercept, r_value, p_value,prefix='RZ_nonbeaconed')

    slope,intercept,r_value, p_value, std_err = calculate_regression_line(bins[30:90],probe_start[30:90])
    print('--------------------------------------------------------------------------------------------------')
    print('probe firing rate in outbound, lingress: slope:', str(slope), ', intercept:', str(intercept), ', r_value:', str(r_value), ', p_value', str(p_value))
    PostSorting.vr_make_plots.plot_firing_rate_vs_distance_regression(prm, cluster_index,spike_data, bins[30:90],probe_start[30:90], slope,intercept, r_value, p_value,prefix='RZ_probe')

    slope,intercept,r_value, p_value, std_err = calculate_regression_line(bins[10:70],beaconed_end[10:70])
    print('--------------------------------------------------------------------------------------------------')
    print('beaconed firing rate in homebound, lingress:' + 'slope:', str(slope), ', intercept:', str(intercept), ', r_value:', str(r_value), ', p_value', str(p_value))
    PostSorting.vr_make_plots.plot_firing_rate_vs_distance_regression(prm, cluster_index,spike_data, bins[10:70],beaconed_end[10:70], slope,intercept,r_value, p_value, prefix='RZ_beaconed')

    slope,intercept,r_value, p_value, std_err = calculate_regression_line(bins[10:70],nonbeaconed_end[10:70])
    print('--------------------------------------------------------------------------------------------------')
    print('nonbeaconed firing rate in homebound, lingress:' + 'slope:', str(slope), ', intercept:', str(intercept), ', r_value:', str(r_value), ', p_value', str(p_value))
    PostSorting.vr_make_plots.plot_firing_rate_vs_distance_regression(prm, cluster_index,spike_data, bins[10:70],nonbeaconed_end[10:70], slope,intercept, r_value, p_value,prefix='RZ_nonbeaconed')

    slope,intercept,r_value, p_value, std_err = calculate_regression_line(bins[10:70],probe_end[10:70])
    print('--------------------------------------------------------------------------------------------------')
    print('probe firing rate in homebound, lingress: ' + 'slope:', str(slope), ', intercept:', str(intercept), ', r_value:', str(r_value), ', p_value', str(p_value))
    PostSorting.vr_make_plots.plot_firing_rate_vs_distance_regression(prm, cluster_index,spike_data, bins[10:70],probe_end[10:70], slope,intercept, r_value, p_value,prefix='RZ_probe')
    return spike_data


def test_if_track_ramp(prm, cluster_index,spike_data):
    firing_rate_beaconed = np.array(spike_data.at[cluster_index, 'avg_spike_per_bin_b'])
    firing_rate_nonbeaconed = np.array(spike_data.at[cluster_index, 'avg_spike_per_bin_nb'])
    firing_rate_probe = np.array(spike_data.at[cluster_index, 'avg_spike_per_bin_p'])

    bins=np.arange(0,200,1)

    slope,intercept,r_value, p_value, std_err = calculate_regression_line(bins[30:170],firing_rate_beaconed[30:170])
    print('--------------------------------------------------------------------------------------------------')
    print('Analysing ', str(cluster_index), ' now....... : whole track analysis')
    print('--------------------------------------------------------------------------------------------------')
    print('beaconed firing rate, lingress: slope:', str(slope), ', intercept:', str(intercept), ', r_value:', str(r_value), ', p_value', str(p_value))
    PostSorting.vr_make_plots.plot_firing_rate_vs_distance_regression(prm, cluster_index,spike_data, bins[30:170],firing_rate_beaconed[30:170], slope,intercept, r_value, p_value,prefix='Track_beaconed')

    slope,intercept,r_value, p_value, std_err = calculate_regression_line(bins[30:170],firing_rate_nonbeaconed[30:170])
    print('--------------------------------------------------------------------------------------------------')
    print('nonbeaconed firing rate, lingress: slope:', str(slope), ', intercept:', str(intercept), ', r_value:', str(r_value), ', p_value', str(p_value))
    PostSorting.vr_make_plots.plot_firing_rate_vs_distance_regression(prm, cluster_index,spike_data, bins[30:170],firing_rate_nonbeaconed[30:170], slope,intercept, r_value, p_value,prefix='Track_nonbeaconed')

    slope,intercept,r_value, p_value, std_err = calculate_regression_line(bins[30:170],firing_rate_probe[30:170])
    print('--------------------------------------------------------------------------------------------------')
    print('probe firing rate, lingress: slope:', str(slope), ', intercept:', str(intercept), ', r_value:', str(r_value), ', p_value', str(p_value))
    PostSorting.vr_make_plots.plot_firing_rate_vs_distance_regression(prm, cluster_index,spike_data, bins[30:170],firing_rate_probe[30:170], slope,intercept, r_value, p_value,prefix='Track_probe')
    return spike_data


def make_location_continuous(firing_locations):
    trial=0
    continuous_firing_locations=[]
    for rowcount, row in enumerate(firing_locations[:-1]):
        diff = firing_locations[rowcount+1] - firing_locations[rowcount]
        if diff <0:
            trial+=1
        location_of_firing = firing_locations[rowcount] + (trial*200)
        continuous_firing_locations = np.append(continuous_firing_locations,location_of_firing)
    return continuous_firing_locations


def find_intervals(continuous_firing_locations):
    intervals = []
    for rowcount, row in enumerate(continuous_firing_locations[1:]):
        diff = continuous_firing_locations[rowcount] - continuous_firing_locations[rowcount-1]
        intervals = np.append(intervals,diff)
    return intervals


def calculate_interspike_distance_interval(cluster_index,spike_data):
    firing_locations = np.array(spike_data.at[cluster_index, 'x_position_cm'])
    continuous_firing_locations = make_location_continuous(firing_locations)
    location_intervals = find_intervals(continuous_firing_locations)
    spike_data.at[cluster_index, 'location_intervals'] = list(location_intervals)
    return spike_data


def calculate_instantaneous_firing_rate(cluster_index,spike_data):
    firing_locations = np.array(spike_data.at[cluster_index, 'firing_times'])
    firing_intervals = find_intervals(firing_locations)
    spike_data.at[cluster_index, 'firing_intervals'] = list(firing_intervals)
    return spike_data


def analyse_ramp_firing(prm, spike_data):
    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        spike_data = test_if_track_ramp(prm,cluster_index,spike_data)
        spike_data = test_if_reward_zone_ramp(prm,cluster_index,spike_data)

    #for cluster_index in range(len(spike_data)):
    #    cluster_index = spike_data.cluster_id.values[cluster_index] - 1
    #    spike_data = calculate_interspike_distance_interval(cluster_index,spike_data)
    #    spike_data = calculate_instantaneous_firing_rate(cluster_index,spike_data)

    return spike_data
