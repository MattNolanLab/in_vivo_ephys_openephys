import numpy as np


def wrap_spike_interval(spike_list):
    interval_list=[]
    for rowcount, row in enumerate(spike_list[:-1]):
        interval = spike_list[rowcount+1] - spike_list[rowcount]
        if interval < 0:
            interval = (200-(spike_list[rowcount])) + spike_list[rowcount+1]
        interval_list = np.append(interval,interval_list)
    return interval_list


def get_spike_interval(prm, firing_times_unit):
    spike_list=[]
    for rowcount, row in enumerate(firing_times_unit):
        spike_list = np.append(spike_list, row)

    interval_list = wrap_spike_interval(spike_list)
    return interval_list


def distribution_of_intervals(interval_list):
    inverval_locations=np.zeros((200))

    for loc_count,loc in enumerate(np.arange(1,200,1)):
        intervals = interval_list[np.where(np.logical_and(interval_list <= (loc+1),  interval_list > (loc)))]
        prop = len(intervals)*(1/len(interval_list))
        inverval_locations[loc_count] = prop
    return inverval_locations


def analyse_spike_interval(spike_data, prm):
    # analyse spike interval
    spikes_on_trials = []
    interval_list = get_spike_interval(prm, spikes_on_trials)
    inverval_locations = distribution_of_intervals(interval_list)
    return spike_data
