# test for inhibition by comparing number of spikes before and after optical stimulus using Mann-Whitney U
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def get_number_of_spikes_around_light(peristimulus_spike_data, window_ms=20, sampling_rate=30000):
    # get number of spikes before and after light pulse from each light trial
    window_ms = int(sampling_rate/1000) * window_ms  # default is 20 ms
    spikes_around_light = pd.DataFrame()
    spikes_before_light = []
    spikes_after_light = []
    cluster_ids = peristimulus_spike_data.cluster_id.unique()

    for cluster_id in cluster_ids:
        spikes = peristimulus_spike_data[peristimulus_spike_data.cluster_id == cluster_id].iloc[:, 2:].values
        middle = int(spikes.shape[1] / 2)
        before_light = spikes[:, middle - window_ms:middle]
        spikes_before_light.append(np.sum(before_light, axis=1))
        after_light = spikes[:, middle:middle + window_ms]
        spikes_after_light.append(np.sum(after_light, axis=1))

    spikes_around_light['cluster_id'] = cluster_ids
    spikes_around_light['spikes_before_light'] = spikes_before_light
    spikes_around_light['spikes_after_light'] = spikes_after_light

    return spikes_around_light


def analyse_inhibition_of_cells(spikes_around_light, spatial_firing):
    # adds U-value and p-value to spatial firing dataframe for neurons with reduced spiking after stimulus
    u_vals = []
    p_vals = []

    for index, cell in spikes_around_light.iterrows():
        reduced_activity = cell.spikes_before_light.sum() > cell.spikes_after_light.sum()
        if reduced_activity:  # only run analysis on cells with lower spiking after stimulus
            u, p = mannwhitneyu(cell.spikes_before_light, cell.spikes_after_light)
            u_vals.append(u)
            p_vals.append(p)
        else:
            u_vals.append(np.nan)  # add NaN for non-inhibited cells
            p_vals.append(np.nan)

    spatial_firing["inhibition_MW_U"] = u_vals
    spatial_firing["inhibition_MW_p"] = p_vals

    return spatial_firing


def run_test_for_opto_inhibition(spatial_firing, peristimulus_data):
    spikes_around_light = get_number_of_spikes_around_light(peristimulus_data)
    spatial_firing = analyse_inhibition_of_cells(spikes_around_light, spatial_firing)
    return spatial_firing


# main function for testing
def main():
    import PostSorting.parameters
    prm = PostSorting.parameters.Parameters()
    prm.set_sampling_rate(30000)
    path = ('/Users/briannavandrey/Desktop/1474_08_31/')
    peristimulus_spikes = pd.read_pickle('/Users/briannavandrey/Desktop/1474_08_31/peristimulus_spikes.pkl')
    spatial_firing = pd.read_pickle('/Users/briannavandrey/Desktop/1474_08_31/spatial_firing.pkl')
    spatial_firing = run_test_for_opto_inhibition(spatial_firing, peristimulus_spikes)
    spatial_firing.to_pickle(path + 'spatial_firing_with_inhibition.pkl')


if __name__ == '__main__':
    main()