import pandas as pd


def extract_instant_rates(spike_data, cluster_index):
    speed = np.array(spike_data.loc[cluster_index].instant_rates[0])
    location = np.array(spike_data.loc[cluster_index].instant_rates[1])
    firing_rate = np.array(spike_data.loc[cluster_index].instant_rates[2])

    #cluster_firings = pd.DataFrame({ 'speed' :  spike_data.loc[cluster_index].instant_rates[0], 'location' :  spike_data.loc[cluster_index].instant_rates[1], 'firing_rate' :  spike_data.loc[cluster_index].instant_rates[2]})
    return speed, location, firing_rate

