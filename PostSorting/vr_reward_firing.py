import numpy as np
import pandas as pd



def add_columns_to_dataframe(spike_data):
    spike_data["rewarded_beaconed_position_cm"] = ""
    spike_data["rewarded_nonbeaconed_position_cm"] = ""
    spike_data["rewarded_probe_position_cm"] = ""
    spike_data["rewarded_beaconed_trial_numbers"] = ""
    spike_data["rewarded_nonbeaconed_trial_numbers"] = ""
    spike_data["rewarded_probe_trial_numbers"] = ""
    spike_data["nonrewarded_beaconed_position_cm"] = ""
    spike_data["nonrewarded_nonbeaconed_position_cm"] = ""
    spike_data["nonrewarded_probe_position_cm"] = ""
    spike_data["nonrewarded_beaconed_trial_numbers"] = ""
    spike_data["nonrewarded_nonbeaconed_trial_numbers"] = ""
    spike_data["nonrewarded_probe_trial_numbers"] = ""
    return spike_data



def split_trials_by_reward(processed_position_data,spike_data):

    spike_data = add_columns_to_dataframe(spike_data)

    #rewarded_trials = np.array(processed_position_data['rewarded_trials'].dropna(axis=0), dtype=np.int16)
    rewarded_trials = np.array(np.arange(25,100,1))

    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1

        beaconed_position_cm = np.array(spike_data.beaconed_position_cm[cluster_index])
        beaconed_trial_number = np.array(spike_data.beaconed_trial_number[cluster_index])
        nonbeaconed_position_cm = np.array(spike_data.nonbeaconed_position_cm[cluster_index])
        nonbeaconed_trial_number = np.array(spike_data.nonbeaconed_trial_number[cluster_index])
        probe_position_cm = np.array(spike_data.probe_position_cm[cluster_index])
        probe_trial_number = np.array(spike_data.probe_trial_number[cluster_index])

        #take firing locations when on rewarded trials
        rewarded_beaconed_position_cm = beaconed_position_cm[np.isin(beaconed_trial_number,rewarded_trials)]
        rewarded_nonbeaconed_position_cm = nonbeaconed_position_cm[np.isin(nonbeaconed_trial_number,rewarded_trials)]
        rewarded_probe_position_cm = probe_position_cm[np.isin(probe_trial_number,rewarded_trials)]

        #take firing trial numbers when on rewarded trials
        rewarded_beaconed_trial_numbers = beaconed_trial_number[np.isin(beaconed_trial_number,rewarded_trials)]
        rewarded_nonbeaconed_trial_numbers = nonbeaconed_trial_number[np.isin(nonbeaconed_trial_number,rewarded_trials)]
        rewarded_probe_trial_numbers = probe_trial_number[np.isin(probe_trial_number,rewarded_trials)]

        spike_data.at[cluster_index, 'rewarded_beaconed_position_cm'] = list(rewarded_beaconed_position_cm)
        spike_data.at[cluster_index, 'rewarded_nonbeaconed_position_cm'] = list(rewarded_nonbeaconed_position_cm)
        spike_data.at[cluster_index, 'rewarded_probe_position_cm'] = list(rewarded_probe_position_cm)
        spike_data.at[cluster_index, 'rewarded_beaconed_trial_numbers'] = list(rewarded_beaconed_trial_numbers)
        spike_data.at[cluster_index, 'rewarded_nonbeaconed_trial_numbers'] = list(rewarded_nonbeaconed_trial_numbers)
        spike_data.at[cluster_index, 'rewarded_probe_trial_numbers'] = list(rewarded_probe_trial_numbers)

    return spike_data

