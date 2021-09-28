from astropy.stats import rayleightest
import os
import math
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import subprocess
import sys
import settings
from numpy import inf
from scipy import stats

import PostSorting.open_field_firing_maps


def moving_sum(array, window):
    ret = np.cumsum(array, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window:]


def get_rolling_sum(array_in, window):
    if window > (len(array_in) / 3) - 1:
        print('Window for head-direction histogram is too big, HD plot cannot be made.')
    inner_part_result = moving_sum(array_in, window)
    edges = np.append(array_in[-2 * window:], array_in[: 2 * window])
    edges_result = moving_sum(edges, window)
    end = edges_result[window:math.floor(len(edges_result)/2)]
    beginning = edges_result[math.floor(len(edges_result)/2):-window]
    array_out = np.hstack((beginning, inner_part_result, end))
    return array_out


def get_hd_histogram(angles, window_size=23):
    angles = angles[~np.isnan(angles)]
    theta = np.linspace(0, 2*np.pi, 361)  # x axis
    binned_hd, _, _ = plt.hist(angles, theta)
    smooth_hd = get_rolling_sum(binned_hd, window=window_size)
    smooth_hd = smooth_hd/window_size
    plt.close()
    return smooth_hd


# max firing rate at the angle where the firing rate is highest
def get_max_firing_rate(spatial_firing):
    max_firing_rates = []
    preferred_directions = []
    for index, cluster in spatial_firing.iterrows():
        hd_hist = cluster.hd_spike_histogram
        max_firing_rate = np.max(hd_hist.flatten())
        max_firing_rates.append(max_firing_rate)

        preferred_direction = np.where(hd_hist == max_firing_rate)
        preferred_directions.append(preferred_direction[0])

    spatial_firing['max_firing_rate_hd'] = np.array(max_firing_rates) / 1000  # Hz
    spatial_firing['preferred_HD'] = preferred_directions
    return spatial_firing


def get_hd_score_for_cluster(hd_hist):
    angles = np.linspace(-179, 180, 360)
    angles_rad = angles*np.pi/180
    dy = np.sin(angles_rad)
    dx = np.cos(angles_rad)

    totx = sum(dx * hd_hist)/sum(hd_hist)
    toty = sum(dy * hd_hist)/sum(hd_hist)
    r = np.sqrt(totx*totx + toty*toty)
    return r


'''
This test is used to identify a non-uniform distribution, i.e. it is designed for detecting an unimodal deviation from 
uniformity. More precisely, it assumes the following hypotheses: - H0 (null hypothesis): The population is distributed 
uniformly around the circle. - H1 (alternative hypothesis): The population is not distributed uniformly around the 
circle. Small p-values suggest to reject the null hypothesis.

This is an alternative to using the population mean vector as a head-directions score.

https://docs.astropy.org/en/stable/_modules/astropy/stats/circstats.html#rayleightest
'''


def get_rayleigh_score_for_cluster(hd_hist: np.ndarray) -> float:
    bins_in_histogram = len(hd_hist)
    values = np.radians(np.arange(0, 360, int(360 / bins_in_histogram)))
    rayleigh_p = rayleightest(values, weights=hd_hist)
    return rayleigh_p


def add_rayleigh_score_for_all_clusters(spatial_firing: pd.DataFrame) -> pd.DataFrame:
    print('I will do the Rayleigh test to check if head-direction tuning is uniform.')
    rayleigh_ps = []
    for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
        cluster_df = spatial_firing[(spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster
        hd_hist = cluster_df.hd_spike_histogram.iloc[0].copy()
        p = get_rayleigh_score_for_cluster(hd_hist)
        rayleigh_ps.append(p)
    spatial_firing['rayleigh_score'] = np.array(rayleigh_ps)
    return spatial_firing


def calculate_hd_score(spatial_firing):
    hd_scores = []
    for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
        cluster_df = spatial_firing[(spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster
        hd_hist = cluster_df.hd_spike_histogram.iloc[0].copy()
        r = get_hd_score_for_cluster(hd_hist)
        hd_scores.append(r)
    spatial_firing['hd_score'] = np.array(hd_scores)
    return spatial_firing

def calculate_hd_information_score(spatial_firing, hd_histogram):
    '''
    Calculates the spatial information score in bits per spike as in Skaggs et al.,
    1996, 1993). see calculate_spatial_information(spatial_firing)
    '''
    hd_information_scores = []
    for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
        cluster_df = spatial_firing[(spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster
        mean_firing_rate = cluster_df.iloc[0]["mean_firing_rate"] # λ
        hd_hist = cluster_df.hd_spike_histogram.iloc[0].copy()
        hd_histogram[np.isnan(hd_histogram)] = 0
        occupancy_probability_map = hd_histogram/np.sum(hd_histogram) # Pj
        log_term = np.log2(hd_hist/mean_firing_rate)
        log_term[log_term == -inf] = 0
        Isec = np.sum(occupancy_probability_map*hd_hist*log_term)
        Ispike = Isec/mean_firing_rate
        hd_information_scores.append(Ispike)

    spatial_firing['hd_information_score'] = hd_information_scores
    return spatial_firing

# save hd
def save_hd_for_r(hd_session, hd_cluster, cluster, prm):
    fields_path = prm.get_filepath() + '/Firing_fields/'
    save_path = fields_path + str(int(cluster + 1)) + '_whole_field/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    np.savetxt(save_path + 'session.csv', hd_session, delimiter=',')
    np.savetxt(save_path + 'cluster.csv', hd_cluster, delimiter=',')


def write_shell_script_to_call_r_analysis(prm, cluster):
    firing_field_path = prm.get_filepath() + '/Firing_fields/' + str(int(cluster + 1)) + '_whole_field/'
    python_script_path = os.path.dirname(sys.argv[0])
    script_path = prm.get_filepath() + '/Firing_fields' + '/run_r.sh'
    batch_writer = open(script_path, 'w', newline='\n')
    batch_writer.write('#!/bin/bash\n')
    batch_writer.write('echo "-----------------------------------------------------------------------------------"\n')
    batch_writer.write('echo "This is a shell script that will call R to analyze firing fields."\n')
    batch_writer.write('Rscript ' + python_script_path + '/PostSorting/process_fields.r ' + firing_field_path)
    batch_writer.close()


# calculate statistics for hd in fields
def analyze_hd_r(prm, cluster):
    fields_path = prm.get_filepath() + '/Firing_fields/'
    path = fields_path
    write_shell_script_to_call_r_analysis(prm, cluster)
    os.chmod(path + '/run_r.sh', 484)
    subprocess.call(path + '/run_r.sh', shell=True)


def put_stat_results_in_spatial_df(spatial_firing, prm):
    df_stats = pd.DataFrame([])
    for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
        fields_path = prm.get_filepath() + '/Firing_fields/'
        circular_statistics_path = fields_path + str(int(cluster_id)) + '_whole_field/circular_out.csv'
        if os.path.isfile(circular_statistics_path) is True:
            path_to_hd_stats = circular_statistics_path
            hd_stats_cluster_df = pd.read_csv(path_to_hd_stats)
            df_stats = df_stats.append(hd_stats_cluster_df)
    if 'Watson_two_sample' in df_stats:
        spatial_firing['watson_test_hd'] = df_stats.Watson_two_sample.values
        spatial_firing['kuiper_cluster'] = df_stats.Kuiper_Cluster.values
        spatial_firing['kuiper_session'] = df_stats.Kuiper_Session.values
        spatial_firing['watson_cluster'] = df_stats.Watson_Cluster.values
        spatial_firing['watson_session'] = df_stats.Watson_Session.values
    return spatial_firing

def split_spatial_data_into_quadrants(spatial_data):
    x_midline = np.nanmean(spatial_data["position_x"])
    y_midline = np.nanmean(spatial_data["position_y"])
    quadA = spatial_data[(spatial_data["position_x"] < x_midline) & (spatial_data["position_y"] < y_midline)]
    quadB = spatial_data[(spatial_data["position_x"] > x_midline) & (spatial_data["position_y"] < y_midline)]
    quadC = spatial_data[(spatial_data["position_x"] < x_midline) & (spatial_data["position_y"] > y_midline)]
    quadD = spatial_data[(spatial_data["position_x"] > x_midline) & (spatial_data["position_y"] > y_midline)]
    return quadA, quadB, quadC, quadC, quadD, x_midline, y_midline

def split_spatial_firing_into_quadrants(cluster_spatial_firing, x_midline, y_midline):
    x_positions_spikes = np.asarray(cluster_spatial_firing["x_position"].iloc[0])
    y_positions_spikes = np.asarray(cluster_spatial_firing["y_position"].iloc[0])

    A_mask = (x_positions_spikes < x_midline) & (y_positions_spikes < y_midline)
    B_mask = (x_positions_spikes > x_midline) & (y_positions_spikes < y_midline)
    C_mask = (x_positions_spikes < x_midline) & (y_positions_spikes > y_midline)
    D_mask = (x_positions_spikes > x_midline) & (y_positions_spikes > y_midline)

    quad_A = cluster_spatial_firing.copy()
    quad_B = cluster_spatial_firing.copy()
    quad_C = cluster_spatial_firing.copy()
    quad_D = cluster_spatial_firing.copy()

    quad_A["hd"] = np.asarray(cluster_spatial_firing["hd"].iloc[0])[A_mask]
    quad_B["hd"] = np.asarray(cluster_spatial_firing["hd"].iloc[0])[B_mask]
    quad_C["hd"] = np.asarray(cluster_spatial_firing["hd"].iloc[0])[C_mask]
    quad_D["hd"] = np.asarray(cluster_spatial_firing["hd"].iloc[0])[D_mask]

    return quad_A, quad_B, quad_C, quad_D

def calculate_spatial_stability(spatial_firing, spatial_quadA, spatial_quadB,
                                spatial_quadC, spatial_quadD, x_midline, y_midline):
    hd_stability_scores = []
    for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
        cluster_df = spatial_firing[(spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster
        cluster_quad_A, cluster_quad_B, cluster_quad_C, cluster_quad_D = split_spatial_firing_into_quadrants(cluster_df, x_midline, y_midline)

        hd_histogram_A = get_hd_histogram((np.array(spatial_quadA.hd) + 180) * np.pi / 180)
        hd_histogram_B = get_hd_histogram((np.array(spatial_quadB.hd) + 180) * np.pi / 180)
        hd_histogram_C = get_hd_histogram((np.array(spatial_quadC.hd) + 180) * np.pi / 180)
        hd_histogram_D = get_hd_histogram((np.array(spatial_quadD.hd) + 180) * np.pi / 180)

        hd_spike_histogram_A = get_hd_histogram((np.array(cluster_quad_A.hd.iloc[0]) + 180) * np.pi / 180)
        hd_spike_histogram_B = get_hd_histogram((np.array(cluster_quad_B.hd.iloc[0]) + 180) * np.pi / 180)
        hd_spike_histogram_C = get_hd_histogram((np.array(cluster_quad_C.hd.iloc[0]) + 180) * np.pi / 180)
        hd_spike_histogram_D = get_hd_histogram((np.array(cluster_quad_D.hd.iloc[0]) + 180) * np.pi / 180)

        hd_preference_A = hd_spike_histogram_A/hd_histogram_A
        hd_preference_B = hd_spike_histogram_B/hd_histogram_B
        hd_preference_C = hd_spike_histogram_C/hd_histogram_C
        hd_preference_D = hd_spike_histogram_D/hd_histogram_D

        AB_corr = stats.pearsonr(hd_preference_A, hd_preference_B)
        AC_corr = stats.pearsonr(hd_preference_A, hd_preference_C)
        AD_corr = stats.pearsonr(hd_preference_A, hd_preference_D)
        BC_corr = stats.pearsonr(hd_preference_B, hd_preference_C)
        BD_corr = stats.pearsonr(hd_preference_B, hd_preference_D)
        CD_corr = stats.pearsonr(hd_preference_C, hd_preference_D)

        hd_stability = np.mean([AB_corr, AC_corr, AD_corr, BC_corr, BD_corr, CD_corr])
        hd_stability_scores.append(hd_stability)
    spatial_firing["hd_stability_score"] = hd_stability_scores
    return

def calculate_hd_spike_histogram(spatial_firing, hd_histogram):
    hd_spike_histograms = []
    for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
        cluster_df = spatial_firing[(spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster
        angles_spike = (np.array(cluster_df["hd"].iloc[0]) + 180) * np.pi / 180
        hd_spike_histogram = get_hd_histogram(angles_spike)
        hd_spike_histogram = hd_spike_histogram / hd_histogram
        hd_spike_histograms.append(hd_spike_histogram)
    spatial_firing['hd_spike_histogram'] = hd_spike_histograms
    return spatial_firing

def process_hd_data(spatial_firing, spatial_data):
    print('I will process head-direction data now.')
    angles_whole_session = (np.array(spatial_data.hd) + 180) * np.pi / 180
    hd_histogram = get_hd_histogram(angles_whole_session)
    avg_sampling_rate_bonsai = float(1 / spatial_data['time_seconds'].diff().mean())
    hd_histogram /= avg_sampling_rate_bonsai
    quadA, quadB, quadC, quadC, quadD, x_midline, y_midline = split_spatial_data_into_quadrants(spatial_data)

    spatial_firing = get_max_firing_rate(spatial_firing)
    spatial_firing = calculate_hd_score(spatial_firing)
    spatial_firing = calculate_hd_information_score(spatial_firing, hd_histogram)
    spatial_firing = calculate_spatial_stability(spatial_firing, quadA, quadB, quadC, quadD, x_midline, y_midline)
    spatial_firing = add_rayleigh_score_for_all_clusters(spatial_firing)
    return hd_histogram, spatial_firing


# get HD data for a specific bin of the rate map
def get_indices_for_bin(bin_in_field, spatial_data):
    bin_size_pixels = PostSorting.open_field_firing_maps.get_bin_size()
    bin_x = bin_in_field[0]
    bin_x_left_pixels = bin_x * bin_size_pixels
    bin_x_right_pixels = (bin_x+1) * bin_size_pixels
    bin_y = bin_in_field[1]
    bin_y_bottom_pixels = bin_y * bin_size_pixels
    bin_y_top_pixels = (bin_y+1) * bin_size_pixels

    left_x_border = spatial_data.x > bin_x_left_pixels
    right_x_border = spatial_data.x < bin_x_right_pixels
    bottom_y_border = spatial_data.y > bin_y_bottom_pixels
    top_y_border = spatial_data.y < bin_y_top_pixels

    inside_bin = spatial_data[left_x_border & right_x_border & bottom_y_border & top_y_border]
    return inside_bin


# get head-direction data from bins of field
def get_hd_in_field_spikes(rate_map_indices, spatial_data):
    hd_in_field = []
    event_times_in_field = []
    for bin_in_field in rate_map_indices:
        inside_bin = get_indices_for_bin(bin_in_field, spatial_data)
        hd = inside_bin.hd.values
        hd_in_field.extend(hd)
        event_times = inside_bin.firing_times.values
        event_times_in_field.extend(event_times)
    return hd_in_field, event_times_in_field


# get head-direction data from bins of field
def get_hd_in_field(rate_map_indices, spatial_data):
    hd_in_field = []
    event_times_in_field = []
    for bin_in_field in rate_map_indices:
        inside_bin = get_indices_for_bin(bin_in_field, spatial_data)
        hd = inside_bin.hd.values
        hd_in_field.extend(hd)
        event_times = inside_bin.synced_time.values
        event_times_in_field.extend(event_times)
    return hd_in_field, event_times_in_field


# return array of HD in subfield when cell fired for cluster
def get_hd_in_firing_rate_bins_for_cluster(spatial_firing, rate_map_indices, cluster_id):
    cluster_df = spatial_firing[(spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster
    spatial_firing_cluster = pd.DataFrame(np.arange(len(cluster_df['firing_times'].iloc[0])))

    if type(cluster_df['position_x_pixels'].iloc[0]) is np.ndarray:
        spatial_firing_cluster['x'] = cluster_df['position_x_pixels'].iloc[0]
        spatial_firing_cluster['y'] = cluster_df['position_y_pixels'].iloc[0]
        spatial_firing_cluster['hd'] = cluster_df['hd'].iloc[0]
    elif type(cluster_df['position_x_pixels'].iloc[0]) is list:
        spatial_firing_cluster['x'] = cluster_df['position_x_pixels'].iloc[0]
        spatial_firing_cluster['y'] = cluster_df['position_y_pixels'].iloc[0]
        spatial_firing_cluster['hd'] = cluster_df['hd'].iloc[0]
    else:
        spatial_firing_cluster['x'] = cluster_df['position_x_pixels'].iloc[0].values
        spatial_firing_cluster['y'] = cluster_df['position_y_pixels'].iloc[0].values
        spatial_firing_cluster['hd'] = cluster_df['hd'].iloc[0].values

    spatial_firing_cluster['firing_times'] = cluster_df['firing_times'].iloc[0]
    hd_in_field, spike_times = get_hd_in_field_spikes(rate_map_indices, spatial_firing_cluster)
    hd_in_field = (np.array(hd_in_field) + 180) * np.pi / 180
    return hd_in_field, spike_times


# return array of HD angles in subfield when from the whole session
def get_hd_in_firing_rate_bins_for_session(spatial_data, rate_map_indices):
    spatial_data_field = pd.DataFrame()
    spatial_data_field['x'] = spatial_data.position_x_pixels
    spatial_data_field['y'] = spatial_data.position_y_pixels
    spatial_data_field['hd'] = spatial_data.hd
    spatial_data_field['synced_time'] = spatial_data.synced_time
    hd_in_field, times = get_hd_in_field(rate_map_indices, spatial_data_field)
    hd_in_field = (np.array(hd_in_field) + 180) * np.pi / 180
    return hd_in_field, times


def main():
    pass


if __name__ == '__main__':
    main()
