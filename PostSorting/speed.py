import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import scipy.ndimage
import seaborn as sns
import scipy.stats


def calculate_speed_score(position, spatial_firing, sigma=8, sampling_rate_conversion=30000):
    speed = scipy.ndimage.filters.gaussian_filter(position.speed, sigma)
    speed_scores = []
    speed_score_ps = []
    for index, cell in spatial_firing.iterrows():
        firing_times = cell.firing_times
        firing_hist, edges = np.histogram(firing_times, bins=len(speed), range=(0, max(position.synced_time) * sampling_rate_conversion))
        smooth_hist = scipy.ndimage.filters.gaussian_filter(firing_hist.astype(float), sigma)
        # plt.plot(speed)
        speed_score, p = scipy.stats.pearsonr(speed, smooth_hist)
        speed_scores.append(speed_score)
        speed_score_ps.append(p)
    spatial_firing['speed_score'] = speed_scores
    spatial_firing['speed_score_p_values'] = speed_score_ps

    return spatial_firing


