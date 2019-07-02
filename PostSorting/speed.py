import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import scipy.ndimage


def calculate_speed_score(position, spatial_firing, sigma=8):
    speed = position.speed
    for index, cell in spatial_firing.iterrows():
        firing_times = cell.firing_times
        firing_hist, edges = np.histogram(firing_times, bins=len(speed))
        # todo this should be firing rates not just histograms
        smooth_hist = scipy.ndimage.filters.gaussian_filter(firing_hist, sigma)
        plt.plot(speed)
        plt.plot(np.histogram(smooth_hist, bins=len(speed))[0])

        pass



    return spatial_firing


