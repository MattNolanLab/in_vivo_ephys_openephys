#using Klara's core functions: OverallAnalysis/count_modes_in_fields.py; PostSorting/open_field_grid_cells.py; PostSorting/open_field_head_direction.py

import numpy as np
from matplotlib import pyplot as plt
import random
import math 
from neuron import h, gui
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import h5py
from os import listdir 
from os.path import isfile
import csv
import glob
import os
import pandas as pd
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
import cmocean
import pickle
from skimage import measure
from skimage.transform import rotate
import glob
import matplotlib.pylab as plt
#import math_utility
import math
import numpy as np
import os
import pandas as pd
#import plot_utility
from rpy2 import robjects as robj
from scipy.stats import circstd
from rpy2.robjects import pandas2ri
from scipy.stats import pearsonr
import scikit_posthocs._posthocs as sp
from scipy.stats import pearsonr
import seaborn as sns
from scipy import stats
import rpy2 as rpy

f = open('single_positions', 'rb')
positions=pickle.load(f)
f.close()

f = open('single_hd', 'rb')
hd=pickle.load(f)
f.close()

f = open('single_t_array', 'rb')
t_array=pickle.load(f)
f.close()

#experimental analysis results:
f = open('mouse_exp.pkl', 'rb')
mouse=pickle.load(f)
f.close()

f = open('rat_exp.pkl', 'rb')
rat=pickle.load(f)
f.close()


##beginning of core functions
def get_shifted_map(firing_rate_map, x, y):
    #add null arrays before x and before yx_shift=np.asarray([x]*firing_rate_map.shape[0])
    xn=abs(x)
    yn=abs(y)
    col=firing_rate_map.shape[1]
    row=firing_rate_map.shape[0]
    
    if x>=0:
        firing_rate_map_shift=np.delete(firing_rate_map, np.arange(0,xn), axis=1) #deleting columns hanging off left edge
        shifted_map=np.delete(firing_rate_map, np.arange((col-xn),col), axis=1) #deleting columns hanging off right edge
    
    else: #x<0
        firing_rate_map_shift=np.delete(firing_rate_map, np.arange((col-xn),col), axis=1) #deleting columns hanging off left edge
        shifted_map=np.delete(firing_rate_map, np.arange(0,xn), axis=1)
    
    if y>=0:
        firing_rate_map_shift=np.delete(firing_rate_map_shift, np.arange(0,yn), axis=0) #deleting columns hanging off left edge
        shifted_map=np.delete(shifted_map, np.arange((row-yn),row), axis=0) #deleting columns hanging off right edge
    
    else: #y<0
        firing_rate_map_shift=np.delete(firing_rate_map_shift, np.arange((row-yn),row), axis=0) #deleting columns hanging off left edge
        shifted_map=np.delete(shifted_map, np.arange(0,yn), axis=0) #deleting columns hanging off right edge
    
    return shifted_map, firing_rate_map_shift


def remove_zeros(array1, array2):
    lose1=np.argwhere(np.isnan(array1))
    lose2=np.argwhere(np.isnan(array2))
    lose=np.append(lose1, lose2, axis=0)
    lose_0=np.unique(lose[:,0]) #lose these axis 0's
    lose_1=np.unique(lose[:,1]) #lose these axis 1's
    array1=np.delete(array1, lose_0, axis=0)
    array1=np.delete(array1, lose_1, axis=1)
    array2=np.delete(array2, lose_0, axis=0)
    array2=np.delete(array2, lose_1, axis=1)
    
    return array1.flatten(), array2.flatten()

'''
The array is shifted along the x and y axes into every possible position where it overlaps with itself starting from
the position where the shifted array's bottom right element overlaps with the top left of the map. Correlation is
calculated for all positions and returned as a correlation_vector. The correlation vector is 2x * 2y.
'''


def get_rate_map_autocorrelogram(firing_rate_map):
    length_y = firing_rate_map.shape[0] - 1
    length_x = firing_rate_map.shape[1] - 1
    correlation_vector = np.empty((length_x * 2 + 1, length_y * 2 + 1)) * 0
    for shift_x in range(-length_x, length_x):
        for shift_y in range(-length_y, length_y):
            # shift map by x and y and remove extra bits
            shifted_map, firing_rate_map_shift = get_shifted_map(firing_rate_map, shift_x, -shift_y)
            firing_rate_map_to_correlate, shifted_map = remove_zeros(firing_rate_map_shift, shifted_map)

            correlation_y = shift_y + length_y
            correlation_x = shift_x + length_x

            if len(shifted_map) > 20:
                # np.corrcoef(x,y)[0][1] gives the same result for 1d vectors as matlab's corr(x,y) (Pearson)
                # https://stackoverflow.com/questions/16698811/what-is-the-difference-between-matlab-octave-corr-and-python-numpy-correlate
                correlation_vector[correlation_x, correlation_y] = np.corrcoef(firing_rate_map_to_correlate, shifted_map)[0][1]
            else:
                correlation_vector[correlation_x, correlation_y] = np.nan
    return correlation_vector


# make autocorr map binary based on threshold
def threshold_autocorrelation_map(autocorrelation_map):
    autocorrelation_map[autocorrelation_map > 0.2] = 1
    autocorrelation_map[autocorrelation_map <= 0.2] = 0
    return autocorrelation_map


# find peaks of autocorrelogram
def find_autocorrelogram_peaks(autocorrelation_map):
    autocorrelation_map_thresholded = threshold_autocorrelation_map(autocorrelation_map)
    autocorr_map_labels = measure.label(autocorrelation_map_thresholded)  # each field is labelled with a single digit
    field_properties = measure.regionprops(autocorr_map_labels)
    return field_properties


# calculate distances between the middle of the rate map autocorrelogram and the field centres
def find_field_distances_from_mid_point(autocorr_map, field_properties):
    distances = []
    mid_point_coord_x = np.ceil(autocorr_map.shape[0] / 2)
    mid_point_coord_y = np.ceil(autocorr_map.shape[1] / 2)

    for field in range(len(field_properties)):
        field_central_x = field_properties[field].centroid[0]
        field_central_y = field_properties[field].centroid[1]
        distance = np.sqrt(np.square(field_central_x - mid_point_coord_x) + np.square(field_central_y - mid_point_coord_y))
        distances.append(distance)
    return distances


'''
Grid spacing/wavelength:
Defined by Hafting, Fyhn, Molden, Moser, Moser (2005) as the distance from the central autocorrelogram peak to the
vertices of the inner hexagon in the autocorrelogram (the median of the six distances). This should be in cm.
'''


def calculate_grid_spacing(field_distances, bin_size):
    grid_spacing = np.median(field_distances) * bin_size
    return grid_spacing


'''
Defined by Wills, Barry, Cacucci (2012) as the square root of the area of the central peak of the autocorrelogram
divided by pi. (This should be in cm2.)
'''


def calculate_field_size(field_properties, field_distances, bin_size):
    central_field_index = np.argmin(field_distances)
    field_size_pixels = field_properties[central_field_index].area  # number of pixels in central field
    field_size = np.sqrt(field_size_pixels * np.squeeze(bin_size)) / np.pi
    return field_size


# https://stackoverflow.com/questions/481144/equation-for-testing-if-a-point-is-inside-a-circle
def in_circle(center_x, center_y, radius, x, y):
    square_dist = (center_x - x) ** 2 + (center_y - y) ** 2
    return square_dist <= radius ** 2



#  replace values not in the grid ring with nan
def remove_inside_and_outside_of_grid_ring(autocorr_map, field_properties, field_distances):
    ring_distances = get_ring_distances(field_distances)
    inner_radius = (np.mean(ring_distances) * 0.5) / 2
    outer_radius = (np.mean(ring_distances) * 2.5) / 2
    center_x = field_properties[np.argmin(field_distances)].centroid[0]
    center_y = field_properties[np.argmin(field_distances)].centroid[1]
    for row in range(autocorr_map.shape[0]):
        for column in range(autocorr_map.shape[1]):
            in_ring = in_circle(center_x, center_y, outer_radius, row, column)
            in_middle = in_circle(center_x, center_y, inner_radius, row, column)
            if not in_ring or in_middle:
                autocorr_map[row, column] = np.nan
    return autocorr_map


'''
Defined by Krupic, Bauza, Burton, Barry, O'Keefe (2015) as the difference between the minimum correlation coefficient
for autocorrelogram rotations of 60 and 120 degrees and the maximum correlation coefficient for autocorrelogram
rotations of 30, 90 and 150 degrees. This score can vary between -2 and 2, although generally values
below -1.5 or above 1.5 are uncommon.
'''


def calculate_grid_score(autocorr_map, field_properties, field_distances):
    correlation_coefficients = []
    for angle in range(30, 180, 30):
        autocorr_map_to_rotate = np.nan_to_num(autocorr_map)
        rotated_map = rotate(autocorr_map_to_rotate, angle, resize=False)
        rotated_map_binary = np.round(rotated_map)
        autocorr_map_ring = remove_inside_and_outside_of_grid_ring(autocorr_map, field_properties, field_distances)
        rotated_map_ring = remove_inside_and_outside_of_grid_ring(rotated_map_binary, field_properties, field_distances)
        autocorr_map_ring_to_correlate, rotated_map_ring_to_correlate = remove_nans(autocorr_map_ring, rotated_map_ring)
        pearson_coeff = np.corrcoef(autocorr_map_ring_to_correlate, rotated_map_ring_to_correlate)[0][1]
        correlation_coefficients.append(pearson_coeff)
    grid_score = min(correlation_coefficients[i] for i in [1, 3]) - max(correlation_coefficients[i] for i in [0, 2, 4])
    return grid_score


def get_ring_distances(field_distances_from_mid_point):
    field_distances_from_mid_point = np.array(field_distances_from_mid_point)[~np.isnan(field_distances_from_mid_point)]
    ring_distances = np.sort(field_distances_from_mid_point)[1:7]
    return ring_distances


def calculate_grid_metrics(autocorr_map, field_properties):
    bin_size = 2.5  # cm
    field_distances_from_mid_point = find_field_distances_from_mid_point(autocorr_map, field_properties)
    # the field with the shortest distance is the middle and the next 6 closest are the middle 6
    ring_distances = get_ring_distances(field_distances_from_mid_point)
    grid_spacing = calculate_grid_spacing(ring_distances, bin_size)
    field_size = calculate_field_size(field_properties, field_distances_from_mid_point, bin_size)
    grid_score = calculate_grid_score(autocorr_map, field_properties, field_distances_from_mid_point)
    return grid_spacing, field_size, grid_score

def remove_nans(array1, array2):
    array2 = array2.flatten()
    array2[np.isnan(array2)] = 666
    array1 = array1.flatten()
    array1[np.isnan(array1)] = 666
    array2_tmp = np.take(array2, np.where(array1 != 666))
    array1_tmp = np.take(array1, np.where(array2 != 666))
    array2 = np.take(array2_tmp, np.where(array2_tmp[0] != 666))
    array1 = np.take(array1_tmp, np.where(array1_tmp[0] != 666))
    return array1.flatten(), array2.flatten()


def process_grid_data(firing_rate_map, spatial_firing):
    rate_map_correlograms = []
    grid_spacings = []
    field_sizes = []
    grid_scores = []
    rate_map_correlogram = get_rate_map_autocorrelogram(firing_rate_map)
    rate_map_correlograms.append(np.copy(rate_map_correlogram))
    field_properties = find_autocorrelogram_peaks(rate_map_correlogram)
    if len(field_properties) > 7:
        grid_spacing, field_size, grid_score = calculate_grid_metrics(rate_map_correlogram, field_properties)
        grid_spacings.append(grid_spacing)
        field_sizes.append(field_size)
        grid_scores.append(grid_score)
    else:
        print('Not enough fields to calculate grid metrics.')
        grid_spacings.append(np.nan)
        field_sizes.append(np.nan)
        grid_scores.append(np.nan)
    spatial_firing['rate_map_autocorrelogram'] = rate_map_correlograms
    spatial_firing['grid_spacing'] = grid_spacings
    spatial_firing['field_size'] = field_sizes
    spatial_firing['grid_score'] = grid_scores
    return spatial_firing

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

def get_hd_histogram(hd_fr):
    smooth_hd = get_rolling_sum(hd_fr, window=23)
    return smooth_hd

def get_hd_hist(hd_firing, hd):
    shape=(360)
    hd_fr=np.zeros(shape)
    time_per_bin2=np.zeros(shape)
    for i in np.arange(0,360):
        spikes_per_bin=len(hd_firing[(hd_firing<i+1) & (hd_firing>i)]) #spikes per 1 degree bin
        time_per_bin2[i]= len(hd[(hd<i+1) & (hd>i)])/1000 #ms
        time_per_bin= len(hd[(hd<i+1) & (hd>i)])/1000 #ms
        if time_per_bin==0:
            hd_fr[i]=0
        else: 
            hd_fr[i]=spikes_per_bin/time_per_bin

    max_firing_rate_hd = np.max(hd_fr)
    smooth_hd=get_hd_histogram(hd_fr)
    smooth_hd=(smooth_hd/np.amax(smooth_hd))*max_firing_rate_hd
    preferred_direction =np.where(smooth_hd==max(smooth_hd))
    r=get_hd_score_for_cluster(smooth_hd)
    return smooth_hd, preferred_direction, max_firing_rate_hd, time_per_bin2, r

def get_hd_score_for_cluster(hd_hist):
    angles = np.linspace(0, 180, 360)
    angles_rad = angles*np.pi/180
    dy = np.sin(angles_rad)
    dx = np.cos(angles_rad)

    totx = sum(dx * hd_hist)/sum(hd_hist)
    toty = sum(dy * hd_hist)/sum(hd_hist)
    r = np.sqrt(totx*totx + toty*toty)
    return r

def get_hd_in_firing_rate_bins_for_cluster(spatial_firing, rate_map_indices, cluster): #for specific
    cluster_id = 0
    #hd_in_field, spike_times = get_hd_in_field_spikes(rate_map_indices, spatial_firing_cluster)
    hd_in_bin=[]
    hd_at_fire_in_bin=[]
    #define area to count spikes and times in, maybe switch away from square?
    for i in np.arange(rate_map_indices.shape[0]):
        j=rate_map_indices[i,1]
        k=rate_map_indices[i,0]
        y_min=j*bin_size_cm
        y_max=(j+1)*bin_size_cm
        x_min=k*bin_size_cm 
        x_max=(k+1)*bin_size_cm

        #determine head-direction data within this bin
        hd_in_bin.extend(hd[(positions[:,0]>=x_min) & (positions[:,0]<x_max) & (positions[:,1]>=y_min) & (positions[:,1]<y_max),0]) #head_directions recorded when the animal was in bin
        hd_at_fire_in_bin.extend(hd_firing[(target_firing[:,0]>=x_min)&(target_firing[:,0]<x_max) & (target_firing[:,1]>=y_min) & (target_firing[:,1]<y_max)]) #what hd are we firing at at this time
     
    hd_in_bin=np.asarray(hd_in_bin)
    hd_at_fire_in_bin=np.asarray(hd_at_fire_in_bin)
    smooth_hd, preferred_direction, max_firing_rate_hd, time_per_bin, r=get_hd_hist(hd_at_fire_in_bin, hd_in_bin)
        
    return smooth_hd, preferred_direction, max_firing_rate_hd, r

def style_open_field_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        right=False,
        left=False,
        labelleft=False,
        labelbottom=False)  # labels along the bottom edge are off

    ax.set_aspect('equal')
    return ax

def style_polar_plot(ax):
    ax.spines['polar'].set_visible(False)
    ax.set_yticklabels([])  # remove yticklabels
    # ax.grid(None)
    plt.xticks([math.radians(0), math.radians(90), math.radians(180), math.radians(270)])
    ax.axvline(math.radians(90), color='black', linewidth=1, alpha=0.6)
    ax.axvline(math.radians(180), color='black', linewidth=1, alpha=0.6)
    ax.axvline(math.radians(270), color='black', linewidth=1, alpha=0.6)
    ax.axvline(math.radians(0), color='black', linewidth=1, alpha=0.6)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi/2.0)
    ax.xaxis.set_tick_params(labelsize=25)
    return ax

def resample_histogram(histogram):
    number_of_times_to_sample = 100000
    seed = 210
    hd_cluster_r = robj.FloatVector(histogram)
    rejection_sampling_r = robj.r['rejection.sampling']
    resampled_distribution = rejection_sampling_r(number_of_times_to_sample, hd_cluster_r, seed)
    return resampled_distribution

def fit_von_mises_mixed_model(resampled_distribution):
    fit_von_mises_mix = robj.r('vMFmixture')
    find_best_fit = robj.r('vMFmin')
    fit = find_best_fit(fit_von_mises_mix(resampled_distribution))
    print(fit)
    return fit


def get_model_fit_alpha_value(fit):
    get_model_fit_alpha = robj.r('get_model_fit_alpha')
    alpha = get_model_fit_alpha(fit)
    return alpha


def get_model_fit_theta_value(fit):
    get_model_fit_theta = robj.r('get_model_fit_theta')
    theta = get_model_fit_theta(fit)
    return theta


def get_estimated_density_function(fit):
    get_estimated_density = robj.r('get_estimated_density')
    estimated_density = get_estimated_density(fit)
    return estimated_density

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def get_mode_angles_degrees(theta):
    angles = []
    for mode in range(int(len(theta)/2)):
        length, angle = cart2pol(np.asanyarray(theta)[mode][0], np.asanyarray(theta)[mode][1])
        angle *= 180 / np.pi
        angles.append(angle)
    return angles

def find_angles_and_lengths(theta):
    lengths = []
    angles = []
    number_of_modes = int(len(theta)/2)
    for mode in range(number_of_modes):
        length, angle = cart2pol(np.asanyarray(theta)[mode][0], np.asanyarray(theta)[mode][1])
        lengths.append(length)
        angles.append(angle)
    return angles, lengths

##end of core functions


##working on paralelising
inp=[3,6,9,12,15,18,21,24,27,30,40,50,60,70,80,90,100,150,200,250,300,350,400,450,500,600,700,800,900,1000]
x=105
y=105
trial_nums=20

max_firing_rate=np.zeros(len(inp))
max_firing_rate_hd=np.zeros(len(inp))
std_modes=np.zeros((len(inp), trial_nums))
std_modes[:]=-1
mode_count=[]
pearson_coefs_avg=std_modes=np.zeros((len(inp), trial_nums))

for a in np.arange(len(inp)):
    n_hd=inp[a]
    for c in np.arange(trial_nums):
        f = open('attempt2grids'+str(n_hd)+'trial'+str(c), 'rb')
        grid_cells_hd = pickle.load(f) #obj contains values
        f.close()

        soma_v_vec=grid_cells_hd[1]
        t_vec=grid_cells_hd[0]

        soma_array=soma_v_vec.to_python()
        peaks, _=find_peaks(soma_array,height=20)
        peak_times = [t_vec[i] for i in peaks] #ms
        peak_times=np.asarray(peak_times).astype(int)
        target_firing=positions[peak_times, :]
        hd_firing=hd[peak_times]

        x_firing=target_firing[:,0]
        y_firing=target_firing[:,1]

        values_cell1=[(t_array, positions[:,0], positions[:,1], hd)]
        labels = ['synced_time', 'position_x','position_y', 'hd']
        spatial_data = pd.DataFrame.from_records(values_cell1, columns=labels)

        cluster_id = np.arange(1,x_firing.shape[0]+1,1)
        values2=[(cluster_id, x_firing, y_firing, hd_firing)]
        labels=['cluster_id', 'position_x', 'position_y', 'hd'] #where x and y are firing locations
        firing_data_spatial=pd.DataFrame.from_records(values2, columns=labels)

        bin_size_cm = 2.5
        number_of_bins_x = math.ceil(x/ bin_size_cm)
        number_of_bins_y = math.ceil(y / bin_size_cm)

        shape=(number_of_bins_x, number_of_bins_y)
        firing_rate_map=np.zeros(shape)
        spikes_in_bin=np.zeros(shape)
        times_in_bin=np.zeros(shape)

        bn=number_of_bins_x*number_of_bins_y
        shape=(number_of_bins_y, number_of_bins_x, bn)
        spikes_in_bin_smoothed=np.zeros(shape)
        times_in_bin_smoothed=np.zeros(shape)

        for i in np.arange(number_of_bins_y):
            y_min=i*bin_size_cm
            y_max=(i+1)*bin_size_cm
            for j in np.arange(number_of_bins_x-1):
                x_min=j*bin_size_cm
                x_max=(j+1)*bin_size_cm

                spikes_in_bin[i,j]=len(x_firing[(target_firing[:,0]>=x_min)&(target_firing[:,0]<x_max) & (target_firing[:,1]>=y_min) & (target_firing[:,1]<y_max)])
                times_in_bin[i,j]=len(positions[(positions[:,0]>=x_min) & (positions[:,0]<x_max) & (positions[:,1]>=y_min) & (positions[:,1]<y_max),0])/1000 #in seconds

                if times_in_bin[i,j]==0:
                    firing_rate_map[i,j]=0

                else: 
                    firing_rate_map[i,j]=spikes_in_bin[i,j]/times_in_bin[i,j] #in Hz raw map

        firing_rate_map1=gaussian_filter(firing_rate_map, 2) #across two bin deviations
        max_firing_rate[a]=np.amax(firing_rate_map1)

        fig = plt.figure(figsize=(9,5))
        ax7 = fig.add_subplot(141)
        im=ax7.imshow(firing_rate_map1, interpolation='nearest', vmin=0, origin='upper', cmap='jet')
        ax7=style_open_field_plot(ax7)

        firing_rate_map2=np.copy(firing_rate_map1)
        firing_data_spatial=analyze_firing_fields(firing_data_spatial, spatial_data, firing_rate_map2)

        ax8 = fig.add_subplot(142)
        im=ax8.imshow(firing_rate_map2, interpolation='nearest', vmin=0, origin='upper', cmap='jet')
        ax8=style_open_field_plot(ax8)

        firing_data_spatial=process_grid_data(firing_rate_map1, firing_data_spatial)
        smooth_hd, preferred_direction, max_firing_rate_hd[a], time_per_bin2, r=get_hd_hist(hd_firing, hd) #for entire field
        #head direction
        theta = np.linspace(0, 2*np.pi, 361)  # x axis
        ax9 = fig.add_subplot(143, polar=True)
        ax9.plot(theta[:-1], smooth_hd, color='red', linewidth=2)
        ax9=style_polar_plot(ax9)

        firing_data_spatial = analyze_hd_in_firing_fields(firing_data_spatial, spatial_data)

        ax11 = fig.add_subplot(144, polar=True)
        ax11.plot(theta[:-1], firing_data_spatial.firing_fields_hd_cluster[0][0], linewidth=2, color='k')

        angles_here=[]
        mode_nums=[]
        ind=[]
        index=0
        for b in np.arange(len(firing_data_spatial.firing_fields_hd_cluster[0])):
            hd_hist=firing_data_spatial.firing_fields_hd_cluster[0][b] #smoothed and adjusted for 
            robj.r.source('count_modes_circular_histogram.R')
            if firing_data_spatial.firing_fields_hd_cluster[0][b]==[None]:
                print('skipping this field, it has none')
                index=index+1
            elif np.isnan(hd_hist).sum() > 0:
                print('skipping this field, it has nans')
                index=index+1
            else:
                print('I will analyze ')
                resampled_distribution = resample_histogram(hd_hist)
                fit = fit_von_mises_mixed_model(resampled_distribution)
                theta = get_model_fit_theta_value(fit)
                angles = get_mode_angles_degrees(theta)
                estimated_density = get_estimated_density_function(fit)
                angles_here.append(angles)
                mode_nums.append(len(angles))
                ind.append(index)
                index=index+1
        
        mode_count.append(mode_nums)
        std_modes[a,c] = circstd(sum(angles_here, []), high=180, low=-180)
        plt.show()
        
        pearson_coefs_cell=[]
        for index1 in ind:
            for index2 in ind:
                if index1 != index2:
                    field1=firing_data_spatial.firing_fields_hd_cluster[0][index1]
                    field2=firing_data_spatial.firing_fields_hd_cluster[0][index2]
                    pearson_coef = pearsonr(field1, field2)[0]
                    pearson_coefs_cell.append(pearson_coef)
        pearson_coefs_avg[a,c]=np.mean(pearson_coefs_cell)
        
boxes=[] #standard deviation of modes
for c in np.arange(std_modes.shape[0]):
    boxvals=std_modes[c,std_modes[c,:]>-1]
    boxes.append(boxvals)
    
counts=[] #number of modes
for i in np.arange(0,len(mode_count),20):
    mode_counts_all=np.zeros(20)
    for j in np.arange(20):
        mode_counts_all[j]=np.mean(mode_count[i+j])
    counts.append(hm)

corrs=[] #correlation
for c in np.arange(pearson_coefs_avg.shape[0]):
    boxvals=pearson_coefs_avg[c,pearson_coefs_avg[c,:]>-1]
    corrs.append(boxvals)
        
        
##working on rewriting into functions        
#Correlation coefficients
pearsonr(inp, np.median(corrs, axis=1))
pearsonr(inp, np.median(boxes, axis=1))
pearsonr(inp, np.median(counts, axis=1))

#kruskal wallis test: standard deviation of modes
stats.kruskal(mouse[0],rat[0],boxes[0], boxes[1], boxes[2], boxes[3], boxes[4], boxes[5], boxes[6], boxes[7], boxes[8], boxes[9], boxes[10], boxes[11], boxes[12], boxes[13], boxes[14], boxes[15], boxes[16], boxes[17], boxes[18], boxes[19], boxes[20], boxes[21], boxes[22], boxes[23], boxes[24], boxes[25], boxes[26], boxes[27], boxes[28], boxes[29])

#plot mouse 0-100
fig = plt.figure(figsize=(7,5))
ax7 = fig.add_subplot(111)
sns.regplot(x=inp[17:], y=np.median(boxes[17:], axis=1), color='darkblue', ci=None, truncate=True)
ax7.fill_between(inp[17:],np.median(boxes[17:], axis=1)-iqr(boxes[17:], axis=1),iqr(boxes[17:], axis=1)+np.median(boxes[17:], axis=1),color='darkblue', alpha=0.2)
plt.plot(inp[17:],[np.median(mouse[0])]*len(inp[17:]), color='red')
ax7.fill_between(inp[17:],[np.median(mouse[0])-iqr(mouse[0])]*len(inp[17:]),[iqr(mouse[0])+np.median(mouse[0])]*len(inp[17:]),color='red', alpha=0.1)
plt.xlabel('Number of inputs')
plt.rcParams.update({'font.size': 20})
plt.ylabel('Standard dev of modes')


#kruskal wallis test: pearson correlation coefficients of grid field hds
stats.kruskal(sum(mouse[5], [])), sum(rat[5], []), corrs[0], corrs[1], corrs[2], corrs[3], corrs[4], corrs[5], corrs[6], corrs[7], corrs[8], corrs[9], corrs[10], corrs[11], corrs[12], corrs[13], corrs[14], corrs[15], corrs[16], corrs[17], corrs[18], corrs[19], corrs[20], corrs[21], corrs[22], corrs[23], corrs[24], corrs[25], corrs[26], corrs[27], corrs[28], corrs[29])


#kruskal wallis test: number of modes
stats.kruskal(mouse[2], rat[2], counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], counts[6], counts[7], counts[8], counts[9], counts[10], counts[11], counts[12], counts[13], counts[14], counts[15], counts[16], counts[17], counts[18], counts[19], counts[20], counts[21], counts[22], counts[23], counts[24], counts[25], counts[26], counts[27], counts[28], counts[29])

