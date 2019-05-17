#using Klara's core functions: PostSorting/open_field_spatial_data.py

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
import pickle
import matplotlib.ticker as ticker
from sympy import Eq, Symbol, solve
from scipy.optimize import curve_fit
from random import seed
from scipy.ndimage import gaussian_filter
import Param_mod
import sampling_analysis

def convert_time_to_seconds(position_data):
    position_data['hours'], position_data['minutes'], position_data['seconds'] = position_data['time'].str.split(':', 2).str
    position_data['hours'] = position_data['hours'].astype(int)
    position_data['minutes'] = position_data['minutes'].astype(int)
    position_data['seconds'] = position_data['seconds'].astype(float)
    position_data['time_seconds'] = position_data['hours'] * 3600 + position_data['minutes']*60 + position_data['seconds']
    position_data['time_seconds'] = position_data['time_seconds'] - position_data['time_seconds'][0]
    return position_data


def read_position(path_to_bonsai_file):
    position_data = pd.read_csv(path_to_bonsai_file, sep=' ', header=None)
    if len(list(position_data)) > 6:
        position_data = position_data.drop([6], axis=1)  # remove column of NaNs due to extra space at end of lines
    position_data['date'], position_data['time'] = position_data[0].str.split('T', 1).str

    position_data['time'], position_data['str_to_remove'] = position_data['time'].str.split('+', 1).str
    position_data = position_data.drop([0, 'str_to_remove'], axis=1)  # remove first column that got split into date and time

    position_data.columns = ['x_left', 'y_left', 'x_right', 'y_right', 'syncLED', 'date', 'time']
    position_data = convert_time_to_seconds(position_data)
    return position_data


def calculate_speed(position_data):
    elapsed_time = position_data['time_seconds'].diff()
    distance_travelled = np.sqrt(position_data['x_left'].diff().pow(2) + position_data['y_left'].diff().pow(2))
    position_data['speed_left'] = distance_travelled / elapsed_time
    distance_travelled = np.sqrt(position_data['x_right'].diff().pow(2) + position_data['y_right'].diff().pow(2))
    position_data['speed_right'] = distance_travelled / elapsed_time
    return position_data


def calculate_central_speed(position_data):
    elapsed_time = position_data['time_seconds'].diff()
    distance_travelled = np.sqrt(position_data['position_x'].diff().pow(2) + position_data['position_y'].diff().pow(2))
    position_data['speed'] = distance_travelled / elapsed_time
    return position_data


def remove_jumps(position_data, prm):
    max_speed = 1  # m/s, anything above this is not realistic
    pixel_ratio = prm.get_pixel_ratio()
    max_speed_pixels = max_speed * pixel_ratio
    speed_exceeded_left = position_data['speed_left'] > max_speed_pixels
    position_data['x_left_without_jumps'] = position_data.x_left[speed_exceeded_left == False]
    position_data['y_left_without_jumps'] = position_data.y_left[speed_exceeded_left == False]

    speed_exceeded_right = position_data['speed_right'] > max_speed_pixels
    position_data['x_right_without_jumps'] = position_data.x_right[speed_exceeded_right == False]
    position_data['y_right_without_jumps'] = position_data.y_right[speed_exceeded_right == False]

    remains_left = (len(position_data) - speed_exceeded_left.sum())/len(position_data)*100
    remains_right = (len(position_data) - speed_exceeded_right.sum())/len(position_data)*100
    print('{} % of right side tracking data, and {} % of left side'
          ' remains after removing the ones exceeding speed limit.'.format(remains_right, remains_left))
    return position_data


def get_distance_of_beads(position_data):
    distance_between_beads = np.sqrt((position_data['x_left'] - position_data['x_right']).pow(2) + (position_data['y_left'] - position_data['y_right']).pow(2))
    return distance_between_beads


def remove_points_where_beads_are_far_apart(position_data):
    minimum_distance = 40
    distance_between_beads = get_distance_of_beads(position_data)
    distance_exceeded = distance_between_beads > minimum_distance
    position_data['x_left_cleaned'] = position_data.x_left_without_jumps[distance_exceeded == False]
    position_data['x_right_cleaned'] = position_data.x_right_without_jumps[distance_exceeded == False]
    position_data['y_left_cleaned'] = position_data.y_left_without_jumps[distance_exceeded == False]
    position_data['y_right_cleaned'] = position_data.y_right_without_jumps[distance_exceeded == False]
    return position_data


def curate_position(position_data, prm):
    position_data = remove_jumps(position_data, prm)
    position_data = remove_points_where_beads_are_far_apart(position_data)
    return position_data


def calculate_position(position_data):
    position_data['position_x_tmp'] = (position_data['x_left_cleaned'] + position_data['x_right_cleaned']) / 2
    position_data['position_y_tmp'] = (position_data['y_left_cleaned'] + position_data['y_right_cleaned']) / 2

    position_data['position_x'] = position_data['position_x_tmp'].interpolate()  # interpolate missing data
    position_data['position_y'] = position_data['position_y_tmp'].interpolate()
    return position_data

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def calculate_head_direction(position):
    position['head_dir_tmp'] = np.degrees(np.arctan((position['y_left_cleaned'] + position['y_right_cleaned']) / (position['x_left_cleaned'] + position['x_right_cleaned'])))
    rho, hd = cart2pol(position['x_right_cleaned'] - position['x_left_cleaned'], position['y_right_cleaned'] - position['y_left_cleaned'])
    position['hd'] = np.degrees(hd)
    position['hd'] = position['hd'].interpolate()  # interpolate missing data
    return position


def convert_to_cm(position_data, params):
    pixel_ratio = params.get_pixel_ratio()
    position_data['position_x_pixels'] = position_data.position_x
    position_data['position_y_pixels'] = position_data.position_y
    position_data['position_x'] = position_data.position_x / pixel_ratio * 100
    position_data['position_y'] = position_data.position_y / pixel_ratio * 100
    return position_data


def shift_to_start_from_zero_at_bottom_left(position_data):
    # this is copied from MATLAB script, 0.0001 is here to 'avoid bin zero in first point'
    position_data['position_x'] = position_data.position_x - min(position_data.position_x[~np.isnan(position_data.position_x)])
    position_data['position_y'] = position_data.position_y - min(position_data.position_y[~np.isnan(position_data.position_y)])
    return position_data


def get_sides(position_data):
    left_side_edge = position_data.position_x.round().min()
    right_side_edge = position_data.position_x.round().max()
    top_side_edge = position_data.position_y.round().max()
    bottom_side_edge = position_data.position_y.round().min()
    return left_side_edge, right_side_edge, top_side_edge, bottom_side_edge


def remove_edge_from_horizontal_side(position_data, left_side_edge, right_side_edge):
    points_on_left_edge = np.where(position_data.position_x < (left_side_edge + 1))[0]
    number_of_points_on_left_edge = len(points_on_left_edge)
    points_on_right_edge = np.where(position_data.position_x > (right_side_edge - 1))[0]
    number_of_points_on_right_edge = len(points_on_right_edge)

    if number_of_points_on_left_edge > number_of_points_on_right_edge:
        # remove left edge
        position_data = position_data.drop(position_data.index[points_on_right_edge])
    else:
        # remove right edge
        position_data = position_data.drop(position_data.index[points_on_left_edge])
    return position_data


def remove_edge_from_vertical_side(position_data, top_side_edge, bottom_side_edge):
    points_on_top_edge = np.where(position_data.position_y > (top_side_edge - 1))[0]
    number_of_points_on_top_edge = len(points_on_top_edge)
    points_on_bottom_edge = np.where(position_data.position_y < (bottom_side_edge + 1))[0]
    number_of_points_on_bottom_edge = len(points_on_bottom_edge)

    if number_of_points_on_top_edge > number_of_points_on_bottom_edge:
        # remove left edge
        position_data = position_data.drop(position_data.index[points_on_bottom_edge])
    else:
        # remove right edge
        position_data = position_data.drop(position_data.index[points_on_top_edge])
    return position_data


def get_dimensions_of_arena(position_data):
    left_side_edge, right_side_edge, top_side_edge, bottom_side_edge = get_sides(position_data)
    x_length = right_side_edge - left_side_edge
    y_length = top_side_edge - bottom_side_edge
    return x_length, y_length


def remove_position_outlier_rows(position_data):
    is_square = False
    x_length, y_length = get_dimensions_of_arena(position_data)
    while is_square is False:
        left_side_edge, right_side_edge, top_side_edge, bottom_side_edge = get_sides(position_data)
        if x_length == y_length:
            is_square = True
        elif x_length > y_length:
            position_data = remove_edge_from_horizontal_side(position_data, left_side_edge, right_side_edge)
        else:
            position_data = remove_edge_from_vertical_side(position_data, top_side_edge, bottom_side_edge)
        x_length, y_length = get_dimensions_of_arena(position_data)
    return position_data


def process_position_data(path, params, res_t):
    position_of_mouse = None
    position_data = read_position(path)  # raw position data from bonsai output
    position_data = calculate_speed(position_data)
    position_data = curate_position(position_data, params)  # remove jumps from data, and when the beads are far apart
    position_data = calculate_position(position_data)  # get central position and interpolate missing data
    position_data = calculate_head_direction(position_data)  # use coord from the two beads to get hd and interpolate
    position_data = shift_to_start_from_zero_at_bottom_left(position_data)
    # position_data = remove_position_outlier_rows(position_data)
    position_data = convert_to_cm(position_data, params)
    x_length, y_length = get_dimensions_of_arena(position_data)
    position_data = calculate_central_speed(position_data)
    position_of_mouse = position_data[['time_seconds', 'position_x', 'position_x_pixels', 'position_y', 'position_y_pixels', 'hd', 'syncLED', 'speed']].copy()
    #sampling_analysis.check_if_hd_sampling_was_high_enough(position_of_mouse, params)
    t=position_data.time_seconds.values
    location_x=position_data.position_x.values
    location_y=position_data.position_y.values
    #create bins
    max_t = max(t)
    t_array=np.arange(0,max_t,res_t) #1 s bins, 0-1, 1-2...
    shape=(t_array.shape[0],2)
    positions=np.zeros(shape)
    h=position_data.hd
    shape=(t_array.shape[0])
    hd=np.zeros(shape)

    for i in np.arange(t_array.shape[0]):
        positions[i,0]=np.mean(location_x[(t>=t_array[i]) & (t<(t_array[i]+res_t))])
        positions[i,1]=np.mean(location_y[(t>=t_array[i]) & (t<(t_array[i]+res_t))])
        hd[i]=np.mean(h[(t>=t_array[i]) & (t<(t_array[i]+res_t))])
    
    df = pd.DataFrame(positions[:,0]).interpolate()
    positions[:,0]=df.values[:,0]
    df2 = pd.DataFrame(positions[:,1]).interpolate()
    positions[:,1]=df2.values[:,0]
    
    df3 = pd.DataFrame(hd).interpolate()
    hd=df3.values
    hd=hd+180
    
    return positions, x_length, y_length, hd, t_array
    
    
#run function
prm=Param_mod.Parameters()
prm.set_pixel_ratio(440)
path='./behavior_open_field/M5-0313-of.csv'
res=1 #cm
res_t=0.001 #0.001 ms run
positions, x, y, hd, t_array=process_position_data(path, prm, res_t)


#save output
f = open('single_t_array', 'wb')
pickle.dump(t_array, f)
f.close()

f = open('single_positions', 'wb')
pickle.dump(positions, f)
f.close()

f = open('single_hd', 'wb')
pickle.dump(hd, f)
f.close()

