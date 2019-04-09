import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from rpy2 import robjects as robj
from rpy2.robjects import pandas2ri


# count modes in firing fields
robj.r.source('count_modes_circular_histogram.R')

number_of_times_to_sample = robj.r(1000)
hd_cluster_r = robj.FloatVector([1, 1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 6, 5, 2, 1])  # bimodal
# hd_cluster = pd.DataFrame({'hd': head_direction_histogram})
# hd_cluster_r = pandas2ri.py2ri(hd_cluster)
rejection_sampling_r = robj.r['rejection.sampling']
resampled_distribution = rejection_sampling_r(number_of_times_to_sample, hd_cluster_r)
print(resampled_distribution)