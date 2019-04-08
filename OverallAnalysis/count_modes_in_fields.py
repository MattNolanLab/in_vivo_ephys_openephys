import numpy as np


# count modes in firing fields

number_of_times_to_sample = 1000
head_direction_histogram = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 6, 5, 2, 1]  # bimodal
resampled_distribution = rejection.sampling(number_of_times_to_sample, head_direction_histogram)