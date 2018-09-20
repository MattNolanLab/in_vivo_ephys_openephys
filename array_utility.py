import numpy as np


# https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
def shift(array_to_shift, n):
    if n >= 0:
        return np.concatenate((np.full(n, np.nan), array_to_shift[:-n]))
    else:
        return np.concatenate((array_to_shift[-n:], np.full(-n, np.nan)))


# shift 2d array by n (not roll)
def shift_2d(array_to_shift, n, axis):
    shifted_array = np.empty_like(array_to_shift)
    if axis == 0:
        if n >= 0:
            shifted_array[:n, :] = np.nan
            shifted_array[n:, :] = array_to_shift[:-n]
        else:
            shifted_array[:, n:] = np.nan
            shifted_array[:, n] = array_to_shift[-n:]
    return shifted_array

