from PostSorting.spatial_information import get_spatial_info
import numpy as np

def test_spatial_info():

    # Active in one of the quarter portion
    test_map = np.zeros((40,40))
    test_map[:20,:20] = 2
    test_occ = np.ones((40,40))*10

    assert np.round(get_spatial_info(test_map,test_occ)) == 2 

    # Active in the top half
    test_map = np.zeros((40,40))
    test_map[:20,] = 2
    test_occ = np.ones((40,40))*10


    assert np.round(get_spatial_info(test_map,test_occ)) == 1