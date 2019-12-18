import numpy as np
import PostSorting.heading_direction


def test_calculate_heading_direction():
    x = [0, 1, 2, 2, 1]
    y = [0, 1, 1, 0, 1]

    desired_result = [45, 0, -90, -45]
    result = PostSorting.heading_direction.calculate_heading_direction(x, y)

    assert np.allclose(result, desired_result, rtol=1e-05, atol=1e-08)
