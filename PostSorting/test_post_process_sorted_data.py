import PostSorting.post_process_sorted_data
import numpy as np


def test_process_running_parameter_tag():
    tags = 'interleaved_opto*test1*cat'
    result = PostSorting.post_process_sorted_data.process_running_parameter_tag(tags)
    desired_result = False, True, False
    assert np.allclose(result, desired_result, rtol=1e-05, atol=1e-08)


def main():
    test_process_running_parameter_tag()


if __name__ == '__main__':
    main()
