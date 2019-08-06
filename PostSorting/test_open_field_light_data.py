import PostSorting.open_field_light_data
import pandas as pd

def test_make_opto_data_frame():

    # pulses equally spaced and same length
    array_in = ([1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 17, 18, 19, 20, 21])
    desired_df = pd.DataFrame({'opto_start_times': [1, 9, 17],
                               'opto_end_times': [5, 13, 21]})
    result_df = PostSorting.open_field_light_data.make_opto_data_frame(array_in)
    assert desired_df.equals(result_df)

    # lengths of pulses are different
    array_in = ([1, 2, 3, 10, 11, 12, 13, 14, 21, 22, 23, 24, 25, 26, 27])
    desired_df = pd.DataFrame({'opto_start_times': [1, 10, 21],
                               'opto_end_times': [3, 14, 27]})
    result_df = PostSorting.open_field_light_data.make_opto_data_frame(array_in)
    assert desired_df.equals(result_df)

    # spacings between pulses are different
    array_in = ([1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 23, 24, 25, 26, 27])
    desired_df = pd.DataFrame({'opto_start_times': [1, 10, 23],
                               'opto_end_times': [5, 14, 27]})
    result_df = PostSorting.open_field_light_data.make_opto_data_frame(array_in)
    assert desired_df.equals(result_df)

    # lengths and spacing between pulses are different
    array_in = ([1, 2, 3, 10, 11, 12, 13, 14, 26, 27, 28, 29, 30, 31, 32])
    desired_df = pd.DataFrame({'opto_start_times': [1, 10, 26],
                               'opto_end_times': [3, 14, 32]})
    result_df = PostSorting.open_field_light_data.make_opto_data_frame(array_in)
    assert desired_df.equals(result_df)

    # pulse start != 1
    array_in = ([10, 11, 12, 13, 14, 26, 27, 28, 29, 30, 42, 43, 44, 45, 46])
    desired_df = pd.DataFrame({'opto_start_times': [10, 26, 42],
                               'opto_end_times': [14, 30, 46]})
    result_df = PostSorting.open_field_light_data.make_opto_data_frame(array_in)
    assert desired_df.equals(result_df)


    def main():
        test_make_opto_data_frame()

    if __name__ == '__main__':
        main()
