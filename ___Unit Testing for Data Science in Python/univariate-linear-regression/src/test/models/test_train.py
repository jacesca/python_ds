
import pytest
import numpy as np

from models.train import split_into_training_and_testing_sets

# Declare the test class
class TestSplitIntoTrainingAndTestingSets(object):
    def test_on_six_rows(self):
        example_argument = np.array([[2081.0, 314942.0], [1059.0, 186606.0],
                                     [1148.0, 206186.0], [1506.0, 248419.0],
                                     [1210.0, 214114.0], [1697.0, 277794.0]]
                                    )
        # Fill in with training array's expected number of rows
        expected_training_array_num_rows = int(example_argument.shape[0]*0.75)
    
        # Fill in with testing array's expected number of rows
        expected_testing_array_num_rows = example_argument.shape[0] - expected_training_array_num_rows
    
        # Call the function to test
        actual = split_into_training_and_testing_sets(example_argument)
    
        # Write the assert statement checking training array's number of rows
        assert actual[0].shape[0] == expected_training_array_num_rows,             "The actual number of rows in the training array is not {}".format(expected_training_array_num_rows)
    
        # Write the assert statement checking testing array's number of rows
        assert actual[1].shape[0] == expected_testing_array_num_rows,             "The actual number of rows in the testing array is not {}".format(expected_testing_array_num_rows)

    
    def test_on_one_row(self):
        test_argument = np.array([[1382.0, 390167.0]])
        with pytest.raises(ValueError) as exc_info:
            split_into_training_and_testing_sets(test_argument)
        expected_error_msg = "Argument data_array must have at least 2 rows, it actually has just 1"
        assert exc_info.match(expected_error_msg)
    
    
    def test_valueerror_on_one_dimensional_argument(self):
        example_argument = np.array([2081, 314942, 1059, 186606, 1148, 206186])
    
        with pytest.raises(ValueError) as exception_info:
            # store the exception
            split_into_training_and_testing_sets(example_argument)
    
        # Check if ValueError contains correct message
        assert exception_info.match("Argument data_array must be two dimensional. Got 1 dimensional array instead!")
    