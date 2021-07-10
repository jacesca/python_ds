
import pytest
import numpy as np
from math import cos, pi, sin

from models.train import split_into_training_and_testing_sets, train_model, model_test

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


class TestTrainModel(object):
    def test_on_linear_data(self):
        test_argument = np.array([[1.0, 3.0],
                                  [2.0, 5.0],
                                  [3.0, 7.0]])
        expected_slope = 2.0
        expected_intercept = 1.0
    
        slope, intercept = train_model(test_argument)
    
        assert slope == pytest.approx(expected_slope)
        assert intercept == pytest.approx( expected_intercept )
    
    def test_on_positively_correlated_data(self):
        test_argument = np.array([[1.0, 4.0], [2.0, 4.0],
                                  [3.0, 9.0], [4.0, 10.0],
                                  [5.0, 7.0], [6.0, 13.0],])
        slope, intercept = train_model(test_argument)
        assert slope > 0
        

class TestModelTest(object):
    def test_on_perfect_fit(self):
        # Assign to a NumPy array containing a linear testing set
        test_argument = np.array([[1.0, 3.0], [2.0, 5.0], [3.0, 7.0]])
        # Fill in with the expected value of r^2 in the case of perfect fit
        expected = 1.0
        # Fill in with the slope and intercept of the model
        slope = (test_argument[-1,1]-test_argument[0,1])/(test_argument[-1,0]-test_argument[0,0])
        intercept = test_argument[0,1] - (slope * test_argument[0, 0]) 
        actual = model_test(test_argument, slope=slope, intercept=intercept)
        # Complete the assert statement
        assert actual == pytest.approx(expected), "Expected: {0}, Actual: {1}".format(expected, actual)
    
    def test_on_circular_data(self):
        theta = pi/4.0
        # Assign to a NumPy array holding the circular testing data
        test_argument = np.array([[1.0, 0.0], [cos(theta), sin(theta)],
                                  [0.0, 1.0],
                                  [cos(3 * theta), sin(3 * theta)],
                                  [-1.0, 0.0],
                                  [cos(5 * theta), sin(5 * theta)],
                                  [0.0, -1.0],
                                  [cos(7 * theta), sin(7 * theta)]]
                                 )
        # Fill in with the slope and intercept of the straight line
        actual = model_test(test_argument, slope=0.0, intercept=0.0)
        # Complete the assert statement
        assert actual == pytest.approx(0.0)
    