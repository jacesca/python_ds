
# Import libraries
import pytest

# Import the function to test
import numpy as np
import sys
sys.path.append("../univariate-linear-regression/src/models")
from train import split_into_training_and_testing_sets

def test_valueerror_on_one_dimensional_argument():
    example_argument = np.array([2081, 314942, 1059, 186606, 1148, 206186])
    
    with pytest.raises(ValueError) as exception_info:
        # store the exception
        split_into_training_and_testing_sets(example_argument)
    
    # Check if ValueError contains correct message
    assert exception_info.match("Argument data_array must be two dimensional. Got 1 dimensional array instead!")
    