
# Import libraries
import pytest

# Import the function to test
import numpy as np
import sys
sys.path.append("../univariate-linear-regression/src/features")
from as_numpy import get_data_as_numpy_array

# Complete the unit test name by adding a prefix
def test_on_clean_file():
  expected = np.array([[2081.0, 314942.0],
                       [1059.0, 186606.0],
                       [1148.0, 206186.0]
                       ]
                      )
  actual = get_data_as_numpy_array("datasets/example_clean_data2.txt", num_columns=2)
  message = "Expected return value: {0}, Actual return value: {1}".format(expected, actual)
  # Complete the assert statement
  assert actual == pytest.approx(expected), message
    