
# Import the pytest package
import pytest

# Import the function convert_to_int()
import sys
sys.path.append("../univariate-linear-regression/src/data")
from preprocessing_helpers import convert_to_int

# Complete the unit test name by adding a prefix
def test_on_string_with_one_comma():
    return_value = convert_to_int("2,081")
    assert isinstance(return_value, int)
    assert return_value == 2081
    