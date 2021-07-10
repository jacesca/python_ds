
import pytest

import sys
sys.path.append("../univariate-linear-regression/src/data")
from preprocessing_helpers import row_to_list

def test_on_normal_argument_1():
    actual = row_to_list("123\t4,567\n")
    expected = ["123", "4,567"]
    assert actual == expected, "Expected: {0}, Actual: {1}".format(expected, actual)
    
def test_on_normal_argument_2():
    actual = row_to_list("1,059\t186,606\n")
    expected = ["1,059", "186,606"]
    assert actual == expected, "Expected: {0}, Actual: {1}".format(actual, expected)
    