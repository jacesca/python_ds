
import pytest
import sys
import numpy as np

from features import as_numpy

@pytest.mark.xfail(reason="Using TDD, as_numpy.get_pandas_data is not implemented.")
class TestGetPandasData(object):
    def test_on_clean_file(self):
        expected = np.array([[2081.0, 314942.0],
                             [1059.0, 186606.0],
                             [1148.0, 206186.0]
                            ])
        actual = as_numpy.get_pandas_data("example_clean_data2.txt", num_columns=2)
        message = "Expected return value: {0}, Actual return value: {1}".format(expected, actual)
        assert (actual == expected).all()
        
@pytest.mark.skipif(sys.version_info > (2, 7), reason="requires Python 2.7 or lower.")
class TestGetDataAsNumpyArray(object):
    def test_on_clean_file(self):
        expected = np.array([[2081.0, 314942.0],
                             [1059.0, 186606.0],
                             [1148.0, 206186.0]
                            ])
        actual = as_numpy.get_data_as_numpy_array("example_clean_data2.txt", num_columns=2)
        # Requires Python 2.7 or lower
        message = unicode("Expected return value: {0}, Actual return value: {1}".format(expected, actual)) 
        assert (actual == expected).all()
    