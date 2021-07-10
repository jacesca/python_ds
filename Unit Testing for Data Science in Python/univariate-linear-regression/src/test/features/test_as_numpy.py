
import pytest

from features.as_numpy import get_data_as_numpy_array
import numpy as np
import os


@pytest.fixture
def empty_file(tmpdir):
    file_path = tmpdir.join("empty.txt")
    open(file_path, "w").close()
    yield file_path
    

class TestGetDataAsNumpyArray(object):
    def test_on_clean_file(self):
        expected = np.array([[2081.0, 314942.0],
                             [1059.0, 186606.0],
                             [1148.0, 206186.0]
                            ])
        actual = get_data_as_numpy_array("example_clean_data2.txt", num_columns=2)
        message = "Expected return value: {0}, Actual return value: {1}".format(expected, actual)
        assert (actual == expected).all()


    def test_on_empty_file(self, empty_file):
        expected = np.empty((0, 2))
        actual = get_data_as_numpy_array(empty_file, 2)
        assert actual == pytest.approx(expected), "Expected: {0}, Actual: {1}".format(expected, actual)
    