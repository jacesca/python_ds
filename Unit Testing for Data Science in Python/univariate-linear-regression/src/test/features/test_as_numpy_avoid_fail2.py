
import numpy as np
import pytest
import sys

# importin a dummy function
def get_data_as_numpy_array(clean_data_file_path, num_columns): 
    result = np.empty((0, num_columns)) 
    with open(clean_data_file_path, "r") as f: 
        rows = f.readlines() 
        for row_num in xrange(len(rows)): 
            try: 
                row = np.array([rows[row_num].rstrip("\ ").split("\t")], dtype=float) 
            except ValueError: 
                raise ValueError("Line {0} of {1} is badly formatted".format(row_num + 1, clean_data_file_path)) 
            else: 
                if row.shape != (1, num_columns): 
                    raise ValueError("Line {0} of {1} does not have {2} columns".format(
                        row_num + 1, clean_data_file_path, num_columns
                    )) 
            result = np.append(result, row, axis=0) 
    return result 

class TestGetDataAsNumpyArray(object):
    @pytest.mark.skipif(sys.version_info > (2, 7), reason="Works only on Python 2.7 or lower.")
    def test_on_clean_file(self):
        expected = np.array([[2081.0, 314942.0],
                             [1059.0, 186606.0],
                             [1148.0, 206186.0]
                             ]
                            )
        actual = get_data_as_numpy_array("example_clean_data.txt", num_columns=2)
        message = "Expected return value: {0}, Actual return value: {1}".format(expected, actual)
        assert actual == pytest.approx(expected), message
    