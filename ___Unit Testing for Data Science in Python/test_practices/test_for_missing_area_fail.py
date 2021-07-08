
import pytest
from _pytest.assertion import truncate
truncate.DEFAULT_MAX_LINES = 9999
truncate.DEFAULT_MAX_CHARS = 9999 

# The dummy function to evaluate
def row_to_list(row):
    # emulating a wrong function
    return [row.strip()]

# The unit test
def test_for_missing_area():
    val = '\t293,410\n'
    actual = row_to_list(val)
    expected = None
    message = "row_to_list({0}) returned {1} instead of {2}".format(repr(val), actual, expected)
    assert actual is expected, message
    