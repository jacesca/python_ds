
import pytest
import sys

from data.preprocessing_helpers import convert_to_int

class TestConvertToInt(object):
    @pytest.mark.skipif(sys.version_info > (2, 7), reason="Requires Python > 2.7")
    def test_with_no_comma(self):
        """Only runs on Python 2.7 or lower"""
        test_argument = "756"
        expected = 756
        actual = convert_to_int(test_argument)
        message = unicode("Expected: 2081, Actual: {0}".format(actual)) # Requires Python 2.7 or lower
        assert actual == expected, message
    