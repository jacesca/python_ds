
# Import the pytest package
import pytest

# Import the dummy function
def convert_to_int(string_with_comma):
    # emulating a wrong function
    return None

# Complete the unit test name by adding a prefix
def test_on_string_with_one_comma():
    test_argument = "2,081"
    expected = 2081
    actual = convert_to_int(test_argument)
    # Format the string with the actual return value
    message = "convert_to_int('2,081') should return the int 2081, but it actually returned {0}".format(actual)
    # Write the assert statement which prints message on failure
    assert actual == expected, message
    