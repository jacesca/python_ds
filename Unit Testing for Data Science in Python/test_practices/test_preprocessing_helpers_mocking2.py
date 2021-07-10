
import pytest
import os
from unittest.mock import call

from pythoncode.preprocessing_helpers_wrong_converttoint import row_to_list, convert_to_int, preprocess

@pytest.fixture
def raw_and_clean_data_file(tmpdir):
    raw_data_file_path = tmpdir.join("raw.txt")
    clean_data_file_path = tmpdir.join("clean.txt")
    with open(raw_data_file_path, "w") as f: 
        f.write("1,801\t201,411\n"
                "1,767565,112\n"
                "2,002\t333,209\n"
                "1990\t782,911\n"
                "1,285\t389129\n"
                )
    yield raw_data_file_path, clean_data_file_path
    # No teardown code necessary

# Making the MagicMock() bug-free
def convert_to_int_bug_free(comma_separated_integer_string):
    # Assign to the dictionary holding the correct return values 
    return_values = {"1,801"  : 1801, 
                     "201,411": 201411, 
                     "2,002"  : 2002, 
                     "333,209": 333209, 
                     "1990"   : None, 
                     "782,911": 782911, 
                     "1,285"  : 1285, 
                     "389129" : None}
    # Return the correct result using the dictionary return_values
    return return_values[comma_separated_integer_string]

class TestRowToList(object):
    def test_on_normal_argument_1(self):
        actual = row_to_list("123\t4,567\n")
        expected = ["123", "4,567"]
        assert actual == expected, "Expected: {0}, Actual: {1}".format(expected, actual)
    
    def test_on_normal_argument_2(self):
        actual = row_to_list("1,059\t186,606\n")
        expected = ["1,059", "186,606"]
        assert actual == expected, "Expected: {0}, Actual: {1}".format(actual, expected)

    def test_on_no_tab_with_missing_value(self):      # (0, 1) case
        actual = row_to_list('\n')
        assert actual is None, "Expected: None, Actual: {0}".format(actual)
    
    def test_on_two_tabs_with_missing_value(self):    # (2, 1) case
        actual = row_to_list("123\t\t89\n")
        assert actual is None, "Expected: None, Actual: {0}".format(actual)

    def test_on_no_tab_no_missing_value(self):        # (0, 0) boundary value
        actual = row_to_list('123\n')
        assert actual is None, 'Expected: None, Actual: {0}'.format(actual)
    
    def test_on_two_tabs_no_missing_value(self):      # (2, 0) boundary value
        actual = row_to_list('123\t4,567\t89\n')
        assert actual is None, 'Expected: None, Actual: {0}'.format(actual)
    
    def test_on_one_tab_with_missing_value(self):     # (1, 1) boundary value
        actual = row_to_list('\t4,567\n')
        assert actual is None, 'Expected: None, Actual: {0}'.format(actual)
        

class TestConvertToInt(object):
    def test_with_no_comma(self):
        actual = convert_to_int("756")
        assert actual == 756, "Expected: 756, Actual: {0}".format(actual)
    
    def test_with_one_comma(self):
        actual = convert_to_int("2,081")
        assert actual == 2081, "Expected: 2081, Actual: {0}".format(actual)
    
    def test_with_two_commas(self):
        actual = convert_to_int("1,034,891")
        assert actual == 1034891, "Expected: 2081, Actual: {0}".format(actual)
    
    def test_on_string_with_missing_comma(self):
        actual = convert_to_int("178100,301")
        assert actual is None, "Expected: None, Actual: {0}".format(actual)
    
    def test_on_string_with_incorrectly_placed_comma(self):
        actual = convert_to_int("12,72,891")
        assert actual is None, "Expected: None, Actual: {0}".format(actual)
    
    def test_on_float_valued_string(self):
        actual = convert_to_int("23,816.92")
        assert actual is None, "Expected: None, Actual: {0}".format(actual)


class TestPreprocess(object):
    def test_on_raw_data(self, raw_and_clean_data_file, mocker):
        raw_path, clean_path = raw_and_clean_data_file 
        
        # Replace the dependency with the bug-free mock
        convert_to_int_mock = mocker.patch("pythoncode.preprocessing_helpers_wrong_converttoint.convert_to_int",
                                           side_effect=convert_to_int_bug_free)
        
        preprocess(raw_path, clean_path)
        
        # Check if preprocess() called the dependency correctly
        print(convert_to_int_mock.call_args_list)
        assert convert_to_int_mock.call_args_list == [call("1,801"), call("201,411"), 
                                                      call("2,002"), call("333,209"),
                                                      call("1990"),  call("782,911"), 
                                                      call("1,285"), call("389129")]
        
        with open(clean_path, 'r') as f: 
            lines = f.readlines() 
        
        assert lines[0] == "1801\t201411\n" # Compare first_line
        assert lines[1] == "2002\t333209\n" # Compare second_line
    