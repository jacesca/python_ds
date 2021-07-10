
import pytest
import os

from data.preprocessing_helpers import convert_to_int, row_to_list, preprocess

@pytest.fixture
def raw_and_clean_data_file(tmpdir):
    raw_data_file_path = tmpdir.join("raw.txt")
    clean_data_file_path = tmpdir.join("clean.txt")
    with open(raw_data_file_path, "w") as f: 
        f.write("1,801\t201,411\n1,767565,112\n2,002\t333,209\n1990\t782,911\n1,285\t389129\n")
    yield raw_data_file_path, clean_data_file_path
    # No teardown code necessary


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
        

class TestPreprocess(object):
    def test_on_raw_data(self, raw_and_clean_data_file):
        raw_path, clean_path = raw_and_clean_data_file 
        preprocess(raw_path, clean_path)
        
        with open(clean_path, 'r') as f: 
            lines = f.readlines() 
        
        assert lines[0] == "1801\t201411\n" # Compare first_line
        assert lines[1] == "2002\t333209\n" # Compare second_line
    