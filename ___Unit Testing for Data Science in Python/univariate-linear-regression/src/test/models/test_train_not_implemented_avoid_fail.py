
import pytest
import numpy as np

from models import train

class TestTrainModelNotImplemented(object):
    @pytest.mark.xfail(reason="Using TDD, train.train_model_not_implemented() is not implemented.")
    def test_on_linear_data(self):
        example_argument = np.array([[2081.0, 314942.0], [1059.0, 186606.0], [1697.0, 277794.0]])
        expected_value   = True
        actual_value     = train.train_model_not_implemented(example_argument)
        message          = 'This function is not implemented yet'
        assert expected_value == actual_value, message
    