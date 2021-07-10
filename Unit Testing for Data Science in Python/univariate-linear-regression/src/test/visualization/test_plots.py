
import pytest
import numpy as np

from visualization import get_plot_for_best_fit_line
#from visualization.plots import get_plot_for_best_fit_line ## Another way to call the function.

# Declare the test class
class TestGetPlotForBestFitLine(object):
    @pytest.mark.mpl_image_compare  # Under the hood baseline generation and comparison
    def test_plot_for_linear_data(self):
        slope = 2.0
        intercept = 1.0
        x_array = np.array([1.0, 2.0, 3.0]) # Linear data set
        y_array = np.array([3.0, 5.0, 7.0])
        title = "Test plot for linear data"
        return get_plot_for_best_fit_line(slope, intercept, x_array, y_array, title)
        
    # Add the pytest marker which generates baselines and compares images
    @pytest.mark.mpl_image_compare
    def test_plot_for_almost_linear_data(self):
        slope = 5.0
        intercept = -2.0
        x_array = np.array([1.0, 2.0, 3.0])
        y_array = np.array([3.0, 8.0, 11.0])
        title = "Test plot for almost linear data"
        # Return the matplotlib figure returned by the function under test
        return get_plot_for_best_fit_line(slope, intercept, x_array, y_array, title)
    