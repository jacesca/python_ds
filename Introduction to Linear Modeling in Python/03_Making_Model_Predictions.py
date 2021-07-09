# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 16:33:16 2020

@author: jacesca@gmail.com
Subject: Practicing Statistics Interview Questions in Python
Chapter 3: Making Model Predictions
    Next we will apply models to real data and make predictions. We will 
    explore some of the most common pit-falls and limitations of predictions, 
    and we evaluate and compare models by quantifying and contrasting several 
    measures of goodness-of-fit, including RMSE and R-squared. 
"""

###############################################################################
## Importing libraries
###############################################################################
import astropy.time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import dates as mdates
from matplotlib.ticker import StrMethodFormatter
from scipy.stats import linregress #Fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.
from sklearn.linear_model import LinearRegression #Calculate a linear least-squares regression for two sets of measurements. To get the parameters (slope and intercept) from a model
from statsmodels.formula.api import ols #Create a Model from a formula and dataframe.


###############################################################################
## Preparing the environment
###############################################################################
SEED = 42
np.random.seed(SEED) 

def plot_model(ax, x, y, a0, a1, y_model, x_model=np.NaN, x_future=np.NaN, y_future=np.NaN, rss=np.NaN, prediction_as_point=True, data_as_point=True, 
               title='', x_label='', y_label=''):
    """
    Plot the model predicted for relation between X and Y, detecting RSS.
    Parameters
    ----------
    ax : axis to plot
    x : numpy array. Data measured [Serie X].
    y : numpy array. Data measured [Serie Y].
    a0 : float. Interception of the lineal relation between x and y.
    a1 : float. Slope of the lineal relation between x and y.
    y_model : numpy array. Data modeled [Serie Y_Model]
    X_model : numpy array. Data modeled [Serie X_Model]
    X_future : numpy array. Data future[Serie X_Future]
    y_future : numpy array. Data predicted [Serie Y_Future]
    suptitle : String, Suptitle of the graph.
    prediction_as_point : Bool, To plot predicted data as point instead of line
    """
    if np.isnan(rss).any(): rss = np.square(y - y_model).sum() 
    if np.isnan(x_model).any(): x_model = x
    
    # Plotting the model
    ax.set_title('{}\nMinimum RSS = {:,.4f}'.format(title, rss), fontsize=12, color='maroon')  
    params = dict(linestyle=" ", ms=3, marker="o") if data_as_point else {}
    ax.plot(x, y, color='blue', **params, label='Measured data')
    ax.plot(x_model, y_model, lw=2, color='darkgoldenrod', label='Model: Y = {:,.4f} + {:,.4f} X'.format(a0, a1))
    if not(np.isnan(x_future).any() | np.isnan(x_future).any()): 
        if prediction_as_point==True:
            ax.plot(x_future, y_future, linestyle=" ", ms=5, marker="s" , color="red", label='Future data')
        else:
            ax.plot(x_future, y_future, lw=2, color="red", label='Future data')
    ax.tick_params(labelsize=6)
    ax.legend(loc='best', fontsize=6)
    ax.set_xlabel('X Data' if x_label=='' else x_label, fontsize=8)
    ax.set_ylabel('Y Data' if y_label=='' else y_label, fontsize=8)
    ax.grid(True)
    
    
def plot_data_and_model(axis, title, df, x_column, y_column, x_model, y_model, a0, a1, x_future=np.NaN, y_future=np.NaN, rss=np.NaN, prediction_as_point=True, data_as_point_first_graph=False):
    """
    Plot the model predicted for relation between X and Y, detecting RSS.
    Parameters
    ----------
    axis : axis to plot
    df : dataframe
    x_column : df column to plot as x
    y_column : df column to plot as y
    x_model : df column to plot as x_model
    y_model : df column to plot as y_model
    a0 : float. Interception of the lineal relation between x and y.
    a1 : float. Slope of the lineal relation between x and y.
    X_future : numpy array. Data future[Serie X_Future]
    y_future : numpy array. Data predicted [Serie Y_Future]
    suptitle : String, Suptitle of the graph.
    prediction_as_point : Bool, To plot predicted data as point instead of line
    data_as_point_first_graph : Bool, To plot data as point instead of line in the first graph. 
    """
    axis[0].set_title(title, fontsize=12, color='maroon')  
    params = dict(linestyle=" ", ms=3, marker="o") if data_as_point_first_graph else {}
    df.plot(x_column, y_column, ax=axis[0], **params)
    axis[0].legend().set_visible(False)
    axis[0].set_xlabel(x_column, fontsize=8)
    axis[0].set_ylabel(y_column, fontsize=8)
    axis[0].labelsize = 8
    axis[0].tick_params(labelsize=6)
    axis[0].grid(True)
    
    plot_model(axis[1], df[x_column], df[y_column], a0, a1, df[y_model], x_model=df[x_model], x_future=x_future, y_future=y_future, prediction_as_point=prediction_as_point)
    

def model_fit_and_predict(x, y):
    """
    Return the model and its parameters.
    Parameters
    ----------
    x : Data serie x.
    y : Data serie y.

    Returns
    -------
    slope, intercept, r_value, p_value, std_err 
    """
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    y_model = intercept + (slope * x)
    return y_model, slope, intercept, r_value, p_value, std_err 
    
    
###############################################################################
## Main part of the code
###############################################################################
def Modeling_Real_Data(seed=SEED):
    print("****************************************************")
    topic = "1. Modeling Real Data"; print("** %s\n" % topic)
    
    file = 'sea_level_data.csv'
    df   = pd.read_csv(file, skiprows=6)
    
    ###########################################################################
    print("** Method Scikit-Learn:")
    ###########################################################################
    # Load and shape the data
    x_data = df.year.values.reshape(-1,1)
    y_data = df.sea_level_inches.values.reshape(-1,1)
    
    # Initialize a general model
    model = LinearRegression(fit_intercept=True)
    
    # Fit the model to the data
    _ = model.fit(x_data, y_data)
    
    # Getting the modeled Y 
    y_model = model.predict(x_data)
    
    # Extract the linear model parameters
    intercept = model.intercept_[0]
    slope = model.coef_[0,0]
    print("Model found: Y = {:.4f} + {:.4f} X".format(intercept, slope))
    
    # Use the model to make predictions
    future_x = [[2020]]
    future_y = model.predict(future_x)
    
    fig, ax = plt.subplots()
    fig.suptitle("{} [through Scikit-Learn]".format(topic), fontsize=17, color='darkblue', weight='bold')
    plot_model(ax, x_data, y_data, intercept, slope, y_model, x_future=future_x, y_future=future_y)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=None); #To set the marginsplt.show() 
    plt.show()
    
    
    ###########################################################################
    print("\n** Method statsmodels:")
    ###########################################################################
    model_fit = ols(formula="sea_level_inches ~ year", data=df).fit()
    a0 = model_fit.params['Intercept']
    a1 = model_fit.params['year']
    e0 = model_fit.bse['Intercept']
    e1 = model_fit.bse['year']
    print("Model found: Y = {:.4f} + {:.4f} X [Intercept with uncertainty of {:.4f} and Slope with {:.4f}]".format(a0, a1, e0, e1))
    df['Sea_level_modeled'] = a0 + a1*df.year
    future_x = 2020
    future_y = a0 + a1*2020
        
    fig, axis = plt.subplots(1, 2, figsize=(11.5, 4))
    fig.suptitle("{} [through statsmodels]".format(topic), fontsize=17, color='darkblue', weight='bold')
    plot_data_and_model(axis, 'Sea Level Data', df, 'year', 'sea_level_inches', 'year', 'Sea_level_modeled', a0, a1, future_x, future_y)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=None); #To set the marginsplt.show() 
    plt.show()
    
    
    
def Linear_Model_in_Anthropology(seed=SEED):
    print("****************************************************")
    topic = "2. Linear Model in Anthropology"; print("** %s\n" % topic)
    
    file = 'femur_data.csv'
    df   = pd.read_csv(file)
    
    ###########################################################################
    print("** Method Scikit-Learn (Setting fit_intercept=False):")
    ## One difference found is the way to get the intercept. In this case,
    ## intercept=0.
    ###########################################################################
    # Load and shape the data
    legs = df.length.values.reshape(-1,1)
    heights = df.height.values.reshape(-1,1)
    
    # import the sklearn class LinearRegression and initialize the model
    model = LinearRegression(fit_intercept=False)
    
    # Prepare the measured data arrays and fit the model to them
    _ = model.fit(legs, heights)
    
    # Getting the modeled Y 
    df['height_model'] = model.predict(legs)
    
    # Extract the linear model parameters
    a0 = model.intercept_ #Intercept
    a1 = model.coef_[0,0] #Slope
    print("Model found: Y = {:.4f} + {:.4f} X".format(a0, a1))
    
    # Use the fitted model to make a prediction for the found femur
    fossil_leg = [[50.7]]
    fossil_height = model.predict(fossil_leg)
    print("Predicted fossil height = {:0.2f} cm".format(fossil_height[0,0]))
    
    fig, axis = plt.subplots(2, 2, figsize=(11.5, 5.5))
    fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
    title = 'Femur length vs Human Height Data\n[through Scikit-Learn - fit_intercept=False]'
    plot_data_and_model(axis[0, :], title, df, 'length', 'height', 'length', 'height_model', a0, a1, fossil_leg, fossil_height)
    
    
    ###########################################################################
    print("\n** Method Scikit-Learn (Setting fit_intercept=True):")
    ## One difference found is the way to get the intercept. In this case,
    ## intercept!=0.
    ###########################################################################
    # Load and shape the data
    legs = df.length.values.reshape(-1,1)
    heights = df.height.values.reshape(-1,1)
    
    # import the sklearn class LinearRegression and initialize the model
    model = LinearRegression(fit_intercept=True)
    
    # Prepare the measured data arrays and fit the model to them
    _ = model.fit(legs, heights)
    
    # Getting the modeled Y 
    df['height_model2'] = model.predict(legs)
    
    # Extract the linear model parameters
    a0 = model.intercept_[0] #Intercept
    a1 = model.coef_[0,0] #Slope
    print("Model found: Y = {:.4f} + {:.4f} X".format(a0, a1))
    
    # Use the fitted model to make a prediction for the found femur
    fossil_leg = [[50.7]]
    fossil_height = model.predict(fossil_leg)
    print("Predicted fossil height = {:0.2f} cm".format(fossil_height[0,0]))
    
    title = 'Femur length vs Human Height Data\n[through Scikit-Learn - fit_intercept=True]'
    plot_data_and_model(axis[1,:], title, df, 'length', 'height', 'length', 'height_model2', a0, a1, fossil_leg, fossil_height)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=.5, hspace=.5); #To set the marginsplt.show() 
    plt.show()
    
    
    
    
def Linear_Model_in_Oceanography(seed=SEED):
    print("****************************************************")
    topic = "3. Linear Model in Oceanography"; print("** %s\n" % topic)
    
    file = 'sea_level_data.csv'
    df   = pd.read_csv(file, skiprows=6)
    
    ###########################################################################
    print("** Method Scikit-Learn (Setting fit_intercept=False):")
    ## One difference found is the way to get the intercept. In this case,
    ## intercept=0.
    ###########################################################################
    # Load and shape the data
    years = df.year.values.reshape(-1,1)
    levels = df.sea_level_inches.values.reshape(-1,1)
    
    # import the sklearn class LinearRegression and initialize the model
    model = LinearRegression(fit_intercept=False)
    
    # Prepare the measured data arrays and fit the model to them
    _ = model.fit(years, levels)
    
    # Getting the modeled Y 
    df['sea_level_model'] = model.predict(years)
    
    # Extract the linear model parameters
    a0 = model.intercept_ #Intercept
    a1 = model.coef_[0,0] #Slope
    print("Model found: Y = {:.4f} + {:.4f} X".format(a0, a1))
    
    # Use the fitted model to make predictions
    future_year = 2020
    future_level = model.predict([[future_year]])
    print("For year {}, the predicted sea level is {:0.2f} cm".format(future_year, future_level[0,0]))
    
    fig, axis = plt.subplots(2, 2, figsize=(11.5, 5.5))
    fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
    title = 'Sea Level Data\n[through Scikit-Learn - fit_intercept=False]'
    plot_data_and_model(axis[0, :], title, df, 'year', 'sea_level_inches', 'year', 'sea_level_model', a0, a1, future_year, future_level)
    
    
    ###########################################################################
    print("\n** Method Scikit-Learn (Setting fit_intercept=True):")
    ## One difference found is the way to get the intercept. In this case,
    ## intercept!=0.
    ###########################################################################
    # Load and shape the data
    years = df.year.values.reshape(-1,1)
    levels = df.sea_level_inches.values.reshape(-1,1)
    
    # import the sklearn class LinearRegression and initialize the model
    model = LinearRegression(fit_intercept=True)
    
    # Prepare the measured data arrays and fit the model to them
    _ = model.fit(years, levels)
    
    # Getting the modeled Y 
    df['sea_level_model'] = model.predict(years)
    
    # Extract the linear model parameters
    a0 = model.intercept_[0] #Intercept
    a1 = model.coef_[0,0] #Slope
    print("Model found: Y = {:.4f} + {:.4f} X".format(a0, a1))
    
    # Use the fitted model to make predictions
    years_forecast = np.linspace(2013, 2020, 8).reshape(-1, 1)
    levels_forecast = model.predict(years_forecast)
    
    # Plot the model and the data
    title = 'Sea Level Data\n[through Scikit-Learn - fit_intercept=True]'
    plot_data_and_model(axis[1, :], title, df, 'year', 'sea_level_inches', 'year', 'sea_level_model', a0, a1, years_forecast, levels_forecast, prediction_as_point=False)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=.5, hspace=.5); #To set the marginsplt.show() 
    plt.show()
    
    
    
def Linear_Model_in_Cosmology(seed=SEED):
    print("****************************************************")
    topic = "4. Linear Model in Cosmology"; print("** %s\n" % topic)
    
    file = 'hubble_data.csv'
    df   = pd.read_csv(file, skiprows=8)
    
    # Fit the model, based on the form of the formula
    model_fit = ols(formula="velocities ~ distances", data=df).fit()
    
    # Extract the model parameters and associated "errors" or uncertainties
    a0 = model_fit.params['Intercept']
    a1 = model_fit.params['distances']
    e0 = model_fit.bse['Intercept']
    e1 = model_fit.bse['distances']
    
    # Defining the y values in the model
    df['velocities_modeled'] = a0 + a1*df.distances
    
    # Print the results
    print('For slope a1={:.02f}, the uncertainty in a1 is {:.02f}'.format(a1, e1))
    print('For intercept a0={:.02f}, the uncertainty in a0 is {:.02f}'.format(a0, e0))
        
    
    fig, axis = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("{} [through statsmodels]".format(topic), fontsize=17, color='darkblue', weight='bold')
    plot_data_and_model(axis, 'Velocity of galaxies - Hubble', df, 'distances', 'velocities', 'distances', 'velocities_modeled', a0, a1, data_as_point_first_graph=True)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=None); #To set the marginsplt.show() 
    plt.show()
    
    
    
def The_Limits_of_Prediction(seed=SEED):
    print("****************************************************")
    topic = "5. The Limits of Prediction"; print("** %s\n" % topic)
    
    # Format to use in the x-axis
    years = mdates.YearLocator()    # only print label for the years
    months = mdates.MonthLocator(bymonth=(4,7,10))  # mark months as ticks
    years_fmt = mdates.DateFormatter('%b\n%Y')
    fmt = mdates.DateFormatter('%b')

    # Reading the data
    start_date = np.datetime64('2008-01-02') # Start date of the data collection
    file = 'DJIA_daily.csv'
    df_daily   = pd.read_csv(file, index_col='Date', parse_dates=True)
    df_daily.columns = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
    
    df_monthly = df_daily.resample('MS').agg({'Open'      : 'first',
                                              'High'      : 'max',
                                              'Low'       : 'min',
                                              'Close'     : 'last',
                                              'Adj_Close' : 'last',
                                              'Volume'    : 'sum'})
    
    df_daily['Jday'] = [astropy.time.Time(row_date).jd for row_date in df_daily.index]
    df_daily['DayCount'] = [(row_date - start_date).days for row_date in df_daily.index]
    
    df_monthly['Jday'] = [astropy.time.Time(row_date).jd for row_date in df_monthly.index]
    df_monthly['DayCount'] = [(row_date - start_date + pd.offsets.Day(1)).days for row_date in df_monthly.index]
    
    
    ###########################################################################
    print("********************************************************")
    print("** Adj_Close ~ DayCount                               **")
    print("** Modeling for monthly and daily data                **")
    print("********************************************************")
    print("** Modeling the monthly data")
    ###########################################################################
    # Fit the model to df_daily
    model_fit = ols("Adj_Close ~ DayCount", data=df_monthly).fit()
    
    # Extract the model parameters and associated "errors" or uncertainties
    a0 = model_fit.params['Intercept']
    a1 = model_fit.params['DayCount']
    e0 = model_fit.bse['Intercept']
    e1 = model_fit.bse['DayCount']
    
    # Defining the y values in the model
    df_monthly['Adj_Close_modeled'] = a0 + a1*df_monthly.DayCount
    
    # Print the results
    print('For slope a1={:.02f}, the uncertainty in a1 is {:.02f}'.format(a1, e1))
    print('For intercept a0={:.02f}, the uncertainty in a0 is {:.02f}'.format(a0, e0))
    
    # Making some interpolation
    New_Date = np.datetime64('2014-01-11')
    New_DayCount = (New_Date - start_date).astype('timedelta64[D]').astype(int)
    Predicted_Adj_close = a0 + a1*New_DayCount
        
    # Plotting the model
    fig, axis = plt.subplots(2, 2, figsize=(10, 5.5))
    fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
    title = 'Dow Jones Industrial Average (DJIA) - Monthly Data'
    plot_data_and_model(axis[0, :], title, df_monthly.reset_index(), 'Date', 'Adj_Close', 
                        'Date', 'Adj_Close_modeled', a0, a1, 
                        New_Date, Predicted_Adj_close)
    axis[0,0].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    axis[0,1].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    axis[0,1].xaxis.set_major_locator(years)
    axis[0,1].xaxis.set_major_formatter(years_fmt)
    axis[0,1].xaxis.set_minor_locator(months)
    axis[0,1].xaxis.set_minor_formatter(fmt)
    plt.setp(axis[0,0].xaxis.get_minorticklabels(), fontsize=6)
    plt.setp(axis[0,0].xaxis.get_majorticklabels(), fontsize=6)
    plt.setp(axis[0,1].xaxis.get_minorticklabels(), fontsize=6)
    plt.setp(axis[0,1].xaxis.get_majorticklabels(), fontsize=6)
        
    ###########################################################################
    print("** Modeling the daily data")
    ###########################################################################
    # Fit the model to df_daily
    model_fit = ols("Adj_Close ~ DayCount", data=df_daily).fit()
    
    # Extract the model parameters and associated "errors" or uncertainties
    a0 = model_fit.params['Intercept']
    a1 = model_fit.params['DayCount']
    e0 = model_fit.bse['Intercept']
    e1 = model_fit.bse['DayCount']
    
    # Defining the y values in the model
    df_daily['Adj_Close_modeled'] = a0 + a1*df_daily.DayCount
    
    # Making some interpolation
    New_Date = np.datetime64('2014-01-11')
    New_DayCount = (New_Date - start_date).astype('timedelta64[D]').astype(int)
    Predicted_Adj_close = a0 + a1*New_DayCount
    
    # Print the results
    print('For slope a1={:.02f}, the uncertainty in a1 is {:.02f}'.format(a1, e1))
    print('For intercept a0={:.02f}, the uncertainty in a0 is {:.02f}'.format(a0, e0))
        
    title = 'Dow Jones Industrial Average (DJIA) - Daily Data'
    plot_data_and_model(axis[1, :], title, df_daily.reset_index(), 'Date', 'Adj_Close', 
                        'Date', 'Adj_Close_modeled', a0, a1, 
                        New_Date, Predicted_Adj_close)
    axis[1,0].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    axis[1,0].xaxis.set_major_locator(years)
    axis[1,0].xaxis.set_major_formatter(years_fmt)
    axis[1,0].xaxis.set_minor_locator(months)
    axis[1,0].xaxis.set_minor_formatter(fmt)
    axis[1,1].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    axis[1,1].xaxis.set_major_locator(years)
    axis[1,1].xaxis.set_major_formatter(years_fmt)
    axis[1,1].xaxis.set_minor_locator(months)
    axis[1,1].xaxis.set_minor_formatter(fmt)
    
    plt.setp(axis[1,0].xaxis.get_minorticklabels(), fontsize=6)
    plt.setp(axis[1,0].xaxis.get_majorticklabels(), fontsize=6, rotation=0)
    plt.setp(axis[1,1].xaxis.get_minorticklabels(), fontsize=6)
    plt.setp(axis[1,1].xaxis.get_majorticklabels(), fontsize=6)
    
    plt.subplots_adjust(left=None, bottom=.1, right=None, top=.85, wspace=.5, hspace=.5); #To set the marginsplt.show() 
    plt.show()
    
    
    ###########################################################################
    print("********************************************************")
    print("** Adj_Close ~ DayCount                               **")
    print("** Modeling for monthly and predicting for daily data **")
    print("********************************************************")
    ###########################################################################
    # Fit the model to df_daily
    model_fit = ols("Adj_Close ~ DayCount", data=df_monthly).fit()
    
    # Extract the model parameters and associated "errors" or uncertainties
    a0 = model_fit.params['Intercept']
    a1 = model_fit.params['DayCount']
    e0 = model_fit.bse['Intercept']
    e1 = model_fit.bse['DayCount']
    
    # Defining the y values in the model
    df_monthly['Adj_Close_modeled'] = model_fit.predict(df_monthly.DayCount)
    df_daily['Adj_Close_modeled'] = model_fit.predict(df_daily.DayCount)
    
    fig, axis = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("{}\nModeling for monthly and predicting for daily data".format(topic), fontsize=17, color='darkblue', weight='bold')
    plot_model(axis[0], df_monthly.index, df_monthly.Adj_Close, a0, a1, df_monthly.Adj_Close_modeled, 
               title='Monthly Model', x_label='Monthly Date', y_label='Adj_Close')
    axis[0].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    axis[0].xaxis.set_major_formatter(years_fmt)
    plot_model(axis[1], df_daily.index, df_daily.Adj_Close, a0, a1, df_daily.Adj_Close_modeled, 
               title='Daily Prediction', x_label='Daily Date', y_label='Adj_Close')
    axis[1].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    axis[1].xaxis.set_major_formatter(years_fmt)
    plt.subplots_adjust(left=None, bottom=.1, right=None, top=.75, wspace=.5, hspace=.5); #To set the marginsplt.show() 
    plt.show()
    
        
def Interpolation_Inbetween_Times(seed=SEED):
    print("****************************************************")
    topic = "6. Interpolation: Inbetween Times"; print("** %s\n" % topic)
    
    # Format to use in the x-axis
    years_fmt = mdates.DateFormatter('%b\n%Y')
    
    # Reading the data
    start_date = np.datetime64('2008-01-02') # Start date of the data collection
    file = 'DJIA_daily.csv'
    df_daily   = pd.read_csv(file, index_col='Date', parse_dates=True)
    df_daily.columns = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
    
    df_monthly = df_daily.resample('MS').agg({'Open'      : 'first',
                                              'High'      : 'max',
                                              'Low'       : 'min',
                                              'Close'     : 'last',
                                              'Adj_Close' : 'last',
                                              'Volume'    : 'sum'})
    
    df_daily['Jday'] = [astropy.time.Time(row_date).jd for row_date in df_daily.index]
    df_daily['DayCount'] = [(row_date - start_date).days for row_date in df_daily.index]
    
    df_monthly['Jday'] = [astropy.time.Time(row_date).jd for row_date in df_monthly.index]
    df_monthly['DayCount'] = [(row_date - start_date + pd.offsets.Day(1)).days for row_date in df_monthly.index]
    
    
    # Fit the model to df_daily
    model_fit = ols("Close ~ DayCount", data=df_monthly).fit()
    
    # Extract the model parameters and associated "errors" or uncertainties
    a0 = model_fit.params['Intercept']
    a1 = model_fit.params['DayCount']
    e0 = model_fit.bse['Intercept']
    e1 = model_fit.bse['DayCount']
    print('For slope a1={:.02f}, the uncertainty in a1 is {:.02f}'.format(a1, e1))
    print('For intercept a0={:.02f}, the uncertainty in a0 is {:.02f}'.format(a0, e0))
    
    # Defining the y values in the model
    df_monthly['Close_modeled'] = model_fit.predict(df_monthly.DayCount)
    df_daily['Close_modeled'] = model_fit.predict(df_daily.DayCount)
    
    fig, axis = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("{}\nModeling for monthly and predicting for daily data".format(topic), fontsize=17, color='darkblue', weight='bold')
    plot_model(axis[0], df_monthly.index, df_monthly.Close, a0, a1, df_monthly.Close_modeled, 
               title='Monthly Model', x_label='Monthly Date', y_label='Close')
    axis[0].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    axis[0].xaxis.set_major_formatter(years_fmt)
    plot_model(axis[1], df_daily.index, df_daily.Close, a0, a1, df_daily.Close_modeled, 
               title='Daily Prediction', x_label='Daily Date', y_label='Close')
    axis[1].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    axis[1].xaxis.set_major_formatter(years_fmt)
    plt.subplots_adjust(left=None, bottom=.1, right=None, top=.75, wspace=.5, hspace=.5); #To set the marginsplt.show() 
    plt.show()
    

def Extrapolation_Going_Over_the_Edge(seed=SEED):
    print("****************************************************")
    topic = "7. Extrapolation: Going Over the Edge"; print("** %s\n" % topic)
    
    # Load data
    x_data = np.array([-10. ,  -9.5,  -9. ,  -8.5,  -8. ,  -7.5,  -7. ,  -6.5,  -6. ,
                        -5.5,  -5. ,  -4.5,  -4. ,  -3.5,  -3. ,  -2.5,  -2. ,  -1.5,
                        -1. ,  -0.5,   0. ,   0.5,   1. ,   1.5,   2. ,   2.5,   3. ,
                         3.5,   4. ,   4.5,   5. ,   5.5,   6. ,   6.5,   7. ,   7.5,
                         8. ,   8.5,   9. ,   9.5,  10. ,  10.5,  11. ,  11.5,  12. ,
                        12.5,  13. ,  13.5,  14. ,  14.5,  15. ,  15.5,  16. ,  16.5,
                        17. ,  17.5,  18. ,  18.5,  19. ,  19.5,  20. ])

    y_data = np.array([  73.33885174,   91.52854842,   41.87555998,  103.06980499,
                         77.57108039,   99.70512917,  106.70722978,  128.26034956,
                        117.88171452,  136.65021987,   82.60474807,   86.82566796,
                        122.477045  ,  114.41893877,  127.63451229,  143.2255083 ,
                        136.61217437,  154.76845765,  182.39147012,  122.51909166,
                        161.78587909,  132.72560763,  210.81767421,  179.6837026 ,
                        181.98528167,  234.67907351,  246.48971034,  221.58691239,
                        250.3924093 ,  206.43287615,  303.75089312,  312.29865056,
                        323.8331032 ,  261.9686295 ,  316.64806585,  337.55295912,
                        360.13633529,  369.72729852,  408.0289548 ,  348.82736117,
                        394.93384188,  366.03460828,  374.7693763 ,  371.26981466,
                        377.88763074,  320.70120977,  336.82269401,  262.00816122,
                        290.35612838,  308.90807157,  259.98783618,  265.86978322,
                        271.12330621,  258.58229827,  241.52677418,  204.38155251,
                        198.05166573,  174.36397174,  190.97570971,  217.20785477,
                        146.83883158])
    
    # Make the model for a small portion of data
    sub_x = x_data[np.logical_and(x_data>=0, x_data<=10)]
    sub_y = y_data[np.where(np.logical_and(x_data>=0, x_data<=10))]
    a1, a0, _, _, _ = linregress(sub_x, sub_y)
    
    y_model = a0 + a1*x_data

    # Compute the residuals, "data - model", and determine where [residuals < tolerance]
    residuals = np.abs(y_data - y_model)
    tolerance = 100
    x_good = x_data[residuals < tolerance]
    y_good = y_model[residuals < tolerance]
    
    # Find the min and max of the "good" values, and plot y_data, y_model, and the tolerance range
    print('Minimum good x value = {}'.format(np.min(x_good)))
    print('Maximum good x value = {}'.format(np.max(x_good)))
    
    fig, ax = plt.subplots()
    fig.suptitle("{}\nHiking Trip Data Example".format(topic), fontsize=17, color='darkblue', weight='bold')
    ax.plot(sub_x, sub_y, linestyle=" ", ms=5, marker="o" , markeredgecolor="black", markerfacecolor='blue', label='Data used for the model')
    ax.plot(x_good, y_good, lw=10, alpha=0.3, color='green', label='Tolerance of the model ({} meters)'.format(tolerance))
    plot_model(ax, x_data, y_data, a0, a1, y_model,
               x_label='Step distance (Kilometers)', y_label='Altitude (Meters)')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the marginsplt.show() 
    plt.show()
    
    
def Goodness_of_Fit(seed=SEED):
    print("****************************************************")
    topic = "8. Goodness-of-Fit"; print("** %s\n" % topic)
    
    # Read the data
    file = 'sea_level_data.csv'
    df   = pd.read_csv(file, skiprows=6)
    
    # Build the model and compute the residuals "model - data"
    x_data = df.year
    y_data = df.sea_level_inches
    y_model, slope, intercept, r_value, p_value, std_err = model_fit_and_predict(x_data, y_data)
    
    print("Model found: Y = {:.4f} + {:.4f} X".format(intercept, slope))
    print("r_value: ", r_value, "\np_value: ", p_value, "\nstd_err: ", std_err)
    
    # Compute the residuals and the deviations
    residuals = y_model - y_data
    deviations = np.mean(y_data) - y_data
    
    ###########################################################################
    print("\nCalculate RMSE\n---------------------")
    ###########################################################################
    print("Using rmse = raiz(mean(square(residuals)))...")
    rss = np.sum(np.square(residuals))
    rmse = np.sqrt(np.mean(np.square(residuals)))
    print('RMSE = {:0.4f}, RSS = {:0.4f}'.format(rmse, rss))
    
    print("\nUsing rmse = raiz(mse); mse = rss/len(residuals)...")
    rss = np.sum(np.square(residuals))
    mse = rss/len(residuals)
    rmse = np.sqrt(mse)
    print('RMSE = {:0.4f}, MSE = {:0.4f}, RSS = {:0.4f}'.format(rmse, mse, rss))


    ###########################################################################
    print("\nCalculate r-square\n---------------------")
    ###########################################################################
    print("Using r-square = 1 - variance(residuals)/variance(deviations)...")
    var_residuals = np.mean(np.square(residuals))/len(residuals)
    var_deviations = np.mean(np.square(deviations))/len(deviations)
    r_squared = 1 - (var_residuals / var_deviations)
    print('R-squared = {:0.4f}'.format(r_squared))
    
    print("\nUsing numpy to calculate r-square...")
    r_squared = 1 - (residuals.var() / deviations.var())
    print('R-squared = {:0.4f}'.format(r_squared))
    
    print("\nUsing r-square = r-value**2")
    r_squared = np.square(r_value)
    print('R-squared = {:0.4f}'.format(r_squared))
    
    print("\nUsing r-square = correlation(y_data, y_model)...")
    dy_data = y_data - np.mean(y_data)
    dy_model = y_model - np.mean(y_model)
    zy_data = dy_data / np.std(dy_data)
    zy_model = dy_model / np.std(dy_model)
    correlation = np.mean(zy_data*zy_model)
    r_squared = np.square(correlation)
    print('R-squared = {:0.4f}, r_valuev = {:0.4f}'.format(r_squared, correlation))
    
    print("\nUsing r-square = correlation(y_data, y_model) = np.corrcoef(y_data, y_model)[0,1]...")
    correlation = np.corrcoef(y_data, y_model)[0,1]
    r_squared = np.square(correlation)
    print('R-squared = {:0.4f}, r_valuev = {:0.4f}'.format(r_squared, correlation))
    
    print("\nNotice that R-squared varies from 0 to 1, where a value of 1 means \
           that the model and the data are perfectly correlated and all \
           variation in the data is predicted by the model. ")
    
    
def RMSE_Step_by_step(seed=SEED):
    print("****************************************************")
    topic = "9. RMSE Step-by-step"; print("** %s\n" % topic)
    
    x_data = np.array([  0. ,   0.5,   1. ,   1.5,   2. ,   2.5,   3. ,   3.5,   4. ,
                         4.5,   5. ,   5.5,   6. ,   6.5,   7. ,   7.5,   8. ,   8.5,
                         9. ,   9.5,  10. ])
    y_data = np.array([ 161.78587909,  132.72560763,  210.81767421,  179.6837026 ,
                        181.98528167,  234.67907351,  246.48971034,  221.58691239,
                        250.3924093 ,  206.43287615,  303.75089312,  312.29865056,
                        323.8331032 ,  261.9686295 ,  316.64806585,  337.55295912,
                        360.13633529,  369.72729852,  408.0289548 ,  348.82736117,
                        394.93384188])
    
    # Build the model and compute the residuals "model - data"
    y_model, slope, intercept, r_value, p_value, std_err = model_fit_and_predict(x_data, y_data)
    residuals = y_model - y_data
    print("Model found: Y = {:.4f} + {:.4f} X".format(intercept, slope))
    
    # Compute the RSS, MSE, and RMSE and print the results
    RSS = np.sum(np.square(residuals))
    MSE = RSS/len(residuals)
    RMSE = np.sqrt(MSE)
    print('RMSE = {:0.2f}, MSE = {:0.2f}, RSS = {:0.2f}'.format(RMSE, MSE, RSS))


def R_Squared(seed=SEED):
    print("****************************************************")
    topic = "10. R-Squared"; print("** %s\n" % topic)
    
    x_data = np.array([  0. ,   0.5,   1. ,   1.5,   2. ,   2.5,   3. ,   3.5,   4. ,
                         4.5,   5. ,   5.5,   6. ,   6.5,   7. ,   7.5,   8. ,   8.5,
                         9. ,   9.5,  10. ])
    y_data = np.array([ 161.78587909,  132.72560763,  210.81767421,  179.6837026 ,
                        181.98528167,  234.67907351,  246.48971034,  221.58691239,
                        250.3924093 ,  206.43287615,  303.75089312,  312.29865056,
                        323.8331032 ,  261.9686295 ,  316.64806585,  337.55295912,
                        360.13633529,  369.72729852,  408.0289548 ,  348.82736117,
                        394.93384188])
    
    # Build the model and compute the residuals "model - data"
    y_model, slope, intercept, r_value, p_value, std_err = model_fit_and_predict(x_data, y_data)
    residuals = y_model - y_data
    print("Model found: Y = {:.4f} + {:.4f} X".format(intercept, slope))
    
    # Compute the residuals and the deviations
    residuals = y_model - y_data
    deviations = np.mean(y_data) - y_data
    
    # Compute the variance of the residuals and deviations
    var_residuals = np.mean(np.square(residuals))/len(residuals)
    var_deviations = np.mean(np.square(deviations))/len(deviations)
    
    # Compute r_squared as 1 - the ratio of RSS/Variance
    r_squared = 1 - (var_residuals / var_deviations)
    print('R-squared is {:0.2f}'.format(r_squared))



def Standard_Error(seed=SEED):
    print("****************************************************")
    topic = "11. Standard Error"; print("** %s\n" % topic)
    
    # Read the data
    file = 'hiking_data.csv'
    df   = pd.read_csv(file)
    
    # Build the model and compute the residuals "model - data"
    model_fit = ols(formula="distance ~ time", data=df).fit()
    
    # Get the params of the model
    a1 = model_fit.params['time']
    a0 = model_fit.params['Intercept']
    print("Model found: Y = {:.4f} + {:.4f} X".format(a0, a1))
    print("slope = {:.4f}".format(a1))
    print("intercept = {:.4f}".format(a0))
    
    # Get the standard error
    e0 = model_fit.bse['Intercept']
    e1 = model_fit.bse['time']
    print("standard_error_of_intercept = {:.4f}".format(e0))
    print("standard_error_of_slope = {:.4f}".format(e1))
    
    

def Variation_Around_the_Trend(seed=SEED):
    print("****************************************************")
    topic = "12. Variation Around the Trend"; print("** %s\n" % topic)
    
    x_data = np.array([ 0.        ,  0.08333333,  0.16666667,  0.25      ,  0.33333333,
                        0.41666667,  0.5       ,  0.58333333,  0.66666667,  0.75      ,
                        0.83333333,  0.91666667,  1.        ,  1.08333333,  1.16666667,
                        1.25      ,  1.33333333,  1.41666667,  1.5       ,  1.58333333,
                        1.66666667,  1.75      ,  1.83333333,  1.91666667,  2.        ])

    y_data = np.array([   4.87303609,    2.33139743,    6.74881808,    9.28109413,
                         19.26288955,   13.92871724,   30.23443529,   26.88304596,
                         34.29045062,   36.75188887,   46.05299048,   39.6529112 ,
                         49.03274839,   53.0145036 ,   61.73464166,   59.2003262 ,
                         66.14938204,   68.19975808,   75.12664124,   80.91511231,
                         80.0314758 ,   90.93417113,   94.37143883,   97.34081635,
                        102.70256785])
    
    # Store x_data and y_data, as times and distances, in df, and use ols() to fit a model to it.
    df = pd.DataFrame(dict(times=x_data, distances=y_data))
    model_fit = ols(formula="distances ~ times", data=df).fit()
    
    # Extact the model parameters and their uncertainties
    a0 = model_fit.params['Intercept']
    e0 = model_fit.bse['Intercept']
    a1 = model_fit.params['times']
    e1 = model_fit.bse['times']
    
    # Print the results with more meaningful names
    print('Estimate    of the intercept = {:0.4f}'.format(a0))
    print('Uncertainty of the intercept =  {:0.4f}'.format(e0))
    print('Estimate    of the slope     = {:0.4f}'.format(a1))
    print('Uncertainty of the slope     =  {:0.4f}'.format(e1))
    
    

def Variation_in_Two_Parts(seed=SEED):
    print("****************************************************")
    topic = "13. Variation in Two Parts"; print("** %s\n" % topic)

    df = pd.DataFrame({'times'     : [   0.        ,  0.08333333,  0.16666667,  0.25      ,  0.33333333,
                                         0.41666667,  0.5       ,  0.58333333,  0.66666667,  0.75      ,
                                         0.83333333,  0.91666667,  1.        ,  1.08333333,  1.16666667,
                                         1.25      ,  1.33333333,  1.41666667,  1.5       ,  1.58333333,
                                         1.66666667,  1.75      ,  1.83333333,  1.91666667,  2.        ],
                       'distances1': [  16.24345364,   -1.95089747,    3.05161581,    1.77031378,
                                        25.32074296,   -2.18205364,   42.44811764,   21.55459766,
                                        36.52372429,   35.00629625,   56.28774604,   25.23192624,
                                        46.77582796,   50.32612312,   69.67102776,   51.50108733,
                                        64.94238459,   62.05474915,   75.42213747,   84.9948188 ,
                                        72.32714156,   98.9472371 ,  100.68257387,  100.85827672,
                                       109.00855949],
                       'distances2': [  16.24345364,  -5.2842308 ,  -3.61505086,  -8.22968622,
                                        11.98740963, -18.8487203 ,  22.44811764,  -1.77873568,
                                         9.85705763,   5.00629625,  22.9544127 , -11.43474043,
                                         6.77582796,   6.99278979,  23.00436109,   1.50108733,
                                        11.60905126,   5.38808249,  15.42213747,  21.66148547,
                                         5.66047489,  28.9472371 ,  27.34924054,  24.19161006,  29.00855949]})
    
    # Build and fit two models, for columns distances1 and distances2 in df
    model_1 = ols(formula="distances1 ~ times", data=df).fit()
    model_2 = ols(formula="distances2 ~ times", data=df).fit()
    df['d1_model'] = model_1.predict(df.times)
    df['d2_model'] = model_2.predict(df.times)
    
    # Extact the model1 parameters and their uncertainties
    a0_1 = model_1.params['Intercept']
    e0_1 = model_1.bse['Intercept']
    a1_1 = model_1.params['times']
    e1_1 = model_1.bse['times']
    se_1 = model_1.bse['times']
    
    # Extact the model2 parameters and their uncertainties
    a0_2 = model_1.params['Intercept']
    e0_2 = model_1.bse['Intercept']
    a1_2 = model_1.params['times']
    e1_2 = model_1.bse['times']
    se_2 = model_1.bse['times']
    
    # Extract R-squared for each model, and the standard error for each slope
    se_1 = model_1.bse['times']
    se_2 = model_2.bse['times']
    rsquared_1 = model_1.rsquared
    rsquared_2 = model_2.rsquared
    
    # Print the results
    msg1 = 'Model 1: SE = {:0.3f}, R-squared = {:0.3f}'.format(se_1, rsquared_1); print(msg1);
    msg2 = 'Model 2: SE = {:0.3f}, R-squared = {:0.3f}'.format(se_2, rsquared_2); print(msg2);
    
    
    fig = plt.figure(figsize=(12, 4.5))
    fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
    ax1 = plt.subplot2grid((2, 3), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((2, 3), (0, 1))
    ax3 = plt.subplot2grid((2, 3), (0, 2))
    ax4 = plt.subplot2grid((2, 3), (1, 1))
    ax5 = plt.subplot2grid((2, 3), (1, 2))
    
    df.plot('times', 'distances1', ax=ax1, linestyle=" ", ms=3, marker="o", label='distances1')
    df.plot('times', 'distances2', ax=ax1, linestyle=" ", ms=3, marker="o", label='distances2')
    ax1.legend(loc='best')
    ax1.set_ylabel("distances")
    
    plot_data_and_model((ax2, ax3), msg1, df, 'times', 'distances1', 
                        'times', 'd1_model', a0_1, a1_1, data_as_point_first_graph=True)
    plot_data_and_model((ax4, ax5), msg2, df, 'times', 'distances2', 
                        'times', 'd2_model', a0_2, a1_2, data_as_point_first_graph=True)
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=.5, hspace=.5); #To set the marginsplt.show() 
    plt.show()
    
    
    
        
def main():
    print("****************************************************")
    print("** BEGIN                                          **")
    
    Modeling_Real_Data()
    Linear_Model_in_Anthropology()
    Linear_Model_in_Oceanography()
    Linear_Model_in_Cosmology()
    The_Limits_of_Prediction()
    Interpolation_Inbetween_Times()
    Extrapolation_Going_Over_the_Edge()
    Goodness_of_Fit()
    RMSE_Step_by_step()
    R_Squared()
    Standard_Error()
    Variation_Around_the_Trend()
    Variation_in_Two_Parts()
        
    print("****************************************************")
    print("** END                                            **")
    print("****************************************************")

if __name__ == '__main__':
    main()