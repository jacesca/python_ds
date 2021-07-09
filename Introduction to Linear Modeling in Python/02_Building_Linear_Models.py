# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:36:32 2020

@author: jacqueline.cortez
Subject: Practicing Statistics Interview Questions in Python
Chapter 2: Building Linear Models
    Here we look at the parts that go into building a linear model. Using the 
    concept of a Taylor Series, we focus on the parameters slope and intercept, 
    how they define the model, and how to interpret the them in several applied 
    contexts. We apply a variety of python modules to find the model that best 
    fits the data, by computing the optimal values of slope and intercept, 
    using least-squares, numpy, statsmodels, and scikit-learn.
"""

###############################################################################
## Importing libraries
###############################################################################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import optimize #Provides several commonly used optimization algorithms
from scipy.stats import linregress #Fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.
from sklearn.linear_model import LinearRegression #Calculate a linear least-squares regression for two sets of measurements. To get the parameters (slope and intercept) from a model
from statsmodels.formula.api import ols #Create a Model from a formula and dataframe.

###############################################################################
## Preparing the environment
###############################################################################
SEED = 42
np.random.seed(SEED) 

# Define the general model as a function
def model(x, a0=3, a1=2, a2=0):
    """
    Retrive y from the function Y = a0 + a1 * X + a2 * X**2
    Parameters
    ----------
    x : Numpy array. Values of x.
    a0 : Float. The default is 3.
    a1 : Float. The default is 2.
    a2 : Float. The default is 0.

    Returns
    -------
    Numpy array. Values of y.
    """
    return a0 + (a1*x) + (a2*x**2)
    

###############################################################################
## Main part of the code
###############################################################################
def What_makes_a_model_linear():
    print("****************************************************")
    topic = "1. What makes a model linear"; print("** %s\n" % topic)
    
    x = np.linspace(0, 5, 1000)
    Taylor_Series = {'Series terms: a₀=1': {'i'    : 0,
                                            'y'    : model(x, a0=1, a1=0, a2=0),
                                            'color': 'red',
                                            'label': 'Y = 1'},
                     'Series terms: a₁=1': {'i'    : 1,
                                            'y'    : model(x, a0=0, a1=1, a2=0),
                                            'color': 'limegreen',
                                            'label': 'Y = X'},
                     'Series terms: a₂=1': {'i'    : 2,
                                            'y'    : model(x, a0=0, a1=0, a2=1),
                                            'color': 'blue',
                                            'label': 'Y = X²'}} 
    
    # Plotting the model 
    fig, axis = plt.subplots(2, 2, figsize=(10.5, 5.5))
    fig.suptitle("{}\nTaylor Series: Y = a₀ + a₁X + a₂X² + ...".format(topic), fontsize=17, color='darkblue', weight='bold')
    
    for name, data in Taylor_Series.items():
        ax = axis[0 if data['i']<2 else 1, data['i'] if data['i']<2 else data['i']-2]
        ax.set_xlim(0,5); ax.set_ylim(0, 5);
        ax.set_title(name, fontsize=12, color='maroon')  
        ax.plot(x, data['y'], lw=2, color=data['color'], label=data['label'])
        ax.tick_params(labelsize=6)
        ax.legend(loc='best', fontsize=8)
        ax.set_xlabel('X Data', fontsize=8)
        ax.set_ylabel('Y Data', fontsize=8)
        ax.grid(True)
        
        ax = axis[1, 1]
        ax.plot(x, data['y'], lw=2, color=data['color'], label=data['label'])
        
    #ax = axis[1,1]
    y = model(x, a0=1, a1=1, a2=1)
    ax.plot(x, y, lw=2, color='black', label='Y = 1 + x + X²')
    ax.set_xlim(0,5); ax.set_ylim(0, 5);
    ax.set_title('Combining all Terms', fontsize=12, color='maroon')  
    ax.tick_params(labelsize=6)
    ax.legend(loc='best', fontsize=8)
    ax.set_xlabel('X Data', fontsize=8)
    ax.set_ylabel('Y Data', fontsize=8)
    ax.grid(True)
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=.3, hspace=.5); #To set the marginsplt.show() 
    plt.show()
    
    
    # Read the data
    file = 'sea_level_data.csv'
    df   = pd.read_csv(file, skiprows=6)
    x    = df[df.year<2001].year
    y    = df[df.year<2001].sea_level_inches
    xn   = df[df.year>2000].year
    yn   = df[df.year>2000].sea_level_inches
    xm   = np.linspace(0, df.shape[0], df.shape[0])
    
    Sea_Level_Model = {'Zeroth Order': {'i'    : 0,
                                        'ym'   : model(xm, a0=6.5, a1=0, a2=0),
                                        'color': 'red',
                                        'label': 'Model: a₀=6.5, a₁=0, a₂=0'},
                       'First Order':  {'i'    : 1,
                                        'ym'    : model(xm, a0=5, a1=0.08, a2=0),
                                        'color': 'limegreen',
                                        'label': 'Model: a₀=5, a₁=0.08, a₂=0'},
                       'Higher Order': {'i'    : 2,
                                        'ym'   : model(xm, a0=5, a1=0, a2=0.0025),
                                        'color': 'blue',
                                        'label': 'Model: a₀=5, a₁=0, a₂=0.0025'}} 
    
    # Plotting the model 
    fig, axis = plt.subplots(2, 2, figsize=(10.5, 5.5))
    fig.suptitle("{}\nTaylor Series: Y = a₀ + a₁X + a₂X² + ...".format(topic), fontsize=17, color='darkblue', weight='bold')
    
    for name, data in Sea_Level_Model.items():
        ax = axis[0 if data['i']<2 else 1, data['i'] if data['i']<2 else data['i']-2]
        ax.set_xlim(1970, 2013); ax.set_ylim(0, 15);
        ax.set_title(name, fontsize=12, color='maroon')  
        ax.plot(x, y, linestyle=" ", ms=3, marker="o" , color="black", label='Measured data')
        ax.plot(df.year, data['ym'], lw=2, color=data['color'], label=data['label'])
        ax.tick_params(labelsize=6)
        ax.legend(loc='best', fontsize=6)
        ax.set_xlabel('Time Elapse [years]', fontsize=8)
        ax.set_ylabel('Sea Level Change [inches]', fontsize=8)
        ax.grid(True)
        
        ax = axis[1, 1]
        ax.plot(df.year, data['ym'], lw=2, color=data['color'], label=data['label'])
   
    #ax = axis[1,1]
    ax.plot(x, y, linestyle=" ", ms=3, marker="o" , color="black", label='Measured data')
    ax.plot(xn, yn, linestyle=" ", ms=3, marker="o" , color="gold", label='New data')
    ax.set_xlim(1970, 2013); ax.set_ylim(0, 15);
    ax.set_title('Combining all Terms', fontsize=12, color='maroon')  
    ax.tick_params(labelsize=6)
    ax.legend(loc='best', fontsize=6)
    ax.set_xlabel('X Data', fontsize=8)
    ax.set_ylabel('Y Data', fontsize=8)
    ax.grid(True)
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=.3, hspace=.5); #To set the marginsplt.show() 
    plt.show()
    
    
def Terms_in_a_Model():
    print("****************************************************")
    topic = "2. Terms in a Model"; print("** %s\n" % topic)
    
    # Reviewing the data
    x = np.array([  0. ,   0.5,   1. ,   1.5,   2. ,   2.5,   3. ,   3.5,   4. ,
                    4.5,   5. ,   5.5,   6. ,   6.5,   7. ,   7.5,   8. ,   8.5,
                    9. ,   9.5,  10. ])
    y = np.array([ 111.78587909,  107.72560763,  210.81767421,  204.6837026 ,
                   231.98528167,  309.67907351,  346.48971034,  346.58691239,
                   400.3924093 ,  381.43287615,  503.75089312,  537.29865056,
                   573.8331032 ,  536.9686295 ,  616.64806585,  662.55295912,
                   710.13633529,  744.72729852,  808.0289548 ,  773.82736117,
                   844.93384188])
    #Model = ['Y = 250','Y = 100 + 200X', 'Y = ']
    #Preparing data for the model
    xm = np.linspace(-10, 15, 1000)
    
    fig, axis = plt.subplots(2, 3, figsize=(11.5, 5.5)) 
    fig.suptitle("{} - Hiking Trip Example".format(topic), fontsize=17, color='darkblue', weight='bold')
    
    for i in range(5):
        ax = axis[0 if i<3 else 1, i if i<3 else i-3]
        if   i==0:
            ym = np.repeat(250, 1000)
            ax.set_title('Model Y=250', fontsize=12, color='maroon')  
        elif i==1:
            ym = 100 + 200*xm
            ax.set_title('Model Y = 100 + 200X', fontsize=12, color='maroon')  
        elif i==2:
            ym = 200 + 75*xm
            ax.set_title('Model Y = 200 + 75X', fontsize=12, color='maroon')  
        elif i==3:
            ym = 100 + 75*xm
            ax.set_title('Model Y = 100 + 75X', fontsize=12, color='maroon')  
        elif i==4:
            ym = 100 + 50*xm + 3.25*(xm**2)
            ax.set_title('Model Y = 100 + 50X + 3.25X²', fontsize=12, color='maroon')  
            
        ax.tick_params(labelsize=6)
        ax.plot( x,  y, linestyle=" ", ms=3, marker="o" , color="black", label='Measured data')
        ax.plot(xm, ym, linestyle="-", lw=1, marker=None, color="red"  , label="Modeled data")
        ax.legend(fontsize=8, loc="upper left")
        ax.set_xlim(-10, 15)
        ax.set_ylim(-250, 1000)
        ax.axvline(0, lw=1, color='#343434')
        ax.axhline(0, lw=1, color='#343434')
        ax.set_xlabel('Step distance [km]')
        ax.set_ylabel('Altitude [meters]')
        ax.grid(True, color='#bfbfbf')
        
    axis[1,2].axis('off')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=.5, hspace=.5); #To set the marginsplt.show() 
    plt.show()
        
    
    
def Model_Components():
    print("****************************************************")
    topic = "3. Model Components"; print("** %s\n" % topic)
    
    # Generate array x, then predict y values for specific, non-default a0 and a1
    x = np.linspace(-10, 10, 21)
    y = model(x)
    
    # Plot the results, y versus x
    fig, ax = plt.subplots() 
    fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
    
    ax.plot( x,  y, "r-o")
    ax.axvline(0, lw=1, color='#343434')
    ax.axhline(0, lw=1, color='#343434')
    ax.set_xlabel('X Data')
    ax.set_ylabel('Y Data')
    ax.grid(True, color='#bfbfbf')
        
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=None);  
    plt.show()
    
    
    
    
def Model_Parameters():
    print("****************************************************")
    topic = "4. Model Parameters"; print("** %s\n" % topic)
    
    xd = np.array([  0. ,   0.5,   1. ,   1.5,   2. ,   2.5,   3. ,   3.5,   4. ,
                     4.5,   5. ,   5.5,   6. ,   6.5,   7. ,   7.5,   8. ,   8.5,
                     9. ,   9.5,  10. ])
    yd = np.array([ 161.78587909,  132.72560763,  210.81767421,  179.6837026 ,
                    181.98528167,  234.67907351,  246.48971034,  221.58691239,
                    250.3924093 ,  206.43287615,  303.75089312,  312.29865056,
                    323.8331032 ,  261.9686295 ,  316.64806585,  337.55295912,
                    360.13633529,  369.72729852,  408.0289548 ,  348.82736117,
                    394.93384188])
    
    # Complete the plotting function definition
    def plot_data_with_model(xd, yd, ym, slope, intercept, topic=topic):
        fig, ax = plt.subplots()
        fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
        
        ax.set_xlim(-5, 15); ax.set_ylim(-250, 750); 
        ax.set_xticks(np.linspace(-5,15,5)); ax.set_yticks(np.linspace(-250, 750, 5))
        ax.axvline(0, lw=1, color='#343434'); ax.axhline(0, lw=1, color='#343434')
        ax.grid(b=True, which='major')
        ax.grid(b=True, which='minor', alpha=.4, color='#bfbfbf')
        ax.minorticks_on()
        
        ax.plot(xd, yd, linestyle=" ", marker="o", color="black", label="Measured data")
        ax.plot(xd, ym, linestyle="-", marker=None, color="red", label="Modeled data")
        
        ax.legend(loc="best")
        ax.set_xlabel('Step distance [km]')
        ax.set_ylabel('Altitude [meters]')
        ax.set_title('Hiking Trip Example\nSlope: {}, Intercept: {}'.format(slope, intercept), fontsize=14, color='maroon')  
        
        plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the marginsplt.show() 
        plt.show()
        
    # Select new model parameters a0, a1, and generate modeled `ym` from them.
    a0 = yd[0]
    a1 = (yd[-1]-yd[0])/(xd[-1]-xd[0])
    ym = model(xd, a0, a1)
    
    
    # Plot the resulting model to see whether it fits the data
    plot_data_with_model(xd, yd, ym, a1, a0)
    
    
    
def Linear_Proportionality():
    print("****************************************************")
    topic = "6. Linear Proportionality"; print("** %s\n" % topic)
    
    # Complete the function to convert C to F
    def convert_scale(temps_C):
        (freeze_C, boil_C) = (0, 100)
        (freeze_F, boil_F) = (32, 212)
        change_in_C = boil_C - freeze_C
        change_in_F = boil_F - freeze_F
        slope = change_in_F / change_in_C
        intercept = freeze_F - freeze_C
        temps_F = intercept + (slope * temps_C)
        return temps_F, slope, intercept, freeze_C, boil_C, freeze_F, boil_F
    
    # Use the convert function to compute values of F and plot them
    temps_C = np.linspace(0, 100, 101)
    temps_F, slope, intercept, freeze_C, boil_C, freeze_F, boil_F = convert_scale(temps_C)
    (avg_human_temp_C, avg_human_temp_F) = (37, 98.6)
    
    # Plot the resulting model
    fig, ax = plt.subplots()
    fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
    
    ax.set_xlim(-50, 150); ax.set_ylim(-50, 300); 
    ax.axvline(0, lw=0.5, color='sienna'); ax.axhline(0, lw=0.5, color='sienna')
    ax.grid(True, color='gainsboro')
    ax.axhline(freeze_F, lw=.5, color='lightseagreen'); ax.axhline(boil_F, lw=.5, color='lightseagreen');
    
    ax.plot(temps_C, temps_F, 'b-', label='Model')
    ax.plot(freeze_C, freeze_F, 'rs', ms=6, label='Water melts')
    ax.plot(boil_C, boil_F, 'ro', ms=6, label='Water boils')
    ax.vlines(freeze_C, freeze_F, boil_F, ls=':', lw=2, color='black'); 
    ax.hlines(boil_F, freeze_C, boil_C, ls=':', lw=2, color='black'); 
    ax.vlines(boil_C, freeze_F, boil_F, ls='--', lw=1, color='black'); 
    ax.hlines(freeze_F, freeze_C, boil_C, ls='--', lw=1, color='black'); 
    ax.plot(avg_human_temp_C, avg_human_temp_F, 'ko', ms=5, label='Avg Human Temp')
    
    ax.legend(loc='best', fontsize=8)
    ax.set_xlabel('Temperature [degrees C]')
    ax.set_ylabel('Temperature [degrees F]')
    ax.set_title('Conversion between the Fahrenheit and Celsius\nSlope: {}, Intercept: {}'.format(slope, intercept), fontsize=14, color='maroon')  
        
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the marginsplt.show() 
    plt.show()
   
    
    
    
    
def Slope_and_Rates_of_Change():
    print("****************************************************")
    topic = "7. Slope and Rates-of-Change"; print("** %s\n" % topic)
    
    times     = np.array([ 0.        ,  0.08333333,  0.16666667,  0.25      ,  0.33333333,
                           0.41666667,  0.5       ,  0.58333333,  0.66666667,  0.75      ,
                           0.83333333,  0.91666667,  1.        ,  1.08333333,  1.16666667,
                           1.25      ,  1.33333333,  1.41666667,  1.5       ,  1.58333333,
                           1.66666667,  1.75      ,  1.83333333,  1.91666667,  2.        ])
    distances = np.array([   0.13536211,    4.11568697,    8.28931902,   12.41058595,
                            16.73878397,   20.64153844,   25.14540098,   29.10323276,
                            33.35991992,   37.47921914,   41.78850899,   45.66165494,
                            49.9731319 ,   54.13466214,   58.42781412,   62.40834239,
                            66.65229765,   70.76017847,   75.00351781,   79.2152346 ,
                            83.24161507,   87.59539364,   91.74179923,   95.87520786,
                           100.07507133])

    # Compute an array of velocities as the slope between each point
    diff_distances = np.diff(distances)
    diff_times = np.diff(times)
    velocities = diff_distances / diff_times

    # Chracterize the center and spread of the velocities
    v_avg = np.mean(velocities)
    v_max = np.max(velocities)
    v_min = np.min(velocities)
    v_range = v_max - v_min

    # Plot the distribution of velocities
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(11, 4))
    fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
    
    ax = ax1
    slope, intercept, r_value, p_value, std_err = linregress(times, distances)
    ax.set_xlim(-1, 2.25); ax.set_ylim(-10, 110); 
    ax.axvline(0, lw=0.5, color='sienna'); ax.axhline(0, lw=0.5, color='sienna')
    ax.grid(True, color='gainsboro')
    ax.plot(times, distances, linestyle=" ", marker="o", color="black", label="Measured data")
    ax.set_xlabel('Elapsed Time [Hours]')
    ax.set_ylabel('Travel Distance [Kilometers]')
    ax.set_title('Driven Trip\nSlope: {:.4f}, Intercept: {:.4f}'.format(slope, intercept), fontsize=12, color='maroon')  
        
    ax = ax2
    ax.set_xlim(-1, 2.25); ax.set_ylim(-10, 110); 
    ax.axvline(0, lw=0.5, color='sienna'); ax.axhline(0, lw=0.5, color='sienna')
    ax.grid(True, color='gainsboro')
    ax.plot(times[1:], velocities, linestyle=" ", marker="o", color="black", label="Measured data")
    ax.axhline(v_avg, lw=1, color='red')
    ax.set_xlabel('Time [Hours]')
    ax.set_ylabel('Instantaneous Velocity [Km/hr]')
    ax.set_title('The avarage and spread of the Velocity\nAverage: {:.4f}, Min: {:.4f}, Mac: {:.4f}, Range: {:.4f}'.format(v_avg, v_min, v_max, v_range), fontsize=12, color='maroon')  
        
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=.5, hspace=None); #To set the marginsplt.show() 
    plt.show()
   
    
def Intercept_and_Starting_Points():
    print("****************************************************")
    topic = "8. Intercept and Starting Points"; print("** %s\n" % topic)
    
    # Read the data
    file = 'solution_data.csv'
    df = pd.read_csv(file, skiprows=5)
    
    # First way
    slope, intercept, _, _, _ = linregress(df['volumes'], df['masses']) 
    print( "(I) USING linregress(x,y)\nIntercept: {}\nslope: {}\n".format(intercept, slope))
    
    #Second way
    m = LinearRegression()
    m.fit(df['volumes'].values.reshape(-1,1), df['masses'])
    slope = m.coef_[0]
    intercept = m.intercept_
    print( "(II) USING LinearRegression(x,y)\nIntercept: {}\nslope: {}\n".format(intercept, slope))
    
    
    #Third way
    model_fit = ols(formula="masses ~ volumes", data=df)
    model_fit = model_fit.fit()
    df['predicted masses'] = model_fit.predict(df['volumes']) 
    a0 = model_fit.params['Intercept'] # Extract the model parameter values, and assign them to a0, a1
    a1 = model_fit.params['volumes']
    
    xm = np.linspace(-2, 15, 1000)
    ym = model(xm, a0, a1)
    
    # Print model parameter values with meaningful names, and compare to summary()
    print( "(III) USING ols(formula=\"masses ~ volumes\", data=df)\nContainer_mass: {}\nSolution_density: {}\n\n".format(a0, a1))
    print( model_fit.summary() )
    
    # Plot the model
    fig, ax = plt.subplots()
    fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
    
    ax.set_xlim(-2, 15); ax.set_ylim(-5, 25); 
    ax.axvline(0, lw=2, color='sienna'); ax.axhline(0, lw=2, color='sienna')
    ax.grid(True, color='gainsboro')
    ax.plot(df['volumes'], df['masses'], linestyle=" ", ms=3, marker="o", color="black", label="Measured data")
    ax.plot(xm, ym, 'r-', lw=2, label="Modeled data")
    ax.legend(loc='best', fontsize=8)
    ax.set_xlabel('Volume of Solution [Liters]')
    ax.set_ylabel('Mass of Containner + Solution [Kilograms]')
    ax.set_title('Solution Density\nSlope: {:.4f}, Intercept: {:.4f}'.format(a1, a0), fontsize=12, color='maroon')  
        
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=.5, hspace=None); #To set the marginsplt.show() 
    plt.show()
   
    
    
def Model_Optimization():
    print("****************************************************")
    topic = "9. Model Optimization"; print("** %s\n" % topic)
    
    # Read the data
    file    = 'sea_level_data.csv'
    df      = pd.read_csv(file, skiprows=6)
    x_data  = np.linspace(0,df.shape[0]-1,df.shape[0])
    x_label = df.year
    y_data  = df.sea_level_inches.values
    
    slope, intercept, _, _, _ = linregress(x_data, df.sea_level_inches) 
    
    ###############################################################
    # ANALIZING THE FIRST MODEL APROX. 
    # Zeroth order: Y = a₀
    ###############################################################
    fig, axis = plt.subplots(2, 2, figsize=(10.5, 5.5))
    fig.suptitle("{}\nTalking about residuals - First Model".format(topic), fontsize=17, color='darkblue', weight='bold')
    
    # Plotting the model
    a0      = y_data.mean()
    y_model = model(x_data, a0=a0, a1=0, a2=0)
    
    ax = axis[0, 0]
    ax.set_title('Example Data', fontsize=12, color='maroon')  
    ax.plot(x_label, y_data, linestyle=" ", ms=3, marker="o" , color="black", label='Measured data')
    ax.plot(x_label, y_model, lw=2, color='red', label='Model: Y = {:.4f}'.format(a0))
    ax.tick_params(labelsize=6)
    ax.legend(loc='best', fontsize=6)
    ax.set_xlabel('Time Elapse [years]', fontsize=8)
    ax.set_ylabel('Sea Level Change [inches]', fontsize=8)
    ax.grid(True)
        
    
    # Plotting the residuals
    res       = y_data - y_model
    res_total =  res.sum()
    
    ax = axis[1, 0]
    ax.set_title('Residuals Summed = {:.4f}'.format(res_total), fontsize=12, color='maroon')  
    ax.plot(res, linestyle=" ", ms=3, marker="o" , color="green", label='Residuals')
    ax.axhline(res_total, lw=1.5, color='darkslategrey')
    ax.tick_params(labelsize=6)
    ax.legend(loc='best', fontsize=6)
    ax.set_xlabel('Index', fontsize=8)
    ax.set_ylabel('Residual', fontsize=8)
    ax.grid(True)
    
    
    # Plotting the squared residuals
    res_sq     = np.square(res)
    res_sq_tot = res_sq.sum()
    
    ax = axis[0, 1]
    ax.set_title('Residuals Squared Summed = {:.4f}'.format(res_sq_tot), fontsize=12, color='maroon')  
    ax.plot(res, linestyle=" ", ms=3, marker="o" , color="green", label='Residuals')
    ax.plot(res_sq, linestyle=" ", ms=3, marker="o" , color="blue", label='Residuals Squared')
    ax.axhline(res_total, lw=1.5, color='darkslategrey')
    ax.tick_params(labelsize=6)
    ax.legend(loc='best', fontsize=6)
    ax.set_xlabel('Index', fontsize=8)
    ax.set_ylabel('Residual Squared', fontsize=8)
    ax.grid(True)
    
    axis[1, 1].axis('off')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=.3, hspace=.5); #To set the marginsplt.show() 
    plt.show()
    
    
    ###############################################################
    # ANALIZING THE SECOND MODEL APROX. 
    # First order: Y = a₀ + a₁x
    ###############################################################
    fig, axis = plt.subplots(2, 2, figsize=(10.5, 5.5))
    fig.suptitle("{}\nTalking about residuals - Second Model".format(topic), fontsize=17, color='darkblue', weight='bold')
    
    # Plotting the model
    a0      = 5
    a1      = 0.08
    y_model = model(x_data, a0=a0, a1=a1, a2=0)
    
    ax = axis[0, 0]
    ax.set_title('Example Data', fontsize=12, color='maroon')  
    ax.plot(x_label, y_data, linestyle=" ", ms=3, marker="o" , color="black", label='Measured data')
    ax.plot(x_label, y_model, lw=2, color='red', label='Model: Y = {} + {}X'.format(a0, a1))
    ax.tick_params(labelsize=6)
    ax.legend(loc='best', fontsize=6)
    ax.set_xlabel('Time Elapse [years]', fontsize=8)
    ax.set_ylabel('Sea Level Change [inches]', fontsize=8)
    ax.grid(True)
        
    
    # Plotting the residuals
    res       = y_data - y_model
    res_total =  res.sum()
    
    ax = axis[1, 0]
    ax.set_title('Residuals Summed = {:.4f}'.format(res_total), fontsize=12, color='maroon')  
    ax.plot(res, linestyle=" ", ms=3, marker="o" , color="green", label='Residuals')
    ax.tick_params(labelsize=6)
    ax.legend(loc='best', fontsize=6)
    ax.set_xlabel('Index', fontsize=8)
    ax.set_ylabel('Residual', fontsize=8)
    ax.grid(True)
    
    
    # Plotting the squared residuals
    res_sq     = np.square(res)
    res_sq_tot = res_sq.sum()
    
    ax = axis[0, 1]
    ax.set_title('Residuals Squared Summed = {:.4f}'.format(res_sq_tot), fontsize=12, color='maroon')  
    ax.plot(res, linestyle=" ", ms=3, marker="o" , color="green", label='Residuals')
    ax.plot(res_sq, linestyle=" ", ms=3, marker="o" , color="blue", label='Residuals Squared')
    ax.tick_params(labelsize=6)
    ax.legend(loc='best', fontsize=6)
    ax.set_xlabel('Index', fontsize=8)
    ax.set_ylabel('Residual Squared', fontsize=8)
    ax.grid(True)
    
    axis[1, 1].axis('off')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=.3, hspace=.5); #To set the marginsplt.show() 
    plt.show()
    
    
    
    ###############################################################
    # UNDERSTANDING THE RSS. 
    # Third order: Y = a₀ + a₁x
    ###############################################################
    fig, axis = plt.subplots(2, 2, figsize=(10.5, 5.5))
    fig.suptitle("{}\nTalking about residuals - Third Model".format(topic), fontsize=17, color='darkblue', weight='bold')
    
    # Plotting the model
    a0      = intercept
    a1      = slope
    y_model = model(x_data, a0=a0, a1=a1, a2=0)
    
    ax = axis[0, 0]
    ax.set_title('Example Data', fontsize=12, color='maroon')  
    ax.plot(x_label, y_data, linestyle=" ", ms=3, marker="o" , color="black", label='Measured data')
    ax.plot(x_label, y_model, lw=2, color='red', label='Model: Y = {:.4f} + {:.4f}X'.format(a0, a1))
    ax.tick_params(labelsize=6)
    ax.legend(loc='best', fontsize=6)
    ax.set_xlabel('Time Elapse [years]', fontsize=8)
    ax.set_ylabel('Sea Level Change [inches]', fontsize=8)
    ax.grid(True)
        
    
    # Plotting the residuals
    res       = y_data - y_model
    res_total =  res.sum()
    
    ax = axis[1, 0]
    ax.set_title('Residuals Summed = {:.4f}'.format(res_total), fontsize=12, color='maroon')  
    ax.plot(res, linestyle=" ", ms=3, marker="o" , color="green", label='Residuals')
    ax.tick_params(labelsize=6)
    ax.legend(loc='best', fontsize=6)
    ax.set_xlabel('Index', fontsize=8)
    ax.set_ylabel('Residual', fontsize=8)
    ax.grid(True)
    
    
    # Plotting the squared residuals
    res_sq     = np.square(res)
    res_sq_tot = res_sq.sum()
    
    ax = axis[0, 1]
    ax.set_title('Residuals Squared Summed = {:.4f}'.format(res_sq_tot), fontsize=12, color='maroon')  
    ax.plot(res, linestyle=" ", ms=3, marker="o" , color="green", label='Residuals')
    ax.plot(res_sq, linestyle=" ", ms=3, marker="o" , color="blue", label='Residuals Squared')
    ax.tick_params(labelsize=6)
    ax.legend(loc='best', fontsize=6)
    ax.set_xlabel('Index', fontsize=8)
    ax.set_ylabel('Residual Squared', fontsize=8)
    ax.grid(True)
    
    # Reviwing the squared residuals summed behavior
    variation = 0.05
    a_1 = np.linspace(slope-variation, slope+variation, 1001)
    res_sq_tot = []
    for a1 in a_1:
        y_model    = model(x_data, a0=a0, a1=a1, a2=0)
        res        = y_data - y_model
        res_sq     = np.square(res)
        res_sq_tot = np.append(res_sq_tot, res_sq.sum())
    
    ax = axis[1, 1]
    ax.set_title('Residuals Behavior, a₁ varies from {:.4f} to {:.4f}'.format(slope-variation, slope+variation), fontsize=12, color='maroon')  
    ax.plot(a_1, res_sq_tot, lw=1.5, color='purple', label='RSS')
    ax.axvline(slope, lw=1.5, color='darkgoldenrod', label='best option, a₁ = {:.4f}'.format(slope))
    ax.tick_params(labelsize=6)
    ax.legend(loc='best', fontsize=6)
    ax.set_xlabel('Slope a₁', fontsize=8)
    ax.set_ylabel('RSS', fontsize=8)
    ax.grid(True)
    
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=.3, hspace=.5); #To set the marginsplt.show() 
    plt.show()
    
    
    
def Residual_Sum_of_the_Squares():
    print("****************************************************")
    topic = "10. Residual Sum of the Squares"; print("** %s\n" % topic)
    
    def plot_data_with_model(xd, yd, ym, a0, a1, rss, slope, intercept, topic=topic):
        fig, ax = plt.subplots()
        fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
        
        ax.set_xlim(-5, 15); ax.set_ylim(-250, 750); 
        ax.set_xticks(np.linspace(-5,15,5)); ax.set_yticks(np.linspace(-250, 750, 5))
        ax.axvline(0, lw=1, color='#343434'); ax.axhline(0, lw=1, color='#343434')
        ax.grid(b=True, which='major')
        ax.grid(b=True, which='minor', alpha=.4, color='#bfbfbf')
        ax.minorticks_on()
        
        ax.plot(xd, yd, linestyle=" ", marker="o", color="black", label="Measured data")
        ax.plot(xd, ym, linestyle="-", marker=None, color="red", label="Modeled data")
        
        ax.legend(loc="best")
        ax.set_xlabel('Step distance [km]')
        ax.set_ylabel('Altitude [meters]')
        ax.set_title('Hiking Trip Example\nModel: Y = {:.4f} + {:.4f}X\nRSS: {}\n(Best Intercept: {:.4f}, Best Slope: {:.4f})'.format(a0, a1, rss, intercept, slope), fontsize=14, color='maroon')  
        
        plt.subplots_adjust(left=None, bottom=None, right=None, top=.72, wspace=None, hspace=None); #To set the marginsplt.show() 
        plt.show()
    
    x_data = np.array([  0. ,   0.5,   1. ,   1.5,   2. ,   2.5,   3. ,   3.5,   4. ,
                         4.5,   5. ,   5.5,   6. ,   6.5,   7. ,   7.5,   8. ,   8.5,
                         9. ,   9.5,  10. ])
    y_data = np.array([ 161.78587909,  132.72560763,  210.81767421,  179.6837026 ,
                        181.98528167,  234.67907351,  246.48971034,  221.58691239,
                        250.3924093 ,  206.43287615,  303.75089312,  312.29865056,
                        323.8331032 ,  261.9686295 ,  316.64806585,  337.55295912,
                        360.13633529,  369.72729852,  408.0289548 ,  348.82736117,
                        394.93384188])
    slope, intercept, _, _, _ = linregress(x_data, y_data)
    a0=150; a1=25;
    params = dict(a0=150, a1=25)
    
    # Model the data with specified values for parameters a0, a1
    y_model = model(x_data, **params)

    # Compute the RSS value for this parameterization of the model
    rss = np.sum(np.square(y_data - y_model))
    print("RSS = {}".format(rss))
    
    plot_data_with_model(x_data, y_data, y_model, a0, a1, rss, slope, intercept)
    
    
    
def Visualizing_the_RSS_Minima():
    print("****************************************************")
    topic = "12. Visualizing the RSS Minima"; print("** %s\n" % topic)
    
    x_data = np.array([  0. ,   0.5,   1. ,   1.5,   2. ,   2.5,   3. ,   3.5,   4. ,
                         4.5,   5. ,   5.5,   6. ,   6.5,   7. ,   7.5,   8. ,   8.5,
                         9. ,   9.5,  10. ])
    y_data = np.array([ 161.78587909,  132.72560763,  210.81767421,  179.6837026 ,
                        181.98528167,  234.67907351,  246.48971034,  221.58691239,
                        250.3924093 ,  206.43287615,  303.75089312,  312.29865056,
                        323.8331032 ,  261.9686295 ,  316.64806585,  337.55295912,
                        360.13633529,  369.72729852,  408.0289548 ,  348.82736117,
                        394.93384188])
    slope, intercept, _, _, _ = linregress(x_data, y_data) 
    print("slope= {}, intercept={}".format(slope, intercept))
    
    fig, axis = plt.subplots(2, 2, figsize=(10.5, 5.5))
    fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
    
    # Plotting the model
    a0      = intercept
    a1      = slope
    y_model = model(x_data, a0=a0, a1=a1, a2=0)
    
    ax = axis[0, 0]
    ax.set_title('Hiking Trip Example', fontsize=12, color='maroon')  
    ax.plot(x_data, y_data, linestyle=" ", ms=3, marker="o" , color="black", label='Measured data')
    ax.plot(x_data, y_model, lw=2, color='red', label='Model: Y = {:.4f} + {:.4f} X'.format(a0, a1))
    ax.tick_params(labelsize=6)
    ax.legend(loc='best', fontsize=6)
    ax.set_xlabel('Step distance [km]', fontsize=8)
    ax.set_ylabel('Altitude [meters]', fontsize=8)
    ax.grid(True)
        
    
    # Plotting the residuals
    res       = y_data - y_model
    res_total =  res.sum()
    
    ax = axis[1, 0]
    ax.set_title('Residuals Summed = {:.4f}'.format(res_total), fontsize=12, color='maroon')  
    ax.plot(res, linestyle=" ", ms=3, marker="o" , color="green", label='Residuals')
    ax.tick_params(labelsize=6)
    ax.legend(loc='best', fontsize=6)
    ax.set_xlabel('Index', fontsize=8)
    ax.set_ylabel('Residual', fontsize=8)
    ax.grid(True)
    
    
    # Plotting the squared residuals
    res_sq     = np.square(res)
    res_sq_tot = res_sq.sum()
    
    ax = axis[0, 1]
    ax.set_title('Residuals Squared Summed = {:.4f}'.format(res_sq_tot), fontsize=12, color='maroon')  
    ax.plot(res, linestyle=" ", ms=3, marker="o" , color="green", label='Residuals')
    ax.plot(res_sq, linestyle=" ", ms=3, marker="o" , color="blue", label='Residuals Squared')
    ax.tick_params(labelsize=6)
    ax.legend(loc='best', fontsize=6)
    ax.set_xlabel('Index', fontsize=8)
    ax.set_ylabel('Residual Squared', fontsize=8)
    ax.grid(True)
    
    # Reviwing the squared residuals summed behavior
    variation = 10
    a_1 = np.linspace(slope-variation, slope+variation, 1001)
    res_sq_tot = []
    for a1 in a_1:
        y_model    = model(x_data, a0=a0, a1=a1, a2=0)
        res        = y_data - y_model
        res_sq     = np.square(res)
        res_sq_tot = np.append(res_sq_tot, res_sq.sum())
    best_rss = res_sq_tot.min() 
    best_a1 = a_1[np.where(res_sq_tot==best_rss)][0]
    
    ax = axis[1, 1]
    ax.set_title('Residuals Behavior, a₁ varies from {:.4f} to {:.4f}'.format(slope-variation, slope+variation), fontsize=12, color='maroon')  
    ax.plot(a_1, res_sq_tot, lw=1.5, color='purple', label='RSS')
    ax.plot(best_a1, best_rss, marker='o', ms=6, color='darkgoldenrod', label='Minimum RSS = {:.4f}, came from a₁ = {:.4f}'.format(best_rss, best_a1))
    ax.tick_params(labelsize=6)
    ax.legend(loc='best', fontsize=6)
    ax.set_xlabel('Slope a₁', fontsize=8)
    ax.set_ylabel('RSS', fontsize=8)
    ax.grid(True)
    
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=.3, hspace=.5); #To set the marginsplt.show() 
    plt.show()
    
    
    
def Least_Squares_Optimization():
    print("****************************************************")
    topic = "13. Least-Squares Optimization"; print("** %s\n" % topic)
    
    x = np.array([  0. ,   0.5,   1. ,   1.5,   2. ,   2.5,   3. ,   3.5,   4. ,
                    4.5,   5. ,   5.5,   6. ,   6.5,   7. ,   7.5,   8. ,   8.5,
                    9. ,   9.5,  10. ])
    y = np.array([ 161.78587909,  132.72560763,  210.81767421,  179.6837026 ,
                   181.98528167,  234.67907351,  246.48971034,  221.58691239,
                   250.3924093 ,  206.43287615,  303.75089312,  312.29865056,
                   323.8331032 ,  261.9686295 ,  316.64806585,  337.55295912,
                   360.13633529,  369.72729852,  408.0289548 ,  348.82736117,
                   394.93384188])
    
    ###############################################################
    # 1st method to get the slope and intercept - numpy
    ###############################################################
    a1 = np.cov(x, y, bias=True)[0, 1] / x.var() #Slope, use bias to normalize the series.
    a0 = y.mean() - a1*x.mean() #Intercept
    print("1st method: numpy                                : a0 = {:.4f}, a1 = {:.4f}".format(a0, a1))
    
    
    ###############################################################
    # 2nd method to get the slope and intercept - numpy manual
    ###############################################################
    dx = x - x.mean()
    dy = y - y.mean()
    
    a1 = np.sum( dx * dy ) / np.sum( dx**2 )
    a0 = y.mean() - (a1*x.mean())
    print("2nd method: numpy in a manual way                : a0 = {:.4f}, a1 = {:.4f}".format(a0, a1))
    
    
    ###############################################################
    # 3rd method to get the slope and intercept - scipy.optimize
    ###############################################################
    def model_func(x, a0, a1):
        return a0 + (a1*x)
    
    param_opt, param_cov = optimize.curve_fit(model_func, x, y)
    a0 = param_opt[0] # a0 is the intercept in y = a0 + a1*x
    a1 = param_opt[1]
    print("3rd method: scipy.optimize                       : a0 = {:.4f}, a1 = {:.4f}".format(a0, a1))
    
    
    ###############################################################
    # 4th method to get the slope and intercept - statsmodels.formula.api.ols
    ###############################################################
    df = pd.DataFrame(dict(x_name=x, y_name=y))
    model_fit = ols(formula="y_name ~ x_name", data=df).fit()
    
    a0 = model_fit.params['Intercept']
    a1 = model_fit.params['x_name']
    print("4th method: statsmodels.formula.api.ols          : a0 = {:.4f}, a1 = {:.4f}".format(a0, a1))
    
    ###############################################################
    # 5th method to get the slope and intercept - scipy.stats.linregress
    ###############################################################
    a1, a0, _, _, _ = linregress(x, y)
    print("5th method: scipy.stats.linregress               : a0 = {:.4f}, a1 = {:.4f}".format(a0, a1))
    
    
    ###############################################################
    # 6th method to get the slope and intercept - sklearn.linear_model.LinearRegression
    ###############################################################
    m = LinearRegression().fit(x.reshape(-1,1), y)
    a1 = m.coef_[0]
    a0 = m.intercept_
    print("6th method: sklearn.linear_model.LinearRegression: a0 = {:.4f}, a1 = {:.4f}".format(a0, a1))
    
    
    ###############################################################
    # 7th method to get the slope and intercept - np.polyfit
    ###############################################################
    a1, a0 = np.polyfit(x, y, 1)
    print("7th method: np.polyfit: a0 = {:.4f}, a1 = {:.4f}".format(a0, a1))
    
    
def Least_Squares_with_numpy():
    print("****************************************************")
    topic = "14. Least-Squares with `numpy`"; print("** %s\n" % topic)
    
    def plot_rss(x, y, a0, a1, variation=10, topic=topic):
        """
        Plot the graph of changes in RSS
        Parameters
        ----------
        x : numpy array. Serie X.
        y : numpy array. Serue Y.
        a0 : float. Interception of the lineal relation between x and y.
        a1 : float. Slope of the lineal relation between x and y.
        variation : float. Value to set the range of the slope from a1-variation to a1+variation.
                    The default is 10.
        topic : String, Suptitle of the graph.
        """
                
        fig, axis = plt.subplots(1, 2, figsize=(11.5, 4))
        fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
        
        # Plotting the model
        y_model = model(x, a0, a1)
        ax = axis[0]
        ax.set_title('The model', fontsize=12, color='maroon')  
        ax.plot(x, y, linestyle=" ", ms=3, marker="o" , color="black", label='Measured data')
        ax.plot(x, y_model, lw=2, color='red', label='Model: Y = {:.4f} + {:.4f} X'.format(a0, a1))
        ax.tick_params(labelsize=6)
        ax.legend(loc='best', fontsize=6)
        ax.set_xlabel('X Data', fontsize=8)
        ax.set_ylabel('Y Data', fontsize=8)
        ax.grid(True)
        
        # Reviwing the squared residuals summed behavior
        a_1 = np.linspace(a1-variation, a1+variation, 1001)
        res_sq_tot = []
        for a1_i in a_1:
            y_model    = model(x, a0, a1_i)
            res_sq_tot = np.append(res_sq_tot, np.square(y - y_model).sum())
        best_rss = res_sq_tot.min() 
        best_a1 = a_1[np.where(res_sq_tot==best_rss)][0]
        
        ax = axis[1]
        ax.set_title('Residuals Behavior, a₁ varies from {:.4f} to {:.4f}'.format(a_1[0], a_1[-1]), fontsize=12, color='maroon')  
        ax.plot(a_1, res_sq_tot, lw=1.5, color='purple', label='RSS')
        ax.plot(best_a1, best_rss, marker='o', ms=6, color='darkgoldenrod', label='Minimum RSS = {:.4f}, came from a₁ = {:.4f}'.format(best_rss, best_a1))
        ax.tick_params(labelsize=6)
        ax.legend(loc='best', fontsize=6)
        ax.set_xlabel('Slope a₁', fontsize=8)
        ax.set_ylabel('RSS', fontsize=8)
        ax.grid(True)
                
        plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=None); #To set the marginsplt.show() 
        plt.show()
    
    # The series data
    x = np.array([  0. ,   0.5,   1. ,   1.5,   2. ,   2.5,   3. ,   3.5,   4. ,
                    4.5,   5. ,   5.5,   6. ,   6.5,   7. ,   7.5,   8. ,   8.5,
                    9. ,   9.5,  10. ])
    y = np.array([ 161.78587909,  132.72560763,  210.81767421,  179.6837026 ,
                   181.98528167,  234.67907351,  246.48971034,  221.58691239,
                   250.3924093 ,  206.43287615,  303.75089312,  312.29865056,
                   323.8331032 ,  261.9686295 ,  316.64806585,  337.55295912,
                   360.13633529,  369.72729852,  408.0289548 ,  348.82736117,
                   394.93384188])
    
    
    # prepare the means and deviations of the two variables
    dx = x - x.mean()
    dy = y - y.mean()

    # Complete least-squares formulae to find the optimal a0, a1
    a1 = np.sum(dx * dy) / np.sum( np.square(dx) )
    a0 = y.mean() - (a1 * x.mean())
    
    # Plot the model and the behavior of the RSS
    plot_rss(x, y, a0, a1)
    
    
    
def Optimization_with_Scipy():
    print("****************************************************")
    topic = "15. Optimization with Scipy"; print("** %s\n" % topic)
    
    
    def plot_model_and_rss(x, y, a0, a1, topic=topic):
        """
        Plot the model predicted for relation between X and Y, detecting RSS.
        Parameters
        ----------
        x : numpy array. Serie X.
        y : numpy array. Serue Y.
        a0 : float. Interception of the lineal relation between x and y.
        a1 : float. Slope of the lineal relation between x and y.
        topic : String, Suptitle of the graph.
        """
                
        y_model = model(x, a0, a1)
        rss = np.square(y - y_model).sum()
        
        # Plotting the model
        fig, ax = plt.subplots()
        fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
        ax.set_title('Minimum RSS = {:.4f}, came from a₁ = {:.4f}'.format(rss, a1), fontsize=12, color='maroon')  
        ax.plot(x, y, linestyle=" ", ms=3, marker="o" , color="black", label='Measured data')
        ax.plot(x, y_model, lw=2, color='red', label='Model: Y = {:.4f} + {:.4f} X'.format(a0, a1))
        ax.tick_params(labelsize=6)
        ax.legend(loc='best', fontsize=6)
        ax.set_xlabel('X Data', fontsize=8)
        ax.set_ylabel('Y Data', fontsize=8)
        ax.grid(True)
        
        plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=None); #To set the marginsplt.show() 
        plt.show()


    # Define a model function needed as input to scipy
    def model_func(x, a0, a1):
        return a0 + (a1*x)

    # The series data
    x = np.array([  0. ,   0.5,   1. ,   1.5,   2. ,   2.5,   3. ,   3.5,   4. ,
                    4.5,   5. ,   5.5,   6. ,   6.5,   7. ,   7.5,   8. ,   8.5,
                    9. ,   9.5,  10. ])
    y = np.array([ 161.78587909,  132.72560763,  210.81767421,  179.6837026 ,
                   181.98528167,  234.67907351,  246.48971034,  221.58691239,
                   250.3924093 ,  206.43287615,  303.75089312,  312.29865056,
                   323.8331032 ,  261.9686295 ,  316.64806585,  337.55295912,
                   360.13633529,  369.72729852,  408.0289548 ,  348.82736117,
                   394.93384188])
    
    # call curve_fit, passing in the model function and data; then unpack the results
    param_opt, param_cov = optimize.curve_fit(model_func, x, y)
    a0 = param_opt[0]  # a0 is the intercept in y = a0 + a1*x
    a1 = param_opt[1]  # a1 is the slope     in y = a0 + a1*x
    
    # test that these parameters result in a model that fits the data
    plot_model_and_rss(x, y, a0, a1)
    
    
def Least_Squares_with_statsmodels():
    print("****************************************************")
    topic = "16. Least-Squares with `statsmodels`"; print("** %s\n" % topic)
    
    def plot_model_and_rss(x, y, y_model, a0, a1, topic=topic):
        """
        Plot the model predicted for relation between X and Y, detecting RSS.
        Parameters
        ----------
        x : numpy array. Serie X.
        y : numpy array. Serue Y.
        y_model : numpy array. Predicted y based in the model.
        a0 : float. Interception of the lineal relation between x and y.
        a1 : float. Slope of the lineal relation between x and y.
        topic : String, Suptitle of the graph.
        """
                
        rss = np.square(y - y_model).sum()
        
        # Plotting the model
        fig, ax = plt.subplots()
        fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
        ax.set_title('Minimum RSS = {:.4f}, came from a₁ = {:.4f}'.format(rss, a1), fontsize=12, color='maroon')  
        ax.plot(x, y, linestyle=" ", ms=3, marker="o" , color="black", label='Measured data')
        ax.plot(x, y_model, lw=2, color='red', label='Model: Y = {:.4f} + {:.4f} X'.format(a0, a1))
        ax.tick_params(labelsize=6)
        ax.legend(loc='best', fontsize=6)
        ax.set_xlabel('X Data', fontsize=8)
        ax.set_ylabel('Y Data', fontsize=8)
        ax.grid(True)
        
        plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=None); #To set the marginsplt.show() 
        plt.show()

    
    # The series data
    x = np.array([  0. ,   0.5,   1. ,   1.5,   2. ,   2.5,   3. ,   3.5,   4. ,
                    4.5,   5. ,   5.5,   6. ,   6.5,   7. ,   7.5,   8. ,   8.5,
                    9. ,   9.5,  10. ])
    y = np.array([ 161.78587909,  132.72560763,  210.81767421,  179.6837026 ,
                   181.98528167,  234.67907351,  246.48971034,  221.58691239,
                   250.3924093 ,  206.43287615,  303.75089312,  312.29865056,
                   323.8331032 ,  261.9686295 ,  316.64806585,  337.55295912,
                   360.13633529,  369.72729852,  408.0289548 ,  348.82736117,
                   394.93384188])
    df = pd.DataFrame(dict(x_column=x, y_column=y))
    
    # Pass data and `formula` into ols(), use and `.fit()` the model to the data
    model_fit = ols(formula="y_column ~ x_column", data=df).fit()
    
    # Use .predict(df) to get y_model values, then over-plot y_data with y_model
    y_model = model_fit.predict(df)

    # Extract the a0, a1 values from model_fit.params
    a0 = model_fit.params['Intercept']
    a1 = model_fit.params['x_column']

    # test that these parameters result in a model that fits the data
    plot_model_and_rss(x, y, y_model, a0, a1)
    
    
def main():
    print("****************************************************")
    print("** BEGIN                                          **")
    
    What_makes_a_model_linear()
    Terms_in_a_Model()
    Model_Components()
    Model_Parameters()
    Linear_Proportionality()
    Slope_and_Rates_of_Change()
    Intercept_and_Starting_Points()
    Model_Optimization()
    Residual_Sum_of_the_Squares()
    Visualizing_the_RSS_Minima()
    Least_Squares_Optimization()
    Least_Squares_with_numpy()
    Optimization_with_Scipy()
    Least_Squares_with_statsmodels()
        
    print("****************************************************")
    print("** END                                            **")
    print("****************************************************")

if __name__ == '__main__':
    main()