# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:36:32 2020

@author: jacqueline.cortez
Subject: Practicing Statistics Interview Questions in Python
Chapter 1: Exploring Linear Trends
    We start the course with an initial exploration of linear relationships, 
    including some motivating examples of how linear models are used, and 
    demonstrations of data visualization methods from matplotlib. We then use 
    descriptive statistics to quantify the shape of our data and use 
    correlation to quantify the strength of linear relationships between two 
    variables.
"""

###############################################################################
## Importing libraries
###############################################################################
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import linregress #Fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.
from sklearn.linear_model import LinearRegression #Calculate a linear least-squares regression for two sets of measurements. To get the parameters (slope and intercept) from a model


###############################################################################
## Preparing the environment
###############################################################################
SEED = 42
np.random.seed(SEED) 
    

###############################################################################
## Main part of the code
###############################################################################
def Reasons_for_Modeling_Interpolation():
    print("****************************************************")
    topic = "2. Reasons for Modeling: Interpolation"; print("** %s\n" % topic)
    
    # Data
    distances = np.array([ 0., 44.04512153, 107.16353484, 148.43674052, 196.39705633, 254.4358147 , 300. ])
    times     = np.array([ 0., 1., 2., 3., 4., 5., 6.])

    # Compute the total change in distance and change in time
    total_distance = distances[-1] - distances[0]
    total_time = times[-1] - times[0]
    
    # Estimate the slope of the data from the ratio of the changes
    average_speed = total_distance / total_time
    
    # Predict the distance traveled for a time not measured
    elapse_time = 2.5
    distance_traveled = average_speed * elapse_time
    msg = "Interpolation:\nThe distance traveled in {} hours is {} miles.".format(elapse_time, distance_traveled)
    print(msg)
    
    # Plotting the model 
    fig = plt.figure()
    fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
    ax = plt.axes(xlim=(times[0]-.35, times[-1]+1), ylim=(distances[0]-20, distances[-1]+50))
    plt.scatter(times, distances, s=30, color='black', label='data')
    plt.plot(times, average_speed*times, lw=2, color='red', label='model')
    
    plt.xlabel('X Data, Time Driven [hours]')
    plt.ylabel('Y Data, Distance Traveled [miles]')
    plt.title('"50 Miles per hour" model', fontsize=14, color='maroon')  
    
    plt.vlines(elapse_time, plt.ylim()[0], distance_traveled, ls='--', color='steelblue')
    plt.hlines(distance_traveled, plt.xlim()[0], elapse_time, ls='--', color='steelblue')
    plt.scatter(elapse_time, distance_traveled, s=30, label='interpolated data', color='steelblue')
    plt.text(.4, .1, msg, transform=ax.transAxes, backgroundcolor='white', fontsize=8, color='steelblue') 
    
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.show()


def Reasons_for_Modeling_Extrapolation():
    print("****************************************************")
    topic = "3. Reasons for Modeling: Extrapolation"; print("** %s\n" % topic)
    
    # Data
    distances = np.array([ 0., 44.04512153, 107.16353484, 148.43674052, 196.39705633, 254.4358147 , 300. ])
    times     = np.array([ 0., 1., 2., 3., 4., 5., 6.])
    
    # Get the model parameters
    slope, intercept, r_value, p_value, std_err = linregress(times, distances)
    
    # Compute the total change in distance and change in time
    total_distance = distances[-1] - distances[0]
    total_time = times[-1] - times[0]
    
    # Estimate the slope of the data from the ratio of the changes
    average_speed = total_distance / total_time
    
    def model(time):
        return (average_speed*time)   
    
    # Select a time not measured.
    elapse_time = 8
    
    # Use the model to compute a predicted distance for that time.
    distance_traveled = model(elapse_time)
    msg = "Extrapolation:\nThe distance traveled in {} hours is {} miles.".format(elapse_time, distance_traveled)
    print(msg)
    
    # Determine if you will make it without refueling.
    answer = (distance_traveled <= 400)
    print("The car can travel, at most, 400 miles on a full tank, and it takes 8 hours to drive home")
    print('If the car has the tank full, Can it return home without refueling?', answer) 
    
    
    # Plotting the model 
    fig = plt.figure()
    fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
    ax = plt.axes(xlim=(times[0]-.35, 10), ylim=(distances[0]-20, 500))
    plt.scatter(times, distances, s=30, color='black', label='data')
    #plt.plot(times, average_speed*times, lw=2, color='red', label='model')
    plt.plot(times, slope*times+intercept, lw=2, color='red', label='model')
    
    plt.xlabel('X Data, Time Driven [hours]')
    plt.ylabel('Y Data, Distance Traveled [miles]')
    plt.title('"50 Miles per hour" model', fontsize=14, color='maroon')  
    
    plt.vlines(elapse_time, plt.ylim()[0], distance_traveled, ls='--', color='steelblue')
    plt.hlines(distance_traveled, plt.xlim()[0], elapse_time, ls='--', color='steelblue')
    plt.scatter(elapse_time, distance_traveled, s=30, label='extrapolated data', color='steelblue')
    plt.text(.79, .1, msg, transform=ax.transAxes, backgroundcolor='white', fontsize=8, color='steelblue', ha='right') 
    
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.show()



def Reasons_for_Modeling_Estimating_Relationships():
    print("****************************************************")
    topic = "4. Reasons for Modeling:\nEstimating Relationships"; print("** %s\n" % topic)
    
    car1 = {'gallons': np.array([ 0.03333333, 1.69666667, 3.36, 5.02333333, 6.68666667, 8.35, 10.01333333, 11.67666667, 13.34, 15.00333333, 16.66666667]),
            'miles'  : np.array([1., 50.9, 100.8, 150.7, 200.6, 250.5, 300.4, 350.3, 400.2, 450.1, 500.])}
    car2 = {'gallons': np.array([ 0.02, 1.018, 2.016, 3.014, 4.012, 5.01, 6.008, 7.006, 8.004, 9.002, 10. ]),
            'miles'  : np.array([ 1., 50.9, 100.8, 150.7, 200.6, 250.5, 300.4, 350.3, 400.2, 450.1, 500. ])}
    
    # Complete the function to model the efficiency.
    def model(miles, gallons):
        return np.mean( miles / gallons )

    # Use the function to estimate the efficiency for each car.
    car1['mpg'] = model(car1['miles'] , car1['gallons'] )
    car2['mpg'] = model(car2['miles'] , car2['gallons'] )
    
    # Finish the logic statement to compare the car efficiencies.
    msg = 'Car1 is the best.' if car1['mpg'] > car2['mpg'] else ('Car2 is the best.' if car1['mpg'] < car2['mpg'] else 'The cars have the same efficiency')
    print(msg)
    
    # Get the model parameters
    slope_car1, intercept_car1, _, _, _ = linregress(car1['miles'], car1['gallons'])
    slope_car2, intercept_car2, _, _, _ = linregress(car2['miles'], car2['gallons'])
        
    
    # Plotting the model 
    fig = plt.figure()
    fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(car1['miles'], car1['gallons'], s=30, color='blue', label='Car 1, slope:{:,.4f}, intercept:{:,.4f}, mpg:{:,.4f}'.format(slope_car1, intercept_car1, car1['mpg']))
    plt.scatter(car2['miles'], car2['gallons'], s=30, color='red', label='Car 2, slope:{:,.4f}, intercept:{:,.4f}, mpg:{:,.4f}'.format(slope_car2, intercept_car2, car2['mpg']))
    
    plt.xlabel('X Data, Distance Traveled [miles]')
    plt.ylabel('Y Data, Fuel Consummed [gallons]')
    plt.title('"MPG Efficiency" model', fontsize=14, color='maroon')  
    
    plt.text(.9, .1, msg, transform=ax.transAxes, backgroundcolor='white', fontsize=8, color='steelblue', ha='right') 
    
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the marginsplt.show() 
    plt.show()


def Plotting_the_Data():
    print("****************************************************")
    topic = "6. Plotting the Data"; print("** %s\n" % topic)
    
    times     = np.array([  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.])
    distances = np.array([ 0.24835708,  0.93086785,  2.32384427,  3.76151493,  3.88292331,
                          4.88293152,  6.78960641,  7.38371736,  7.76526281,  9.27128002,
                          9.76829115])
    
    # Create figure and axis objects using subplots()
    fig, axis = plt.subplots() 
    fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
    
    # Plot line using the axis.plot() method
    options = dict(linestyle=" ", marker="o", color="red")
    axis.plot(times , distances , **options)
    
    # Customizing the plot
    plt.xlabel('X Data')
    plt.ylabel('Y Data')
    plt.title('Object Interface Plot Method', fontsize=14, color='maroon')  
    plt.grid(True)
    
    # Use the plt.show() method to display the figure
    plt.show()


    
def Plotting_the_Model_on_the_Data():
    print("****************************************************")
    topic = "7. Plotting the Model on the Data"; print("** %s\n" % topic)
    
    times              = np.array([  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.])
    measured_distances = np.array([ 0.24835708,  0.93086785,  2.32384427,  3.76151493,  3.88292331,
                                    4.88293152,  6.78960641,  7.38371736,  7.76526281,  9.27128002,
                                    9.76829115])
    
    def model(x, y):
        m = LinearRegression()
        m.fit(x.reshape(-1,1), y)
        # Get parameters
        #slope = model.coef_[0]
        #intercept = model.intercept_
        # Score prediction
        return(m.predict(x.reshape(-1,1)))
    
    # Pass times and measured distances into model
    model_distances = model(times, measured_distances)
    
    # Create figure and axis objects and call axis.plot() twice to plot data and model distances versus times
    fig, axis = plt.subplots()
    fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
    axis.plot(times, measured_distances, linestyle=" ", marker="o", color="black", label="Measured data")
    axis.plot(times, model_distances, linestyle="-", marker=None, color="red", label="Modeled data")

    # Customizing the plot
    axis.set_xlabel('X Data, Time Driven [hours]')
    axis.set_ylabel('Y Data, Distance Traveled [miles]')
    axis.set_title('Distance per hours Model', fontsize=14, color='maroon')  
 
    # Add grid lines and a legend to your plot, and then show to display
    axis.grid(True)
    axis.legend(loc="best")
    plt.show()
    
    
    

def Visually_Estimating_the_Slope_Intercept():
    print("****************************************************")
    topic = "8. Visually Estimating the Slope & Intercept"; print("** %s\n" % topic)

    x_data = np.array([ 2. ,  2.5,  3. ,  3.5,  4. ,  4.5,  5. ,  5.5,  6. ,  6.5,  7. ])
    y_data = np.array([ 4.24835708,  4.43086785,  5.32384427,  6.26151493,  5.88292331,  6.38293152,  
                        7.78960641,  7.88371736,  7.76526281,  8.77128002,  8.76829115])
    
    def model(x, y):
        """
        Retrieve the slope, intercept y y values according to the model.

        Parameters
        ----------
        x : First numpy array.
        y : Second numpy array.

        Returns
        -------
        xm : New x numpy array.
        ym : New y numpy array.
        slope : Slope of the model.
        intercept : Interception of the model.
        """
        m = LinearRegression()
        m.fit(x.reshape(-1,1), y)

        # Get parameters
        slope = m.coef_[0]
        intercept = m.intercept_
        
        # Score prediction
        xm = np.linspace(-5, 15, 41)
        ym = m.predict(xm.reshape(-1,1))
        return xm, ym, slope, intercept
    
    
    def plot_model(x, y, xm, ym, slope, intercept):
        """
        Pot the model according to the recolected data.

        Parameters
        ----------
        x and y: Two numpy array related to the recolected data.
        xm and ym: Modeled data.
        slope and intercept : Slope and interception related to the model.

        Returns
        -------
        The graph ploted.

        """
        # Create figure and axis objects and call axis.plot() twice to plot data and model distances versus times
        fig, ax = plt.subplots()
        fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
        
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.axvline(0, lw=2, color='black')
        ax.axhline(0, lw=2, color='black')
        
        ax.plot(x, y, linestyle=" ", marker="o", color="black", label="Measured data")
        ax.plot(xm, ym, linestyle="-", marker=None, color="red", label="Modeled data")
        
        # Customizing the plot
        ax.set_xlabel('X Data, Time Driven [hours]')
        ax.set_ylabel('Y Data, Distance Traveled [miles]')
        ax.set_title('Distance per hours Model\nSlope: {}, Intercept: {}'.format(slope, intercept), fontsize=14, color='maroon')  
        
        # Add grid lines and a legend to your plot, and then show to display
        ax.set_xticks(np.linspace(-10,10,5))
        ax.set_yticks(np.linspace(-10,10,5))
        ax.grid(b=True, which='major')
        ax.grid(b=True, which='minor', alpha=.4)
        ax.minorticks_on()
        ax.legend(loc="best")
        plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the marginsplt.show() 
        plt.show()
    
    # Pass times and measured distances into model
    x_model, y_model, slope, intercept = model(x_data, y_data)
    
    #Plot the model with the data
    plot_model(x_data, y_data, x_model, y_model, slope, intercept)
    
    
def Mean_Deviation_Standard_Deviation():
    print("****************************************************")
    topic = "10. Mean, Deviation, & Standard Deviation"; print("** %s\n" % topic)
    
    # Data Series
    x = np.array([ 3.20141089,  3.57332076,  4.2284669 ,  4.27233051,  4.49370529,
                   4.5713193 ,  4.74611933,  4.9143694 ,  5.06416613,  5.12046366,
                   5.1332596 ,  5.1382451 ,  5.19463348,  5.30012277,  5.32111385,
                   5.361098  ,  5.3622485 ,  5.42139975,  5.55601804,  5.56564872,
                   5.57108737,  5.60910021,  5.74438063,  5.82636432,  5.85993128,
                   5.90529103,  5.98816951,  6.00284592,  6.2829785 ,  6.28362732,
                   6.33858905,  6.3861864 ,  6.41291216,  6.57380586,  6.68822271,
                   6.73736858,  6.9071052 ,  6.92746243,  6.97873601,  6.99734545,
                   7.0040539 ,  7.17582904,  7.26593626,  7.49073203,  7.49138963,
                   7.65143654,  8.18678609,  8.20593008,  8.23814334,  8.39236527])

    y = np.array([ 146.48264883,  167.75876162,  229.73232314,  205.23686657,
                   224.99693822,  239.79378267,  246.65838372,  264.14477475,
                   268.91257002,  267.25180588,  248.54953839,  265.25831322,
                   263.03153004,  251.08035094,  280.93733241,  276.53088378,
                   268.59007072,  268.62252076,  265.21874   ,  280.37743899,
                   283.47297931,  271.72788298,  299.42217399,  279.79758387,
                   270.70401032,  306.18168601,  295.17313188,  298.81898515,
                   305.35499931,  297.3187572 ,  330.10944498,  312.07619563,
                   338.08560914,  337.16702908,  331.10617501,  325.46645358,
                   337.66440893,  333.64162871,  370.85149057,  351.59390525,
                   362.27985309,  345.48425572,  365.1976818 ,  386.90415177,
                   371.05186831,  393.39852867,  397.95134137,  395.98005292,
                   415.89087335,  415.63691073])

    # Compute the deviations by subtracting the mean offset
    dx = x - np.mean(x)
    dy = y - np.mean(y)

    # Normalize the data by dividing the deviations by the standard deviation
    zx = dx / np.std(x)
    zy = dy / np.std(y)

    # Plot comparisons of the raw data and the normalized data
    fig = plt.figure(figsize=(11.5, 5.5))
    fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
    
    # First part
    ax = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
    ax.plot(x, label="X data")
    ax.plot(y, label="Y data")
    ax.set_xlabel('Index')
    ax.set_ylabel('Data')
    ax.set_title('X and Y Series\nX Mean: {:7.2f}, X.Std: {:7.2f}\nY Mean: {:7.2f}, Y Std: {:7.2f}'.format(x.mean(), x.std(), y.mean(), y.std()), fontsize=10, color='maroon')  
    ax.legend(loc="best")
    
    # Second part
    ax = plt.subplot2grid((2, 2), (0, 1))
    ax.plot(dx, label="DX data")
    ax.plot(dy, label="DY data")
    ax.set_xlabel('Index')
    ax.set_ylabel('Data')
    ax.set_title('Deviation of X and Y Series\nDX Mean: {:7.2f}, DX.Std: {:7.2f}\nDY Mean: {:7.2f}, DY Std: {:7.2f}'.format(dx.mean(), dx.std(), dy.mean(), dy.std()), fontsize=10, color='maroon')  
    ax.legend(loc="best")
    
    # Third part
    ax = plt.subplot2grid((2, 2), (1, 1))
    ax.plot(zx, label="DX normalized")
    ax.plot(zy, label="DY normalized")
    ax.set_xlabel('Index')
    ax.set_ylabel('Data')
    ax.set_title('Normalized deviation of X and Y Series\nZX Mean: {:7.2f}, ZX.Std: {:7.2f}\nZY Mean: {:7.2f}, ZY Std: {:7.2f}'.format(zx.mean(), zx.std(), zy.mean(), zy.std()), fontsize=10, color='maroon')  
    ax.legend(loc="best")
    
    #Show the graph
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=.8); #To set the margins 
    plt.show()
    
    
def Covariance_vs_Correlation():
    print("****************************************************")
    topic = "11. Covariance vs Correlation"; print("** %s\n" % topic)

    # Data Series
    x = np.array([ 3.20141089,  3.57332076,  4.2284669 ,  4.27233051,  4.49370529,
                   4.5713193 ,  4.74611933,  4.9143694 ,  5.06416613,  5.12046366,
                   5.1332596 ,  5.1382451 ,  5.19463348,  5.30012277,  5.32111385,
                   5.361098  ,  5.3622485 ,  5.42139975,  5.55601804,  5.56564872,
                   5.57108737,  5.60910021,  5.74438063,  5.82636432,  5.85993128,
                   5.90529103,  5.98816951,  6.00284592,  6.2829785 ,  6.28362732,
                   6.33858905,  6.3861864 ,  6.41291216,  6.57380586,  6.68822271,
                   6.73736858,  6.9071052 ,  6.92746243,  6.97873601,  6.99734545,
                   7.0040539 ,  7.17582904,  7.26593626,  7.49073203,  7.49138963,
                   7.65143654,  8.18678609,  8.20593008,  8.23814334,  8.39236527])

    y = np.array([ 146.48264883,  167.75876162,  229.73232314,  205.23686657,
                   224.99693822,  239.79378267,  246.65838372,  264.14477475,
                   268.91257002,  267.25180588,  248.54953839,  265.25831322,
                   263.03153004,  251.08035094,  280.93733241,  276.53088378,
                   268.59007072,  268.62252076,  265.21874   ,  280.37743899,
                   283.47297931,  271.72788298,  299.42217399,  279.79758387,
                   270.70401032,  306.18168601,  295.17313188,  298.81898515,
                   305.35499931,  297.3187572 ,  330.10944498,  312.07619563,
                   338.08560914,  337.16702908,  331.10617501,  325.46645358,
                   337.66440893,  333.64162871,  370.85149057,  351.59390525,
                   362.27985309,  345.48425572,  365.1976818 ,  386.90415177,
                   371.05186831,  393.39852867,  397.95134137,  395.98005292,
                   415.89087335,  415.63691073])

    # Compute the covariance from the deviations.
    dx = x - np.mean(x)
    dy = y - np.mean(y)
    covariance = np.mean(dx * dy)
    
    # Compute the correlation from the normalized deviations.
    zx = dx / np.std(x)
    zy = dy / np.std(y)
    correlation = np.mean(zx * zy)
    msg = "Correlation = np.mean(zx * zy) = {}\nCovariance = {}".format(correlation, covariance)
    
    # Print the result
    print(msg)
    
    # Plot the normalized deviations for visual inspection. 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4))
    fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
    
    ax=ax1
    ax.plot(x, y, linestyle=" ", marker="o", color="black")
    #ax.set_xlim(-0.5, 11)
    #ax.set_ylim(-25, 500)
    ax.set_xlabel('Fuel Consumed [gallons]')
    ax.set_ylabel('Trip Distance [miles]')
    ax.set_title('50 Fill ups', fontsize=12, color='maroon')  
    ax.grid(True)
    
    ax=ax2
    ax.plot(zx*zy, lw=2, color="purple")
    ax.axhline(0, linestyle="--", lw=2, color='black')
    ax.set_xlabel('Array Index')
    ax.set_ylabel('Product of Normalized Deviations')
    ax.set_title(msg, fontsize=12, color='maroon')  
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the marginsplt.show() 
    plt.show()



def Correlation_Strength():
    print("****************************************************")
    topic = "12. Correlation Strength"; print("** %s\n" % topic)

    data_sets = {'A': {'i'          : 0,
                       'color'      : 'red',
                       'correlation': np.NaN,
                       'x'          : np.array([  2.55041235,   2.60839969,   2.79619981,   2.84385271,   3.15184751,
                                                  3.21906477,   3.23462037,   3.33976744,   3.47394544,   3.56125803,
                                                  3.67786134,   3.7339611 ,   3.86496991,   4.10019474,   4.24786673,
                                                  4.24920164,   4.29714059,   4.31952159,   4.41315702,   4.41783781,
                                                  4.42072788,   4.42420154,   4.62362038,   4.63538281,   4.70730828,
                                                  4.7073288 ,   4.71777962,   4.82716962,   4.85543965,   4.98312847,
                                                  5.08441026,   5.13865324,   5.21421035,   5.24607654,   5.26107949,
                                                  5.30245284,   5.39280917,   5.42952286,   5.46962252,   5.62089269,
                                                  5.67820005,   5.80961067,   5.92308322,   5.95929341,   6.02818114,
                                                  6.32140278,   6.83206096,   6.90378732,   6.97401602,   7.31534773]),
                       'y'          : np.array([  5.18184568,   5.12052882,   5.42316911,   5.84062449,   6.5614449 ,   
                                                  6.67094956,   6.25943637,   6.60223178,   7.03070673,   7.36640234,   
                                                  7.23592912,   7.42150745,   7.45335607,   7.90133782,   8.69886493,   
                                                  8.83746328,   8.57627865,   8.88992641,   8.91672304,   8.67439568,
                                                  8.93180467,   9.23291221,   9.23828425,   9.66192654,   8.75968029,   
                                                  9.62013323,   9.45732102,   9.57958741,   9.73381949,   9.46936471, 
                                                 10.11390254,  10.36658462,  10.79789421,  10.36258554,  10.32003559,  
                                                 10.47946642,  11.01446886,  10.9412335 ,  10.80680499,  11.37010224,
                                                 11.3806695 ,  11.86138259,  11.67065318,  11.83667129,  11.95833524,  
                                                 12.27692683,  13.73815199,  13.87283846,  13.9493104 ,  14.57204868])},
                 'B': {'i'          : 1,
                       'color'      : 'darkgoldenrod',
                       'correlation': np.NaN,
                       'x'          : np.array([  2.19664381,   2.406278  ,   2.47343147,   2.72871597,   3.06636806,
                                                  3.51128038,   3.87855402,   4.09926408,   4.18003832,   4.20434562,
                                                  4.29194259,   4.41336839,   4.50269971,   4.58240329,   4.59650649,
                                                  4.60918513,   4.74669209,   4.77111432,   4.82900646,   4.84738553,
                                                  5.00264796,   5.01962047,   5.02286149,   5.04517742,   5.09524948,
                                                  5.15589119,   5.24177672,   5.26908573,   5.30974025,   5.36136493,
                                                  5.42179707,   5.50681676,   5.58929395,   5.69179864,   5.84444261,
                                                  5.94426748,   6.05209339,   6.07448552,   6.07964661,   6.10895368,
                                                  6.19165516,   6.23993253,   6.30742282,   6.30947322,   6.32371148,
                                                  6.43754466,   6.64768944,   6.65144774,   6.79088371,   7.98870064]),
                       'y'          : np.array([  7.75732279,  -0.97068431,  -0.66103018,   5.05375913,   3.93976632,   
                                                  6.44408273,   9.17318937,   8.05647607,  10.62302986,  14.59132646,   
                                                  4.68693984,   8.54535728,  10.23727485,   8.33081153,  13.32821592,
                                                 -0.38344428,  17.61579867,   4.97170349,  10.50554646,  12.51365356,
                                                  6.86355506,  11.88747988,  12.86263588,  12.18438671,   6.48548172,  
                                                 18.34315419,  11.39140361,   5.92753502,  13.14739828,  10.8807806 ,  
                                                 12.70116343,  -3.24043311,  16.46301037,  11.99411949,  12.34700338,  
                                                 10.16815219,  15.17366173,  16.0886504 ,  13.24263662,  17.78585212,
                                                 12.70267957,  10.88000673,   8.5034434 ,  10.28007359,  15.91379868,  
                                                 12.5473011 ,  11.91631483,  15.41604806,   9.30581229,  13.92987605])},
                 'C': {'i'          : 2,
                       'color'      : 'darkgreen',
                       'correlation': np.NaN,
                       'x'          : np.array([  1.50176362,   1.96665095,   2.78558362,   2.84041313,   3.11713161,
                                                  3.21414912,   3.43264917,   3.64296175,   3.83020766,   3.90057957,
                                                  3.9165745 ,   3.92280638,   3.99329185,   4.12515346,   4.15139231,
                                                  4.2013725 ,   4.20281062,   4.27674969,   4.44502255,   4.45706091,
                                                  4.46385921,   4.51137526,   4.68047579,   4.7829554 ,   4.8249141 ,
                                                  4.88161379,   4.98521188,   5.00355739,   5.35372312,   5.35453415,
                                                  5.42323631,   5.482733  ,   5.5161402 ,   5.71725733,   5.86027839,
                                                  5.92171072,   6.13388149,   6.15932804,   6.22342001,   6.24668181,
                                                  6.25506737,   6.46978631,   6.58242032,   6.86341504,   6.86423703,
                                                  7.06429567,   7.73348261,   7.7574126 ,   7.79767917,   7.99045658]),
                       'y'          : np.array([-17.70183793, -12.68730947,  33.47056284,  -7.0881775 ,   6.7091949 ,  
                                                 23.53735376,  21.11660059,  35.3641024 ,  31.59072152,  24.91144186,  
                                                 -4.53019043,  20.56341545,  13.01493562, -12.96994045,  30.97956936,  
                                                 21.31852956,   9.13346253,   4.82402639, -10.28277321,  12.10650699,
                                                 16.42274434,  -4.27572923,  27.95621636,  -7.98933795, -24.3197774 ,  
                                                 26.39886103,   3.51656715,   7.99064142,  -2.69282132, -14.98633586,  
                                                 30.93027062,  -0.05643774,  37.60752021,  24.35144564,   6.68442643,  
                                                 -5.53101698,   0.5483712 ,  -7.08171402,  45.84065377,  15.1244233 ,
                                                 30.91342343,  -7.33806017,  16.06140272,  32.57262109,   8.36830187,  
                                                 30.62642269,  -1.88612137,  -6.30071951,  21.66576814,   9.91409021])}}
    
    # Making the function that will compute correlation.
    def correlation(x,y):
        x_dev = x - np.mean(x)
        y_dev = y - np.mean(y)
        x_norm = x_dev / np.std(x)
        y_norm = y_dev / np.std(y)
        return np.mean(x_norm * y_norm)

    fig, axis = plt.subplots(1, 3, figsize=(11.5, 3.5))
    fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
    
    # Compute and store the correlation for each data set in the list.
    for name, data in data_sets.items():
        data['correlation'] = correlation(data['x'], data['y'])
        print('Data set {} has correlation {:.2f}'.format(name, data['correlation']))
        
        # Graphing the serie
        ax = axis[data['i']]
        ax.set_xlim(0, 10)
        ax.set_ylim(-20, 20)
        ax.plot(data['x'], data['y'], linestyle=" ", marker="o", color=data['color'], label="Measured data")
        ax.axhline(0, linestyle="--", color='royalblue')
        ax.set_xlabel('X Data')
        ax.set_ylabel('Y Data')
        ax.set_title('Dataset {}\nCorrelation = {}'.format(name, data['correlation']), fontsize=12, color=data['color'])  
        ax.grid(True)
    
    plt.subplots_adjust(left=.07, right=.95, bottom=.15, top=.8, wspace=.3, hspace=None); #To set the marginsplt.show() 
    #plt.tight_layout() 
    plt.show()
    
    
     
    
def main():
    print("****************************************************")
    print("** BEGIN                                          **")
    
    Reasons_for_Modeling_Interpolation()
    Reasons_for_Modeling_Extrapolation()
    Reasons_for_Modeling_Estimating_Relationships()
    Plotting_the_Data()
    Plotting_the_Model_on_the_Data()
    Visually_Estimating_the_Slope_Intercept()
    Mean_Deviation_Standard_Deviation()
    Covariance_vs_Correlation()
    Correlation_Strength()
        
    print("****************************************************")
    print("** END                                            **")
    print("****************************************************")

if __name__ == '__main__':
    main()