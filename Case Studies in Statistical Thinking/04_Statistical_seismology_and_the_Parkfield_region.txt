# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 4: Statistical seismology and the Parkfield region
    Herein, you'll use your statistical thinking skills to study the frequency 
    and magnitudes of earthquakes. Along the way, you'll learn some basic 
    statistical seismology, including the Gutenberg-Richter law. This exercise 
    exposes two key ideas about data science: 
        1) As a data scientist, you wander into all sorts of domain specific 
           analyses, which is very exciting. You constantly get to learn. 
        2) You are sometimes faced with limited data, which is also the case 
           for many of these earthquake studies. You can still make good 
           progress!
Source: https://learn.datacamp.com/courses/case-studies-in-statistical-thinking
"""
###############################################################################
## Importing libraries
###############################################################################
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats

from dateutil.relativedelta import relativedelta

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import Stamen

#import dc_stat_think as dcst
import my_own_stat_think as most #My own functions 

###############################################################################
## Preparing the environment
###############################################################################
#Global variables
suptitle_param = dict(color='darkblue', fontsize=11)
title_param    = {'color': 'darkred', 'fontsize': 14, 'weight': 'bold'}
plot_param     = {'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                  'legend.fontsize': 8, 'font.size': 8}
figsize        = (12.1, 5.9)
SEED           = 42
SIZE           = 10000

# Global configuration
sns.set()
pd.set_option("display.max_columns", 24)
plt.rcParams.update(**plot_param)
np.random.seed(SEED)
np.set_printoptions(formatter={'float': '{:.3f}'.format})

# Global functions
def b_value(mags, mt, perc=[2.5,97.5], n_reps=None):
    """Compute the b-value and optionally its confidence interval."""
    # Extract magnitudes above completeness threshold: m
    m = mags[mags >= mt]
    # Compute b-value: b
    b = (m.mean() - mt)*np.log(10)
    # Draw bootstrap replicates
    if n_reps is None:
        return b
    else:
        m_bs_reps = most.draw_bs_reps(m, np.mean, n_reps)
        # Compute b-value from replicates: b_bs_reps
        b_bs_reps = (m_bs_reps - mt) * np.log(10)
        # Compute confidence interval: conf_int
        conf_int = np.percentile(b_bs_reps, perc)
        return b, conf_int
        
###############################################################################
## Reading the data
###############################################################################
parkfield = pd.read_csv('parkfield_earthquakes_1950-2017.csv', skiprows=2, parse_dates=['time'])
oklahome = pd.read_csv('oklahoma_earthquakes_1950-2017.csv', skiprows=2, parse_dates=['time'])

###############################################################################
## Main part of the code
###############################################################################
def Statistical_seismology_and_the_Parkfield_region(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "1. Statistical seismology and the Parkfield region"; print("** %s" % topic)
    print("****************************************************")
    
    print('----------------------------------Exploring Oklahome')
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection=ccrs.PlateCarree()))
    #ax.set_extent([-103.5, -93.75, 33, 37.5], crs=ccrs.Geodetic(globe=ccrs.Globe(datum='WGS84')))
    #ax.set_extent([-120.8, -120.2, 36.1, 35.6], crs=ccrs.PlateCarree())
    ax.set_extent([-120.7, -120.2, 36.16, 35.64], crs=ccrs.PlateCarree())
    # Put a background image on for nice sea rendering.
    ax.stock_img()
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE, edgecolor='lightsteelblue')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS, alpha=0.5)
    # Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
    states_provinces = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='50m', facecolor='none')
    ax.add_feature(states_provinces, edgecolor='gray')
    
    # Add the Stamen aerial imagery at zoom level 10.
    ax.add_image(Stamen('terrain-background'), 12)
    
    #Plot the earthquake information (longitud, latitud)
    sns.scatterplot(x='longitude', y='latitude', data=parkfield, s=10*(2.5**oklahome.mag.values), color='red', alpha=.25)
    ax.grid()
    
    ax.set_title("Parkfield city", **title_param)
    fig.suptitle(topic, **suptitle_param)
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    
    
def Parkfield_earthquake_magnitudes(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "2. Parkfield earthquake magnitudes"; print("** %s" % topic)
    print("****************************************************")
    
    print('------------------------------------Reading the data')
    mags = parkfield.mag.values
    m_mean = mags.mean()
    m_median = np.median(mags)
    
    print('------------------------------------------------ECDF')
    fig, ax = plt.subplots()
    
    ax.plot(*most.ecdf(mags), marker='.', ls='none')
    
    ax.axvline(m_mean, color='green', label=f'Average: {m_mean:,.5f}')
    ax.axvline(m_median, color='red', label=f'Median: {m_median:,.5f}')
    ax.axhline(.75, color='black', ls='--', lw=.5)
    ax.axhline(.5, color='black', ls='--', lw=.5)
    ax.axhline(.25, color='black', ls='--', lw=.5)
    
    msg = 'Note the distinctive roll-off\nat magnitudes below 1.0'
    t = ax.text(0.4, 0.55, msg, transform=ax.transAxes, color='black', ha='left', va='bottom', fontsize=14)
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='darkgray'))
    
    ax.set_xlabel('Magnitude')
    ax.set_ylabel('ECDF')
    ax.set_title("Earthquakes detected in the Parkfield region\nfrom 1950 to 2016", **title_param)
    ax.legend()
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    
    
def The_bvalue_for_Parkfield(size=100000, seed=SEED):
    print("****************************************************")
    topic = "4. The b-value for Parkfield"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------------------------Reading the data')
    mags = parkfield.mag.values
    # Define the completeness threshold
    mt = 3
    
    print('---------------------------------------------b_value')
    # Compute b-value and 95% confidence interval
    b, conf_int = b_value(mags, mt, n_reps=size)
    # Report the results
    msg = f"b-value: {b:.2f}\n" +\
           "95% conf int: [{1:.2f}, {2:.2f}]".format(b, *conf_int)
    print(msg)
    
    print('-----------------------Theorical model - Exponential')
    # Generate samples to for theoretical ECDF
    m_theor = np.random.exponential(b/np.log(10), size=size) + mt
    
    print('------------------------------------------------ECDF')
    fig, ax = plt.subplots()
    
    # Plot the theoretical CDF
    ax.plot(*most.ecdf(m_theor), color='red', label='Theorical model')
    # Plot the ECDF (slicing mags >= mt)
    ax.plot(*most.ecdf(mags[mags >= 3]), color='blue', marker='.', ls='none', label='Recorded Data')
    
    t = ax.text(0.45, 0.1, msg, transform=ax.transAxes, color='black', ha='left', va='bottom', fontsize=14)
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='darkgray'))
    
    ax.set_xlim(2.8, 6.2)
    ax.set_xlabel('Magnitude')
    ax.set_ylabel('ECDF')
    ax.set_title("Earthquakes detected in the Parkfield region\nfrom 1950 to 2016 - (Magnitud above 3)", **title_param)
    ax.legend()
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
          
          
def Timing_of_major_earthquakes_and_the_Parkfield_sequence(size=100000, seed=SEED):
    print("****************************************************")
    topic = "5. Timing of major earthquakes and the Parkfield sequence"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('----------------Preparing the data (between>14 days)')
    # Initialize seed and parameters
    np.random.seed(seed) 
    oklahome['days_between'] = oklahome['time'].diff().fillna(pd.Timedelta(seconds=0)) / pd.Timedelta(days=1)
    oklahome_more_than_14days = oklahome[oklahome.days_between>14] 
    print(oklahome_more_than_14days[['time', 'days_between']].info())
    
    print('-------------------------ECDF time (between>14 days)')
    t_mean = oklahome_more_than_14days.days_between.mean()
    t_median = np.median(oklahome_more_than_14days.days_between)
    
    # Theorical model
    theorical_model = np.random.exponential(scale=t_mean, size=size)
    
    # Plot the ECDFs as dots
    fig, ax = plt.subplots()
    
    ax.plot(*most.ecdf(oklahome_more_than_14days.days_between), marker='.', ls='none', label='Recorded data')
    ax.plot(*most.ecdf(theorical_model), color='blue', lw=.75, ls='--', label='Theorical Model')
    ax.axvline(t_mean, color='green', lw=.75, label='Average')
    ax.axvline(t_median, color='red', lw=.75, label='Median')
    #ax.set_xlim(0,425)
    ax.axhline(.5, color='black', lw=.75, ls='--', label='50% of the data')
    
    msg = "It has omitted inter-earthquake\ntimes less than two weeks."
    t = ax.text(0.3, 0.55, msg, transform=ax.transAxes, color='black', ha='left', va='bottom', fontsize=14)
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='darkgray'))
    
    ax.set_xlabel('Time between quakes (days)')
    ax.set_ylabel('ECDF')
    ax.set_title("Oklahoma Earthquakes\nTiming", **title_param)
    ax.legend()
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.show()   
    
    
    print('--------------------------------Reading data (mag>7)')
    ## Magenging dates out of range
    ## OutOfBoundsDatetime when cleaning different date formats with dates before 1677
    ## pd.Timestamp.min --> Timestamp('1677-09-21 00:12:43.145225')
    ## pd.Timestamp.max --> Timestamp('2262-04-11 23:47:16.854775807')
    """
    a = relativedelta.relativedelta(datetime.datetime(887,8,22), datetime.datetime(684,11,24))
    #Out[174]: relativedelta(years=+202, months=+8, days=+29)
    a.years
    #Out[175]: 202
    """
    Earthquakes_nankai = pd.DataFrame({'date': ['0684-11-24', '0887-08-22', '1099-02-16', '1361-07-26', 
                                                '1498-09-11', '1605-02-03', '1707-10-18', '1854-12-23', 
                                                '1946-12-24'],
                                       'mag' : [8.4, 8.6, 8.0, 8.4, 8.6, 7.9, 8.6, 8.4, 8.1]})
    #Earthquakes_nankai = Earthquakes_nankai[Earthquakes_nankai.mag>=8]
    Earthquakes_nankai['date2'] = Earthquakes_nankai.date.shift()
    
    def diff_in_years(row):
        if pd.isnull(row.date2):
            return 0
        else:
            return relativedelta(datetime.datetime.strptime(row.date, '%Y-%m-%d'), 
                                 datetime.datetime.strptime(row.date2, '%Y-%m-%d')).years
    Earthquakes_nankai['years_between'] = Earthquakes_nankai.apply(diff_in_years, axis=1)
    Earthquakes_nankai = Earthquakes_nankai[Earthquakes_nankai.years_between>0][['date', 'mag', 'years_between']]
    print(Earthquakes_nankai)
    


    print('---------------ECDF time between earthquakes (mag>7)')
    mean_time_gap = Earthquakes_nankai.years_between.mean()
    std_time_gap = np.std(Earthquakes_nankai.years_between)
    #median_time_gap = np.median(Earthquakes_nankai.years_between)
    
    # Theorical model
    theorical_model_exp = np.random.exponential(scale=mean_time_gap, size=size)
    theorical_model_gau = np.random.normal(loc=mean_time_gap, scale=std_time_gap, size=size)
    
    # Plot the ECDFs as dots
    fig, ax = plt.subplots()
    
    ax.plot(*most.ecdf(Earthquakes_nankai.years_between, formal=True, min_x=0, max_x=300), color='blue')
    ax.plot(*most.ecdf(Earthquakes_nankai.years_between), color='darkblue', ms=10, marker='.', ls='none', label='Recorded data')
    
    ax.plot(*most.ecdf(theorical_model_exp), color='green', ls='--', label='Exponential Model')
    ax.plot(*most.ecdf(theorical_model_gau), color='red', ls='--', label='Normal Model')
    
    #ax.axvline(mean_time_gap, color='gray', lw=.5, ls='-', label='Average')
    #ax.axvline(median_time_gap, color='gray', lw=.5, ls='--',  label='Median')
    #ax.axhline(.5, color='black', lw=.75, ls='--', label='50% of the data')
    
    msg = "Magnitud greater\nthan 7 megathrust."
    t = ax.text(0.5, 0.1, msg, transform=ax.transAxes, color='black', ha='left', va='bottom', fontsize=14)
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='darkgray'))
    
    ax.set_xlabel('Time between quakes (Years)')
    ax.set_xlim(0,300)
    ax.set_ylabel('ECDF')
    ax.set_title("Nankai Earthquakes\nTiming", **title_param)
    ax.legend(loc='center right')
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.show() 
    
    
        
def Interearthquake_time_estimates_for_Parkfield(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "6. Interearthquake time estimates for Parkfield"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('-----------------------------------Data in excercise')
    time_gap = np.array([24.06570842, 20.07665982, 21.01848049, 12.24640657, 32.05475702, 38.2532512 ])
    
    # Compute the mean time gap: mean_time_gap
    mean_time_gap = time_gap.mean()
    # Standard deviation of the time gap: std_time_gap
    std_time_gap = np.std(time_gap)
    
    # Generate theoretical Exponential distribution of timings: time_gap_exp
    time_gap_exp = np.random.exponential(scale=mean_time_gap, size=size)
    # Generate theoretical Normal distribution of timings: time_gap_norm
    time_gap_norm = np.random.normal(loc=mean_time_gap, scale=std_time_gap, size=size)
    
    # Plot the ECDFs as dots
    fig, ax = plt.subplots()
    
    # Plot theoretical CDFs
    ax.plot(*most.ecdf(time_gap_exp), label='Exponential')
    ax.plot(*most.ecdf(time_gap_norm), label='Normal')
    
    # Plot Parkfield ECDF
    ax.plot(*most.ecdf(time_gap, formal=True, min_x=-10, max_x=50), label='Data')
    ax.axvline(mean_time_gap, color='gray', lw=.5, ls='--',  label='Average Data')
    
    # Add legend
    ax.legend(loc='upper left')
    
    # Label axes, set limits and show plot
    ax.set_xlim(-10, 50)
    ax.set_xlabel('time gap (years)')
    ax.set_ylabel('ECDF')
    ax.set_title("Parkerfield Earthquakes Data Provided\nTiming", **title_param)
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    
    plt.show()
    
    print('----------------------------Working with Parkerfield')
    mag_threshold = 4.5
    df = parkfield[parkfield.mag>=mag_threshold].copy()
    df['years_between'] = df['time'].diff().fillna(pd.Timedelta(seconds=0)) / np.timedelta64(1, 'Y')
    df = df[df.years_between>=1][['time', 'mag', 'years_between']]
    print(df)
    
    time_gap = df.years_between.values
    
    # Compute the mean time gap: mean_time_gap
    mean_time_gap = time_gap.mean()
    # Standard deviation of the time gap: std_time_gap
    std_time_gap = np.std(time_gap)
    
    # Generate theoretical Exponential distribution of timings: time_gap_exp
    time_gap_exp = np.random.exponential(scale=mean_time_gap, size=size)
    # Generate theoretical Normal distribution of timings: time_gap_norm
    time_gap_norm = np.random.normal(loc=mean_time_gap, scale=std_time_gap, size=size)
    
    # Plot the ECDFs as dots
    fig, ax = plt.subplots()
    
    # Plot theoretical CDFs
    ax.plot(*most.ecdf(time_gap_exp), label='Exponential')
    ax.plot(*most.ecdf(time_gap_norm), label='Normal')
    
    # Plot Parkfield ECDF
    ax.plot(*most.ecdf(time_gap, formal=True, min_x=-10, max_x=20), label='Data')
    ax.axvline(mean_time_gap, color='gray', lw=.5, ls='--',  label='Average Data')
    
    # Add legend
    ax.legend(loc='upper left')
    
    # Label axes, set limits and show plot
    ax.set_xlim(-10, 20)
    ax.set_xlabel('time gap (years)')
    ax.set_ylabel('ECDF')
    ax.set_title(f"Parkerfield Earthquakes greater than {mag_threshold}\nBetween {parkfield.time.dt.year.min()} y {parkfield.time.dt.year.max()} - (Timing)", **title_param)
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    
    plt.show()
    
    
    
def When_will_the_next_big_Parkfield_quake_be(size=100000, seed=SEED):
    print("****************************************************")
    topic = "7. When will the next big Parkfield quake be?"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('-----------------------------------Data in excercise')
    time_gap = np.array([24.06570842, 20.07665982, 21.01848049, 12.24640657, 32.05475702, 38.2532512 ])
    last_quake = pd.Timestamp('2004-09-27 18:00:00')
    gap = (pd.Timestamp.today() - last_quake) / np.timedelta64(1, 'Y')
    
    # Compute the mean time gap: mean_time_gap
    mean_time_gap = time_gap.mean()
    # Standard deviation of the time gap: std_time_gap
    std_time_gap = np.std(time_gap)
    
    print(f"The last big earthquake in the Parkfield region was on {last_quake}." +\
           "Your task is to get an estimate as to when the next Parkfield quake will be.\n" +\
          f"Mean time: {mean_time_gap:,.5f}.\n" +\
          f"Std: {std_time_gap:,.5f}.\n\n")
    
    # Draw samples from the Exponential distribution: exp_samples
    exp_samples = np.random.exponential(scale=mean_time_gap, size=size)
    # Draw samples from the Normal distribution: norm_samples
    norm_samples = np.random.normal(loc=mean_time_gap, scale=std_time_gap, size=size)
    
    # No earthquake as of today, so only keep samples that are long enough
    exp_samples = exp_samples[exp_samples > gap]
    norm_samples = norm_samples[norm_samples > gap]
    
    # Compute the confidence intervals with medians
    conf_int_exp = np.percentile(exp_samples, [2.5, 50, 97.5]) + (last_quake.year + ((last_quake.dayofyear - 1)/365))
    conf_int_norm = np.percentile(norm_samples, [2.5, 50, 97.5]) + (last_quake.year + ((last_quake.dayofyear - 1)/365))
    
    # Print the results
    print('Exponential:', conf_int_exp)
    print('     Normal:', conf_int_norm)
    
    
    print('\n\n----------------------------Working with Parkerfield')
    mag_threshold = 4.5
    df = parkfield[parkfield.mag>=mag_threshold].copy()
    df['years_between'] = df['time'].diff().fillna(pd.Timedelta(seconds=0)) / np.timedelta64(1, 'Y')
    df = df[df.years_between>=1][['time', 'mag', 'years_between']]
    
    last_quake = df.time.max()
    gap = (pd.Timestamp.today() - last_quake) / np.timedelta64(1, 'Y')
    time_gap = df.years_between.values
    
    # Compute the mean time gap: mean_time_gap
    mean_time_gap = time_gap.mean()
    # Standard deviation of the time gap: std_time_gap
    std_time_gap = np.std(time_gap)
    
    
    print(f"Now predict the next +{mag_threshold} eartquake in the Parkfield base on the data we have.\n" +\
          f"Mean time: {mean_time_gap:,.5f}.\n" +\
          f"Std: {std_time_gap:,.5f}.\n"
          f"Last earthquake: {last_quake}.\n\n")
    
    # Draw samples from the Exponential distribution: exp_samples
    exp_samples = np.random.exponential(scale=mean_time_gap, size=size)
    # Draw samples from the Normal distribution: norm_samples
    norm_samples = np.random.normal(loc=mean_time_gap, scale=std_time_gap, size=size)
    
    # No earthquake as of today, so only keep samples that are long enough
    exp_samples = exp_samples[exp_samples > gap]
    norm_samples = norm_samples[norm_samples > gap]
    
    # Compute the confidence intervals with medians
    conf_int_exp = np.percentile(exp_samples, [2.5, 50, 97.5]) + (last_quake.year + ((last_quake.dayofyear - 1)/365))
    conf_int_norm = np.percentile(norm_samples, [2.5, 50, 97.5]) + (last_quake.year + ((last_quake.dayofyear - 1)/365))
    
    # Print the results
    print('Exponential:', conf_int_exp)
    print('     Normal:', conf_int_norm)
    
    
    
def How_are_the_Parkfield_interearthquake_times_distributed(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "8. How are the Parkfield interearthquake times distributed?"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('--------------------------------Reading data (mag>7)')
    Earthquakes_nankai = pd.DataFrame({'date': ['0684-11-24', '0887-08-22', '1099-02-16', '1361-07-26', 
                                                '1498-09-11', '1605-02-03', '1707-10-18', '1854-12-23', 
                                                '1946-12-24'],
                                       'mag' : [8.4, 8.6, 8.0, 8.4, 8.6, 7.9, 8.6, 8.4, 8.1]})
    Earthquakes_nankai['date2'] = Earthquakes_nankai.date.shift()
    
    def diff_in_years(row):
        if pd.isnull(row.date2):
            return 0
        else:
            return relativedelta(datetime.datetime.strptime(row.date, '%Y-%m-%d'), 
                                 datetime.datetime.strptime(row.date2, '%Y-%m-%d')).years
    Earthquakes_nankai['years_between'] = Earthquakes_nankai.apply(diff_in_years, axis=1)
    Earthquakes_nankai = Earthquakes_nankai[Earthquakes_nankai.years_between>0][['date', 'mag', 'years_between']]
    print(Earthquakes_nankai)
    
    print('------------------------------------Theorical models')
    time_gap = Earthquakes_nankai.years_between.values
    mean_time_gap = time_gap.mean()
    std_time_gap = np.std(time_gap)
    
    # Theorical model
    theorical_model_exp = np.random.exponential(scale=mean_time_gap, size=size)
    theorical_model_gau = np.random.normal(loc=mean_time_gap, scale=std_time_gap, size=size)
    
    
    print('---------------ECDF time between earthquakes (mag>7)')
    # Plot the ECDFs as dots
    fig, axes = plt.subplots(2,2,figsize=figsize)
    x, y = most.ecdf(time_gap)
    
    ax = axes[0,0]
    theo_x, theo_y = most.ecdf(theorical_model_exp)
    ax.plot(*most.ecdf(time_gap, formal=True, min_x=0, max_x=300), color='blue')
    ax.plot(x, y, color='darkblue', ms=10, marker='.', ls='none', label='Recorded data')
    for vx in x: ax.axvline(vx, color='orange', lw=.5)
    ax.set_xlim(0,300)
    ax.plot(theo_x, theo_y, color='green', label='Exponential Model')
    ax.set_xlabel('Time between quakes (years)')
    ax.set_ylabel('ECDF')
    ax.set_title("Using a Exponential Model", **title_param)
    ax.legend(loc='center right')
    
    ax = axes[1,0]
    trayecto = most.ecdf_formal(theo_x, x)
    distance = np.abs(trayecto - theo_y)
    max_diff = distance.max()
    ax.plot(theo_x, trayecto, color='gray', label='Trayecto')
    ax.plot(theo_x, distance, color='green', ls='--', label='Distance from the model')
    for vx in x: ax.axvline(vx, color='orange', lw=.5)
    msg = f"Max distance: {max_diff}"
    t = ax.text(0.05, 0.8, msg, transform=ax.transAxes, color='black', ha='left', va='bottom', fontsize=14)
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='darkgray'))
    ax.set_xlim(0,300)
    ax.set_xlabel('Time between quakes (years)')
    ax.set_ylabel('ECDF')
    ax.set_title("Distance from the Exponential Model", **title_param)
    ax.legend(loc='center right')
        
    ax = axes[0,1]
    theo_x, theo_y = most.ecdf(theorical_model_gau)
    ax.plot(*most.ecdf(time_gap, formal=True, min_x=0, max_x=300), color='blue')
    ax.plot(x, y, color='darkblue', ms=10, marker='.', ls='none', label='Recorded data')
    ax.plot(theo_x, theo_y, color='red', label='Normal Model')
    for vx in x: ax.axvline(vx, color='orange', lw=.5)
    ax.set_xlim(0,300)
    ax.set_xlabel('Time between quakes (years)')
    ax.set_ylabel('ECDF')
    ax.set_title("Using a Gauss Model", **title_param)
    ax.legend(loc='center right')
    
    ax = axes[1,1]
    trayecto = most.ecdf_formal(theo_x, x)
    distance = np.abs(trayecto - theo_y)
    max_diff = distance.max()
    
    """
    #Getting distances for concave and convex corner step-->recorded data is in the concave corner step
    # Compute distances between concave corners and CDF
    D_top = trayecto - theo_y
    # Compute distance between convex corners and CDF
    D_bottom = trayecto - theo_y + 1/len(theo_y)
    max_diff2 = np.max((D_top, D_bottom))
    
    #Getting distances for concave and convex corner step-->recorded data is in the concave corner step
    max_diff3 = most.ks_stat(time_gap, theorical_model_gau)
    """
    ax.plot(theo_x, trayecto, color='gray', label='Trayecto')
    ax.plot(theo_x, distance, color='red', ls='--', label='Distance from the model')
    for vx in x: ax.axvline(vx, color='orange', lw=.5)
    msg = f"Max distance: {max_diff}"#", ({max_diff2})"
    t = ax.text(0.05, 0.8, msg, transform=ax.transAxes, color='black', ha='left', va='bottom', fontsize=14)
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='darkgray'))
    ax.set_xlim(0,300)
    ax.set_xlabel('Time between quakes (years)')
    ax.set_ylabel('ECDF')
    ax.set_title("Distance from the Gauss Model", **title_param)
    ax.legend(loc='center right')
        
    fig.suptitle(f"{topic}\nNankai Earthquakes Greater than 7 - Timing", **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=.4); #To set the margins 
    plt.show() 
    
    
    print('-----------------------------------HYPOTHESIS TEST 1')
    print('-----------DATA BEHEAVES AS EXPONENTIAL DISTRIBUTION')
    
    print('----------------------------------------------STEP 1')
    # Step 1: Clearly state the null hypothesis
    Hypothesis_test = {'Ho'                    : 'The time between Nankai Trough earthquakes is Exponentially distributed\n' +\
                                                 'with a mean as calculated from the data', 
                       'Test statistic'        : 'Kolmogorov-Smirnov statistic', 
                       'At least as extreme as': 'Greater than or equal to what was observed'}
    
    print('----------------------------------------------STEP 2')
    # Step 2: Define your test statistic
    ks_observed = most.ks_stat(time_gap, theorical_model_exp)
    print(f"{ks_observed:,.5f}")
    
    #print('------------------------------------------STEP 3 y 4')
    ## Step 3: Generate many sets of simulated data assuming the null hypothesis is true
    ## Step 4: Compute the test statistic for each simulated data set
    # Initialize K-S replicates
    reps_ks = np.zeros(size)
    # Draw replicates
    for i in range(size):
        # Draw samples for comparison
        x_samp = np.random.exponential(mean_time_gap, size=len(time_gap))
        # Compute K-S statistic
        reps_ks[i] = most.ks_stat(x_samp, theorical_model_exp)
    
    print('----------------------------------------------STEP 5')
    # Step 5: The p-value is the fraction of your simulated data sets for which the test statistic is at least as extreme as for the real data
    p_val = np.mean(reps_ks >= ks_observed)
    
    msg = "Hypothesis Test 1\n" +\
          "Ho: {}.\n\n".format(Hypothesis_test['Ho']) +\
          "Test statistic observed: {:,.5f}. \n({}).\n\n".format(ks_observed, Hypothesis_test['Test statistic']) +\
          "p-value: {:,.5f}. {}.\n".format(p_val, Hypothesis_test['At least as extreme as']) +\
          "Ho is {}.".format(np.where(p_val<0.01, 'rejected', 'accepted'))
    print(msg)
    
    print('--------------------Visualizing the permutation test')
    #Visualize th p-value to help understanding - Lesson: 20.4.1
    fig, ax = plt.subplots(figsize=figsize)
    most.plotting_test_Ho(ax, reps_ks, ks_observed, msg = msg, 
                          x_label='Kolmogorov-Smirnov statistic', 
                          title='Data behaves as Exponential distribution?',
                          params_title=title_param, greater=True)
    plt.subplots_adjust(left=.1, right=.95, bottom=.15, top=.9, hspace=None, wspace=None);
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    
    
    print('-----------------------------------HYPOTHESIS TEST 2')
    print('-----------------DATA BEHEAVES AS GAUSS DISTRIBUTION')
    
    print('----------------------------------------------STEP 1')
    # Step 1: Clearly state the null hypothesis
    Hypothesis_test = {'Ho'                    : 'The time between Nankai Trough earthquakes is Normally distributed\n' +\
                                                 'with a mean and standard deviation as calculated from the data', 
                       'Test statistic'        : 'Kolmogorov-Smirnov statistic', 
                       'At least as extreme as': 'Greater than or equal to what was observed'}
    
    print('----------------------------------------------STEP 2')
    # Step 2: Define your test statistic
    ks_observed = most.ks_stat(time_gap, theorical_model_gau)
    print(f"{ks_observed:,.5f}")
    
    print('------------------------------------------STEP 3 y 4')
    ## Step 3: Generate many sets of simulated data assuming the null hypothesis is true
    ## Step 4: Compute the test statistic for each simulated data set
    # Initialize K-S replicates
    reps_ks = np.zeros(size)
    # Draw replicates
    for i in range(size):
        # Draw samples for comparison
        x_samp = np.random.normal(mean_time_gap, std_time_gap, size=len(time_gap))
        # Compute K-S statistic
        reps_ks[i] = most.ks_stat(x_samp, theorical_model_gau)
    
    print('----------------------------------------------STEP 5')
    # Step 5: The p-value is the fraction of your simulated data sets for which the test statistic is at least as extreme as for the real data
    p_val = np.mean(reps_ks >= ks_observed)
    
    msg = "Hypothesis Test 1\n" +\
          "Ho: {}.\n\n".format(Hypothesis_test['Ho']) +\
          "Test statistic observed: {:,.5f}. \n({}).\n\n".format(ks_observed, Hypothesis_test['Test statistic']) +\
          "p-value: {:,.5f}. {}.\n".format(p_val, Hypothesis_test['At least as extreme as']) +\
          "Ho is {}.".format(np.where(p_val<0.01, 'rejected', 'accepted'))
    print(msg)
    
    print('--------------------Visualizing the permutation test')
    #Visualize th p-value to help understanding - Lesson: 20.4.1
    fig, ax = plt.subplots(figsize=figsize)
    most.plotting_test_Ho(ax, reps_ks, ks_observed, msg = msg, 
                          x_label='Kolmogorov-Smirnov statistic', 
                          title='Data behaves as Gauss distribution?',
                          params_title=title_param, greater=True)
    plt.subplots_adjust(left=.1, right=.95, bottom=.15, top=.9, hspace=None, wspace=None);
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    
    
def The_KS_test_for_Exponentiality(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "12. The K-S test for Exponentiality"; print("** %s" % topic)
    print("****************************************************")
    
    print('----------------------------------------------------')
    print('--------------------------Nankai earthquakes (mag>7)')
    print('----------------------------------------------------')
    Earthquakes_nankai = pd.DataFrame({'date': ['0684-11-24', '0887-08-22', '1099-02-16', '1361-07-26', 
                                                '1498-09-11', '1605-02-03', '1707-10-18', '1854-12-23', 
                                                '1946-12-24'],
                                       'mag' : [8.4, 8.6, 8.0, 8.4, 8.6, 7.9, 8.6, 8.4, 8.1]})
    Earthquakes_nankai['date2'] = Earthquakes_nankai.date.shift()
    
    def diff_in_years(row):
        if pd.isnull(row.date2):
            return 0
        else:
            return relativedelta(datetime.datetime.strptime(row.date, '%Y-%m-%d'), 
                                 datetime.datetime.strptime(row.date2, '%Y-%m-%d')).years
    Earthquakes_nankai['years_between'] = Earthquakes_nankai.apply(diff_in_years, axis=1)
    Earthquakes_nankai = Earthquakes_nankai[Earthquakes_nankai.years_between>0][['date', 'mag', 'years_between']]
    print(Earthquakes_nankai)
    
    time_gap = Earthquakes_nankai.years_between.values
    mean_time_gap = time_gap.mean()
    std_time_gap = np.std(time_gap)
    
    print('-----------------------------Using my_own_stat_think')
    print('       Hypothesis Test - Data Behaves as exp distrib')
    # Initialize seed and parameters
    np.random.seed(seed) 
    print('Step 1 - State the hypothesis')
    Hypothesis_test = {'Ho'                    : 'The time between Nankai Trough earthquakes is Exponentially distributed ' +\
                                                 'with a mean as calculated from the data', 
                       'Test statistic'        : 'Kolmogorov-Smirnov statistic', 
                       'At least as extreme as': 'Greater than or equal to what was observed'}
    print('Step 2 - Define the test statistic')
    theorical_model = np.random.exponential(scale=mean_time_gap, size=size)
    ks_observed = most.ks_stat(time_gap, theorical_model)
    print('Step 3 - Generate simulated data')
    print('Step 4 - Compute statistic for simulated data')
    reps = most.draw_ks_reps(len(time_gap), np.random.exponential, args=(mean_time_gap,), 
                             size=size, n_reps=size)
    print('Step 5 - Compute the p-value')
    p_val = np.mean(reps >= ks_observed)
    msg = "***Hypothesis Test***\n" +\
          "Ho: {}.\n\n".format(Hypothesis_test['Ho']) +\
          "Test statistic observed: {:,.5f}. \n({}).\n\n".format(ks_observed, Hypothesis_test['Test statistic']) +\
          'p-value: {:,.5f}. (Probability of "{}").\n'.format(p_val, Hypothesis_test['At least as extreme as']) +\
          "Ho is {}.".format(np.where(p_val<0.01, 'rejected', 'accepted'))
    print(msg)
    
    
    print('-----------------------------Using my_own_stat_think')
    print('     Hypothesis Test - Data Behaves as gauss distrib')
    # Initialize seed and parameters
    np.random.seed(seed) 
    print('Step 1 - State the hypothesis')
    Hypothesis_test = {'Ho'                    : 'The time between Nankai Trough earthquakes is Gauss distributed ' +\
                                                 'with a mean and standard deviation as calculated from the data', 
                       'Test statistic'        : 'Kolmogorov-Smirnov statistic', 
                       'At least as extreme as': 'Greater than or equal to what was observed'}
    print('Step 2 - Define the test statistic')
    theorical_model = np.random.normal(loc=mean_time_gap, scale=std_time_gap, size=size)
    ks_observed = most.ks_stat(time_gap, theorical_model)
    print('Step 3 - Generate simulated data')
    print('Step 4 - Compute statistic for simulated data')
    reps = most.draw_ks_reps(len(time_gap), np.random.normal, args=(mean_time_gap, std_time_gap,), 
                             size=size, n_reps=size)
    print('Step 5 - Compute the p-value')
    p_val = np.mean(reps >= ks_observed)
    msg = "***Hypothesis Test***\n" +\
          "Ho: {}.\n\n".format(Hypothesis_test['Ho']) +\
          "Test statistic observed: {:,.5f}. \n({}).\n\n".format(ks_observed, Hypothesis_test['Test statistic']) +\
          'p-value: {:,.5f}. (Probability of "{}").\n'.format(p_val, Hypothesis_test['At least as extreme as']) +\
          "Ho is {}.".format(np.where(p_val<0.01, 'rejected', 'accepted'))
    print(msg)
    
    
    print('----------------------------Using scipy.stats.kstest')
    print('       Hypothesis Test - Data Behaves as exp distrib')
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('Step 1 - State the hypothesis')
    Hypothesis_test = {'Ho'                    : 'The time between Nankai Trough earthquakes is Exponentially distributed ' +\
                                                 'with a mean as calculated from the data', 
                       'Test statistic'        : 'Kolmogorov-Smirnov statistic', 
                       'At least as extreme as': 'Greater than or equal to what was observed'}
    
    print('Step 2 - Define the test statistic')
    print('Step 3 - Generate simulated data')
    print('Step 4 - Compute statistic for simulated data')
    print('Step 5 - Compute the p-value')
    ks_result = stats.kstest(rvs=time_gap, cdf='expon', args=(mean_time_gap,))
    ks_observed, p_val = ks_result.statistic, ks_result.pvalue
    msg = "***Hypothesis Test***\n" +\
          "Ho: {}.\n\n".format(Hypothesis_test['Ho']) +\
          "Test statistic observed: {:,.5f}. \n({}).\n\n".format(ks_observed, Hypothesis_test['Test statistic']) +\
          'p-value: {:,.5f}. (Probability of "{}").\n'.format(p_val, Hypothesis_test['At least as extreme as']) +\
          "Ho is {}.".format(np.where(p_val<0.01, 'rejected', 'accepted'))
    print(msg)
    
    
    print('----------------------------Using scipy.stats.kstest')
    print('      Hypothesis Test - Data Behaves as norm distrib')
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('Step 1 - State the hypothesis')
    Hypothesis_test = {'Ho'                    : 'The time between Nankai Trough earthquakes is Gauss distributed ' +\
                                                 'with a mean and standard deviation as calculated from the data', 
                       'Test statistic'        : 'Kolmogorov-Smirnov statistic', 
                       'At least as extreme as': 'Greater than or equal to what was observed'}
    
    print('Step 2 - Define the test statistic')
    print('Step 3 - Generate simulated data')
    print('Step 4 - Compute statistic for simulated data')
    print('Step 5 - Compute the p-value')
    ks_result = stats.kstest(rvs=time_gap, cdf='norm', args=(mean_time_gap, std_time_gap))
    ks_observed, p_val = ks_result.statistic, ks_result.pvalue
    msg = "***Hypothesis Test***\n" +\
          "Ho: {}.\n\n".format(Hypothesis_test['Ho']) +\
          "Test statistic observed: {:,.5f}. \n({}).\n\n".format(ks_observed, Hypothesis_test['Test statistic']) +\
          'p-value: {:,.5f}. (Probability of "{}").\n'.format(p_val, Hypothesis_test['At least as extreme as']) +\
          "Ho is {}.".format(np.where(p_val<0.01, 'rejected', 'accepted'))
    print(msg)
    
    
    
    print('----------------------------------------------------')
    print('-----------------------Parkfield earthquakes (mag>7)')
    print('----------------------------------------------------')
    time_gap = np.array([24.06570842, 20.07665982, 21.01848049, 12.24640657, 32.05475702, 38.2532512 ])
    mean_time_gap = time_gap.mean()
    std_time_gap = np.std(time_gap)
    
    print('-----------------------------Using my_own_stat_think')
    print('       Hypothesis Test - Data Behaves as exp distrib')
    # Initialize seed and parameters
    np.random.seed(seed) 
    print('Step 1 - State the hypothesis')
    Hypothesis_test = {'Ho'                    : 'The time between Parkfield Trough earthquakes is Exponentially distributed ' +\
                                                 'with a mean as calculated from the data', 
                       'Test statistic'        : 'Kolmogorov-Smirnov statistic', 
                       'At least as extreme as': 'Greater than or equal to what was observed'}
    print('Step 2 - Define the test statistic')
    theorical_model = np.random.exponential(scale=mean_time_gap, size=size)
    ks_observed = most.ks_stat(time_gap, theorical_model)
    print('Step 3 - Generate simulated data')
    print('Step 4 - Compute statistic for simulated data')
    reps = most.draw_ks_reps(len(time_gap), np.random.exponential, args=(mean_time_gap,), 
                             size=size, n_reps=size)
    print('Step 5 - Compute the p-value')
    p_val = np.mean(reps >= ks_observed)
    msg = "***Hypothesis Test***\n" +\
          "Ho: {}.\n\n".format(Hypothesis_test['Ho']) +\
          "Test statistic observed: {:,.5f}. \n({}).\n\n".format(ks_observed, Hypothesis_test['Test statistic']) +\
          'p-value: {:,.5f}. (Probability of "{}").\n'.format(p_val, Hypothesis_test['At least as extreme as']) +\
          "Ho is {}.".format(np.where(p_val<0.01, 'rejected', 'accepted'))
    print(msg)
    
    
    print('-----------------------------Using my_own_stat_think')
    print('     Hypothesis Test - Data Behaves as gauss distrib')
    # Initialize seed and parameters
    np.random.seed(seed) 
    print('Step 1 - State the hypothesis')
    Hypothesis_test = {'Ho'                    : 'The time between Parkfield Trough earthquakes is Gauss distributed ' +\
                                                 'with a mean and standard deviation as calculated from the data', 
                       'Test statistic'        : 'Kolmogorov-Smirnov statistic', 
                       'At least as extreme as': 'Greater than or equal to what was observed'}
    print('Step 2 - Define the test statistic')
    theorical_model = np.random.normal(loc=mean_time_gap, scale=std_time_gap, size=size)
    ks_observed = most.ks_stat(time_gap, theorical_model)
    print('Step 3 - Generate simulated data')
    print('Step 4 - Compute statistic for simulated data')
    reps = most.draw_ks_reps(len(time_gap), np.random.exponential, args=(mean_time_gap,), 
                             size=size, n_reps=size)
    print('Step 5 - Compute the p-value')
    p_val = np.mean(reps >= ks_observed)
    msg = "***Hypothesis Test***\n" +\
          "Ho: {}.\n\n".format(Hypothesis_test['Ho']) +\
          "Test statistic observed: {:,.5f}. \n({}).\n\n".format(ks_observed, Hypothesis_test['Test statistic']) +\
          'p-value: {:,.5f}. (Probability of "{}").\n'.format(p_val, Hypothesis_test['At least as extreme as']) +\
          "Ho is {}.".format(np.where(p_val<0.01, 'rejected', 'accepted'))
    print(msg)
    
    
    print('----------------------------Using scipy.stats.kstest')
    print('       Hypothesis Test - Data Behaves as exp distrib')
    # Initialize seed and parameters
    np.random.seed(seed) 
    print('Step 1 - State the hypothesis')
    Hypothesis_test = {'Ho'                    : 'The time between Parkfield Trough earthquakes is Exponentially distributed ' +\
                                                 'with a mean as calculated from the data', 
                       'Test statistic'        : 'Kolmogorov-Smirnov statistic', 
                       'At least as extreme as': 'Greater than or equal to what was observed'}
    print('Step 2 - Define the test statistic')
    print('Step 3 - Generate simulated data')
    print('Step 4 - Compute statistic for simulated data')
    print('Step 5 - Compute the p-value')
    ks_result = stats.kstest(rvs=time_gap, cdf='expon', args=(mean_time_gap,))
    ks_observed, p_val = ks_result.statistic, ks_result.pvalue
    msg = "***Hypothesis Test***\n" +\
          "Ho: {}.\n\n".format(Hypothesis_test['Ho']) +\
          "Test statistic observed: {:,.5f}. \n({}).\n\n".format(ks_observed, Hypothesis_test['Test statistic']) +\
          'p-value: {:,.5f}. (Probability of "{}").\n'.format(p_val, Hypothesis_test['At least as extreme as']) +\
          "Ho is {}.".format(np.where(p_val<0.01, 'rejected', 'accepted'))
    print(msg)
    
    
    print('----------------------------Using scipy.stats.kstest')
    print('     Hypothesis Test - Data Behaves as gauss distrib')
    # Initialize seed and parameters
    np.random.seed(seed) 
    print('Step 1 - State the hypothesis')
    Hypothesis_test = {'Ho'                    : 'The time between Parkfield Trough earthquakes is Gauss distributed ' +\
                                                 'with a mean and standard devaiation as calculated from the data', 
                       'Test statistic'        : 'Kolmogorov-Smirnov statistic', 
                       'At least as extreme as': 'Greater than or equal to what was observed'}
    print('Step 2 - Define the test statistic')
    print('Step 3 - Generate simulated data')
    print('Step 4 - Compute statistic for simulated data')
    print('Step 5 - Compute the p-value')
    ks_result = stats.kstest(rvs=time_gap, cdf='norm', args=(mean_time_gap, std_time_gap, ))
    ks_observed, p_val = ks_result.statistic, ks_result.pvalue
    msg = "***Hypothesis Test***\n" +\
          "Ho: {}.\n\n".format(Hypothesis_test['Ho']) +\
          "Test statistic observed: {:,.5f}. \n({}).\n\n".format(ks_observed, Hypothesis_test['Test statistic']) +\
          'p-value: {:,.5f}. (Probability of "{}").\n'.format(p_val, Hypothesis_test['At least as extreme as']) +\
          "Ho is {}.".format(np.where(p_val<0.01, 'rejected', 'accepted'))
    print(msg)
    
    
    
def main():
    print("****************************************************")
    print("** BEGIN                                          **")
    
    Statistical_seismology_and_the_Parkfield_region()
    Parkfield_earthquake_magnitudes()
    The_bvalue_for_Parkfield()
    
    Timing_of_major_earthquakes_and_the_Parkfield_sequence()
    Interearthquake_time_estimates_for_Parkfield()
    When_will_the_next_big_Parkfield_quake_be()
    
    How_are_the_Parkfield_interearthquake_times_distributed()
    The_KS_test_for_Exponentiality()

    
    print("****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    pd.reset_option("display.max_columns")
    plt.style.use('default')
    np.set_printoptions(formatter={'float': None})