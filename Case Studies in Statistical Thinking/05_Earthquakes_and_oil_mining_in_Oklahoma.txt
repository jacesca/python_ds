# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 5: Earthquakes and oil mining in Oklahoma
    Of course, earthquakes have a big impact on society, and recently are 
    connected to human activity. In this final chapter, you'll investigate the 
    effect that increased injection of saline wastewater due to oil mining in 
    Oklahoma has had on the seismicity of the region.
Source: https://learn.datacamp.com/courses/case-studies-in-statistical-thinking
"""
###############################################################################
## Importing libraries
###############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
pd.set_option("display.max_columns",24)
plt.rcParams.update(**plot_param)
np.random.seed(SEED)

###############################################################################
## Reading the data
###############################################################################
oklahome = pd.read_csv('oklahoma_earthquakes_1950-2017.csv', skiprows=2, parse_dates=['time'])

###############################################################################
## Main part of the code
###############################################################################
def Variations_in_earthquake_frequency_and_seismicity(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "1. Variations in earthquake frequency and seismicity"; print("** %s" % topic)
    print("****************************************************")
    
    print('----------------------------------Exploring Oklahome')
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection=ccrs.PlateCarree()))
    #ax.set_extent([-103.5, -93.75, 33, 37.5], crs=ccrs.Geodetic(globe=ccrs.Globe(datum='WGS84')))
    ax.set_extent([-103.5, -93.75, 33, 37.5], crs=ccrs.PlateCarree())
    # Put a background image on for nice sea rendering.
    #ax.stock_img()
    ax.add_feature(cfeature.LAND)
    #ax.add_feature(cfeature.OCEAN)
    #ax.add_feature(cfeature.COASTLINE, edgecolor='lightsteelblue')
    #ax.add_feature(cfeature.BORDERS, linestyle=':')
    #ax.add_feature(cfeature.LAKES, alpha=0.5)
    #ax.add_feature(cfeature.RIVERS, alpha=0.5)
    # Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
    states_provinces = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='50m', facecolor='none')
    ax.add_feature(states_provinces, edgecolor='gray')
    
    # Add the Stamen aerial imagery at zoom level 7.
    ax.add_image(Stamen('terrain-background'), 7)
    
    #Plot the earthquake information (longitud, latitud)
    sns.scatterplot(x='longitude', y='latitude', data=oklahome, s=10*(2.5**oklahome.mag.values), color='Red', alpha=.75, legend="brief", ax=ax)
    #ax.legend()
    ax.set_title("Oklahoma city", **title_param)
    fig.suptitle(topic, **suptitle_param)
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    
    
def EDA_Plotting_earthquakes_over_time(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "2. EDA: Plotting earthquakes over time"; print("** %s" % topic)
    print("****************************************************")
    
    print('----------------------------------------Reading data')
    oklahome['year'] = oklahome.time.dt.year + (oklahome.time.dt.dayofyear - 1)/365
    df = oklahome[oklahome.time.between('1980-01-01','2017-06-30')]
    
    print('---------------------------------------------Explore')
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax.set_extent([-103.5, -93.75, 33, 37.5], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.add_image(Stamen('terrain-background'), 7)
    sns.scatterplot(x='longitude', y='latitude', data=df, s=10*(2.5**oklahome.mag.values), color='Red', alpha=.75, legend="brief", ax=ax)
    ax.set_anchor('N')
    ax.set_title("Oklahoma city", **title_param)
    
    ax = fig.add_subplot(1, 2, 2)
    ax.plot('time', 'mag', data=df, marker='.', ls='none', color='red', alpha=.1)
    ax.set_ylim(0,6)
    ax.set_xlabel('Time (year)')
    ax.set_ylabel('Magnitud')
    ax.set_anchor('N')
    ax.set_title("Earthquake between\n1980 and 2017", **title_param)
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=.05, bottom=None, right=.95, top=.85, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    
    
def Estimates_of_the_mean_interearthquake_times(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "3. Estimates of the mean interearthquake times"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('----------------------------------------Reading data')
    oklahome['year'] = oklahome.time.dt.year + (oklahome.time.dt.dayofyear - 1)/365
    df = oklahome[(oklahome.time.between('1980-01-01','2017-06-30')) & (oklahome.mag>=3)].copy()
    df['days_between'] = df['time'].diff().fillna(pd.Timedelta(seconds=0)) / pd.Timedelta(days=1)
    df['segment'] = np.where(df.time.dt.year<2010, 'until 2009','since 2010')
    
    print('-----------------------Finding means in two segments')
    dt_pre = df[df.segment == 'until 2009'].days_between.values[1:]
    dt_post = df[df.segment == 'since 2010'].days_between.values[1:]
    
    # Compute mean interearthquake time
    mean_dt_pre = dt_pre.mean()
    mean_dt_post = dt_post.mean()
    
    # Draw 10,000 bootstrap replicates of the mean
    bs_reps_pre = most.draw_bs_reps(dt_pre, np.mean, size=size)
    bs_reps_post = most.draw_bs_reps(dt_post, np.mean, size=size)
    
    # Compute the confidence interval
    conf_int_pre = np.percentile(bs_reps_pre, [2.5, 97.5])
    conf_int_post = np.percentile(bs_reps_post, [2.5, 97.5])
    
    # Print the results
    msg = "1980 - 2009\n" +\
         f"mean time gap: {mean_dt_pre:.2f} days.\n" +\
          "95% conf int: [{:.2f}, {:.2f}] days.\n\n".format(*conf_int_pre) + \
          "2010 - mid-2017\n" +\
         f"mean time gap: {mean_dt_post:.2f} days.\n" +\
          "95% conf int: [{:.2f}, {:.2f}] days.".format(*conf_int_post)
    print(msg)
    
    print('---------------------------------------------Explore')
    fig = plt.figure(figsize=(12.1, 3.5))
    ax = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax.set_extent([-103.5, -93.75, 33, 37.5], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.add_image(Stamen('terrain-background'), 7)
    sns.scatterplot(x='longitude', y='latitude', data=df, hue='segment', s=10*(2.5**oklahome.mag.values), alpha=.75, legend="brief", ax=ax)
    ax.set_anchor('N')
    ax.set_title("Oklahoma city", **title_param)
    
    ax = fig.add_subplot(1, 2, 2)
    sns.scatterplot(x='days_between', y='mag', data=df, hue='segment', alpha=.5, ax=ax)
    ax.set_ylim(3,6)
    ax.set_xlabel('Days between')
    ax.set_ylabel('Magnitud')
    ax.set_anchor('N')
    ax.set_title("Earthquake between\n1980 and 2017", **title_param)
    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=.05, bottom=.2, right=.95, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    print('------------------------------------------------ECDF')
    fig, ax = plt.subplots()
    
    ax.plot(*most.ecdf(dt_pre), marker='.', ls='none', label='1980-2009')
    ax.plot(*most.ecdf(dt_post), marker='.', ls='none', label='2010-2017')
    
    ax.axvline(mean_dt_pre, color='blue')
    ax.axvline(mean_dt_post, color='orange')
    
    ax.axvspan(*conf_int_pre, color='blue', alpha=.4)
    ax.axvspan(*conf_int_post, color='orange', alpha=.4)
    
    ax.axhline(.75, color='black', ls='--', lw=.5)
    ax.axhline(.5, color='black', ls='--', lw=.5)
    ax.axhline(.25, color='black', ls='--', lw=.5)
    
    t = ax.text(0.5, 0.55, msg, transform=ax.transAxes, color='black', ha='left', va='bottom')
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='darkgray'))
    
    ax.set_xlabel('Days Between')
    ax.set_ylabel('ECDF')
    ax.set_title("Earthquakes detected in the Oklahoma region\nfrom 1850 to 2017", **title_param)
    ax.legend()
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    
    
def Hypothesis_test_did_earthquake_frequency_change(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "4. Hypothesis test: did earthquake frequency change?"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('----------------------------------------Reading data')
    oklahome['year'] = oklahome.time.dt.year + (oklahome.time.dt.dayofyear - 1)/365
    df = oklahome[(oklahome.time.between('1980-01-01','2017-06-30')) & (oklahome.mag>=3)].copy()
    df['days_between'] = df['time'].diff().fillna(pd.Timedelta(seconds=0)) / pd.Timedelta(days=1)
    df['segment'] = np.where(df.time.dt.year<2010, 'until 2009','since 2010')
    
    # Find the two segments
    dt_pre = df[df.segment == 'until 2009'].days_between.values[1:]
    dt_post = df[df.segment == 'since 2010'].days_between.values[1:]
    
    # Compute mean interearthquake time
    mean_dt_pre = dt_pre.mean()
    mean_dt_post = dt_post.mean()
    
    print('-------------------------------------HYPOTHESIS TEST')
    print('Step 1 - State the hypothesis')
    Hypothesis_test = {'Ho'                    : 'Interearthquake times have the same mean before and after 2010', 
                       'Test statistic'        : 'Mean difference', 
                       'At least as extreme as': 'Greater than or equal to what was observed'}
    
    print('Step 2 - Define the test statistic')
    # Compute the observed test statistic
    mean_dt_diff = mean_dt_pre - mean_dt_post
    
    # Shift the post-2010 data to have the same mean as the pre-2010 data
    dt_post_shift = dt_post - mean_dt_post + mean_dt_pre
    
    print('Step 3 - Generate simulated data')
    # Compute 10,000 bootstrap replicates from arrays
    bs_reps_pre = most.draw_bs_reps(dt_pre, np.mean, size)
    bs_reps_post = most.draw_bs_reps(dt_post_shift, np.mean, size)
    
    print('Step 4 - Compute statistic for simulated data')
    # Get replicates of difference of means
    bs_reps = bs_reps_pre - bs_reps_post
    
    print('Step 5 - Compute the p-value')
    # Compute p-value and print the result
    p_val = np.mean(bs_reps >= mean_dt_diff)
    
    msg = "***Hypothesis Test***\n" +\
          "Ho: {}.\n\n".format(Hypothesis_test['Ho']) +\
          "Test statistic observed: {:,.5f}. \n({}).\n\n".format(mean_dt_diff, Hypothesis_test['Test statistic']) +\
          'p-value: {:,.5f}. (Probability of "{}").\n'.format(p_val, Hypothesis_test['At least as extreme as']) +\
          "Ho is {}.".format(np.where(p_val<0.01, 'rejected', 'accepted'))
    print(msg)
    
    print('--------------------Visualizing the permutation test')
    #Visualize th p-value to help understanding - Lesson: 20.4.1
    fig, ax = plt.subplots(figsize=figsize)
    most.plotting_test_Ho(ax, bs_reps, mean_dt_diff, msg = msg, 
                          x_label='Interearthquake times difference', 
                          title='Earthquakes in Oklahoma since 1980 to 2017',
                          params_title=title_param, greater=True)
    plt.subplots_adjust(left=.1, right=.95, bottom=.15, top=.9, hspace=None, wspace=None);
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    
def EDA_Comparing_magnitudes_before_and_after_2010(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "7. EDA: Comparing magnitudes before and after 2010"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('----------------------------------------Reading data')
    oklahome['year'] = oklahome.time.dt.year + (oklahome.time.dt.dayofyear - 1)/365
    df = oklahome[oklahome.time.between('1980-01-01','2017-06-30')].copy()
    df['segment'] = np.where(df.time.dt.year<2010, 'until 2009','since 2010')
    
    # Get magnitudes before and after 2010
    mags_pre = df[df.segment == 'until 2009'].mag.values
    mags_post = df[df.segment == 'since 2010'].mag.values
    
    # Compute mean interearthquake time
    mean_mags_pre = mags_pre.mean()
    mean_mags_post = mags_post.mean()
    
    # Draw 10,000 bootstrap replicates of the mean
    bs_reps_pre = most.draw_bs_reps(mags_pre, np.mean, size=size)
    bs_reps_post = most.draw_bs_reps(mags_post, np.mean, size=size)
    
    # Compute the confidence interval
    conf_int_pre = np.percentile(bs_reps_pre, [2.5, 97.5])
    conf_int_post = np.percentile(bs_reps_post, [2.5, 97.5])
    
    # Print the results
    msg = "1980 - 2009\n" +\
         f"mean magnitud: {mean_mags_pre:.2f}.\n" +\
          "95% conf int: [{:.2f}, {:.2f}] days.\n\n".format(*conf_int_pre) + \
          "2010 - mid-2017\n" +\
         f"mean magnitud: {mean_mags_post:.2f}.\n" +\
          "95% conf int: [{:.2f}, {:.2f}] days.".format(*conf_int_post)
    print(msg)
    
    print('------------------------------------------------ECDF')
    fig, ax = plt.subplots()
    
    ax.plot(*most.ecdf(mags_pre), marker='.', ls='none', label='1980 though 2009')
    ax.plot(*most.ecdf(mags_post), marker='.', ls='none', label='2010 through mid-2017')
    
    ax.axvline(mean_mags_pre, color='blue')
    ax.axvline(mean_mags_post, color='orange')
    
    ax.axvspan(*conf_int_pre, color='blue', alpha=.4)
    ax.axvspan(*conf_int_post, color='orange', alpha=.4)
    
    ax.axhline(.75, color='black', ls='--', lw=.5)
    ax.axhline(.5, color='black', ls='--', lw=.5)
    ax.axhline(.25, color='black', ls='--', lw=.5)
    
    t = ax.text(0.05, 0.7, msg, transform=ax.transAxes, color='black', ha='left', va='bottom')
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='darkgray'))
    
    ax.set_xlabel('Magnitude')
    ax.set_ylabel('ECDF')
    ax.set_title("Earthquakes detected in the Oklahoma region\nfrom 1850 to 2017", **title_param)
    ax.legend(loc='lower right')
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    
    
def Quantification_of_the_bvalues(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "8. Quantification of the b-values"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('----------------------------------------Reading data')
    oklahome['year'] = oklahome.time.dt.year + (oklahome.time.dt.dayofyear - 1)/365
    df = oklahome[oklahome.time.between('1980-01-01','2017-06-30')].copy()
    df['segment'] = np.where(df.time.dt.year<2010, 'until 2009','since 2010')
    
    # Get magnitudes before and after 2010
    mags_pre = df[df.segment == 'until 2009'].mag.values
    mags_post = df[df.segment == 'since 2010'].mag.values
    
    print('---------------------------------------------b-value')
    mt = 3
    # Compute b-value and confidence interval for pre-2010
    b_pre, conf_int_pre = most.b_value(mags_pre, mt, n_reps=size)
    
    # Compute b-value and confidence interval for post-2010
    b_post, conf_int_post = most.b_value(mags_post, mt, n_reps=size)
    
    # Report the results
    msg = "1980 through 2009\n" +\
         f"b-value: {b_pre:.2f}.\n" +\
          "95% conf int: [{:.2f}, {:.2f}].\n\n".format(*conf_int_pre) +\
          "2010 through mid-2017\n" +\
         f"b-value: {b_post:.2f}.\n" +\
          "95% conf int: [{:.2f}, {:.2f}].".format(*conf_int_post)
    print(msg)
    
    print('-----------------------Theorical model - Exponential')
    # Generate samples to for theoretical ECDF
    m_theor_pre = np.random.exponential(b_pre/np.log(10), size=size) + mt
    m_theor_post = np.random.exponential(b_post/np.log(10), size=size) + mt
    
    print('------------------------------------------------ECDF')
    fig, ax = plt.subplots()
    
    # Plot the CDF
    ax.plot(*most.ecdf(mags_post[mags_post>=mt]), color='orange', marker='.', ls='none', ms=4, label='2010 through mid-2017')
    ax.plot(*most.ecdf(m_theor_post), color='orange', lw=.75)
    ax.plot(*most.ecdf(mags_pre[mags_pre>=mt]), color='blue', marker='.', ls='none', ms=1, label='1980 through 2009')
    ax.plot(*most.ecdf(m_theor_pre), color='blue', lw=.5)
    
    t = ax.text(0.4, 0.4, msg, transform=ax.transAxes, color='black', ha='left', va='bottom')
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='darkgray'))
    
    ax.set_xlabel('Magnitude')
    ax.set_ylabel('ECDF')
    ax.set_title("Earthquakes detected in the Oklahome region\nfrom 1980 to mid-2017 - (Magnitud above 3)", **title_param)
    ax.legend(loc='lower right')
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    
    
def Hypothesis_test_are_the_bvalues_different(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "10. Hypothesis test: are the b-values different?"; print("** %s" % topic)
    print("****************************************************")
    
    print('----------------------------------------Reading data')
    mt = 3
    oklahome['year'] = oklahome.time.dt.year + (oklahome.time.dt.dayofyear - 1)/365
    df = oklahome[oklahome.time.between('1980-01-01','2017-06-30')].copy()
    df['segment'] = np.where(df.time.dt.year<2010, 'until 2009','since 2010')
    
    # Get magnitudes before and after 2010
    mags_pre = df[(df.segment == 'until 2009') & (df.mag>=mt)].mag.values
    mags_post = df[(df.segment == 'since 2010') & (df.mag>=mt)].mag.values
    
    print('-------------------------------------HYPOTHESIS TEST')
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('Step 1 - State the hypothesis')
    Hypothesis_test = {'Ho'                    : 'The b-value in Oklahoma from 1980 through 2009 is the same\nthat from 2010 through mid-2017', 
                       'Test statistic'        : 'Difference in mean magnitudes', 
                       'At least as extreme as': 'Smaller than to what was observed'}
    
    print('Step 2 - Define the test statistic')
    # Observed difference in mean magnitudes: diff_obs
    diff_obs = mags_post.mean() - mags_pre.mean()
    
    print('Step 3 - Generate simulated data')
    print('Step 4 - Compute statistic for simulated data')
    # Generate permutation replicates: perm_reps
    perm_reps = most.draw_perm_reps(mags_post, mags_pre, most.diff_of_means, size=size)
    
    print('Step 5 - Compute the p-value')
    # Compute p-value and print the result
    p_val = np.mean(perm_reps < diff_obs)
    
    msg = "***Hypothesis Test***\n" +\
          "Ho: {}.\n\n".format(Hypothesis_test['Ho']) +\
          "Test statistic observed: {:,.5f}. \n({}).\n\n".format(diff_obs, Hypothesis_test['Test statistic']) +\
          'p-value: {:,.5f}. (Probability of "{}").\n'.format(p_val, Hypothesis_test['At least as extreme as']) +\
          "Ho is {}.".format(np.where(p_val<0.01, 'rejected', 'accepted'))
    print(msg)
    
    print('--------------------Visualizing the permutation test')
    #Visualize th p-value to help understanding - Lesson: 20.4.1
    fig, ax = plt.subplots(figsize=figsize)
    most.plotting_test_Ho(ax, perm_reps, diff_obs, msg = msg, 
                          x_label='Difference in mean magnitude', 
                          title='Earthquakes in Oklahoma since 1980 to 2017',
                          params_title=title_param, greater=False)
    plt.subplots_adjust(left=.1, right=.95, bottom=.15, top=.9, hspace=None, wspace=None);
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    
    
def main():
    print("****************************************************")
    print("** BEGIN                                          **")
    
    Variations_in_earthquake_frequency_and_seismicity()
    EDA_Plotting_earthquakes_over_time()
    Estimates_of_the_mean_interearthquake_times()
    Hypothesis_test_did_earthquake_frequency_change()
    
    EDA_Comparing_magnitudes_before_and_after_2010()
    Quantification_of_the_bvalues()
    Hypothesis_test_are_the_bvalues_different()
    
    print("****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    pd.reset_option("display.max_columns")
    plt.style.use('default')
    np.set_printoptions(formatter={'float': None})