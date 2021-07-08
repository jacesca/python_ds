# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 3: The "Current Controversy" of the 2013 World Championships
    Some swimmers said that they felt it was easier to swim in one direction 
    versus another in the 2013 World Championships. Some analysts have posited 
    that there was a swirling current in the pool. In this chapter, you'll 
    investigate this claim! 
References:
    ⦁ Quartz Media - https://qz.com/761280/researchers-believe-certain-lanes-in-the-olympic-pool-may-have-given-some-swimmers-an-advantage/
    ⦁ Washington Post - https://www.washingtonpost.com/news/wonk/wp/2016/09/01/these-charts-clearly-show-how-some-olympic-swimmers-may-have-gotten-an-unfair-advantage/?utm_term=.dba907006ba1
    ⦁ SwimSwam  - https://swimswam.com/rio-olympic-test-event-showed-same-pool-bias-2-0/ 
      (and also here - https://swimswam.com/problem-rio-pool/)
    ⦁ Cornett, et al - https://www.ncbi.nlm.nih.gov/pubmed/25003776
Source: https://learn.datacamp.com/courses/case-studies-in-statistical-thinking
"""
###############################################################################
## Importing libraries
###############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
FINA_2013 = pd.read_csv('2013_FINA.csv', skiprows=4)
FINA_2015 = pd.read_csv('2015_FINA.csv', skiprows=4)

###############################################################################
## Main part of the code
###############################################################################
def Introduction_to_the_current_controversy(seed=SEED, size=SIZE):
    print("****************************************************")
    topic = "1. Introduction to the current controversy"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------Explore 2015 FINA World Championships')
    print(FINA_2013.head(2))
    print(FINA_2013.info())
    print(f"Shape: {FINA_2013.shape}.")
    print(f"Gender ({FINA_2013.gender.nunique()}): ", sorted(FINA_2013.gender.unique()))
    print(f"Country ({FINA_2013.code.nunique()})")#": ", sorted(FINA_2015.code.unique()))
    print(f"Events ({FINA_2013.eventid.nunique()})")#: ", sorted(FINA_2015.eventid.unique()))
    print(f"Heats ({FINA_2013.heat.nunique()}): ", sorted(FINA_2013.heat.unique()))
    print(f"Lanes ({FINA_2013.lane.nunique()}): ", sorted(FINA_2013.lane.unique()))
    print(f"Split ({FINA_2013.split.nunique()}): ", sorted(FINA_2013.split.unique()))
    print(f"Split distances ({FINA_2013.splitdistance.nunique()})")#": ", sorted(FINA_2015.splitdistance.unique()))
    print(f"Round ({FINA_2013['round'].nunique()}): ", sorted(FINA_2013['round'].unique()))
    print(f"Distance ({FINA_2013.distance.nunique()}): ", sorted(FINA_2013.distance.unique()))
    print(f"Relay count ({FINA_2013.relaycount.nunique()}): ", sorted(FINA_2013.relaycount.unique()))
    print(f"Stroke ({FINA_2013.stroke.nunique()}): ", sorted(FINA_2013.stroke.unique()))
    
    
    
def ECDF_of_improvement_from_low_to_high_lanes(seed=SEED, size=SIZE):
    print("****************************************************")
    topic = "3. ECDF of improvement from low to high lanes"; print("** %s" % topic)
    print("****************************************************")
    
    print('----------------------------Reading the data (final)')
    condition = (FINA_2013.distance==50) &\
                (FINA_2013.stroke.isin(['BACK', 'BREAST', 'FLY', 'FREE'])) &\
                (FINA_2013['round'].isin(['SEM','FIN'])) &\
                (FINA_2013['lane'].isin([1,2,3,5,6,7,8,9])) 
    cols = ['athleteid', 'lastname', 'gender', 'stroke', 'distance', 'round', 'lane', 'swimtime']
    competition_50_mts = FINA_2013[condition][cols].drop_duplicates()
    competition_50_mts['lane'] = np.where(competition_50_mts.lane.values<5, 'LOW', 'HIGH')
    competition_50_mts = competition_50_mts.pivot_table(values='swimtime', index=cols[:5], columns='lane').dropna()
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------------------Fractional improvement')
    competition_50_mts['frac'] = (competition_50_mts.LOW - competition_50_mts.HIGH) / competition_50_mts.LOW
    print(competition_50_mts, '\n\n')
    
    # Mean fractional improvement
    f_mean = competition_50_mts.frac.mean()
    f_median = np.median(competition_50_mts.frac)
    msg1 = f"Mean frac. improvement: {f_mean:.5f}\n" +\
           f"Median frac. improvement: {f_median:.5f}"
    print(msg1)
    
    topic = "4. Estimation of mean improvement"; print("\n** %s" % topic)
    print('--------------------------------------------------CI')
    # Get bootstrap reps of mean: bs_reps
    bs_reps = most.draw_bs_reps(competition_50_mts.frac, np.mean, size)
    
    # Compute confidence intervals: conf_int
    conf_int = np.percentile(bs_reps, [2.5, 97.5])
    
    # Report
    msg2 = "95% conf int of mean frac. improvement:\n[{0:.5f}, {1:.5f}].".format(*conf_int)
    print(msg2)
    
    print('------------------------------------------------ECDF')
    # Make x and y values for ECDF: x, y
    x, y = most.ecdf(competition_50_mts.frac)
    
    # Plot the ECDFs as dots
    fig, ax = plt.subplots()
    ax.plot(x, y, marker='.', ls='none')
    ax.axvline(f_mean, color='green', label='Average')
    ax.axvline(f_median, color='red', label='Median')
    ax.axhline(.5, color='black', ls='--', label='50% of the data')
    ax.axvspan(conf_int[0], conf_int[1], color='green', alpha=.2, label='CI of the mean')
    t = ax.text(0.4, 0.55, msg1+'\n\n'+msg2, transform=ax.transAxes, color='black', ha='left', va='bottom')
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='darkgray'))
    ax.set_xlabel('Fractional Improvement')
    ax.set_ylabel('ECDF')
    ax.set_title("FINA 2013\nFractional Improvement from Low Lanes to\nHigh Lanes in 50 mts competition", **title_param)
    ax.legend()
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.show()   
    
    print('------------Plotting boostrapping of the mean and CI')
    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(f_median, color='red', label='Median')
    most.plotting_CI(ax, bs_reps, f_mean, msg1+'\n\n'+msg2, x_label='Fractional Improvement', 
                     title='FINA 2013\nMean fractional improvement from the low to high lanes,\nin a 50 mts competition along with a 95% CI of the mean', 
                     params_title=title_param, msg_ci=False)
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.show()   
    
        
    
def Hypothesis_test_Does_lane_assignment_affect_performance(seed=SEED, size=100000):
    print("****************************************************")
    topic = "6. Hypothesis test: Does lane assignment affect performance?"; print("** %s" % topic)
    print("****************************************************")
    
    print('----------------------------Reading the data (final)')
    condition = (FINA_2013.distance==50) &\
                (FINA_2013.stroke.isin(['BACK', 'BREAST', 'FLY', 'FREE'])) &\
                (FINA_2013['round'].isin(['SEM','FIN'])) &\
                (FINA_2013['lane'].isin([1,2,3,5,6,7,8,9])) 
    cols = ['athleteid', 'lastname', 'gender', 'stroke', 'distance', 'round', 'lane', 'swimtime']
    competition_50_mts = FINA_2013[condition][cols].drop_duplicates()
    competition_50_mts['lane'] = np.where(competition_50_mts.lane.values<5, 'LOW', 'HIGH')
    competition_50_mts = competition_50_mts.pivot_table(values='swimtime', index=cols[:5], columns='lane').dropna()
    
    print('\n\n-----------HYPOTHESIS No.1--------------------------')
    print('-----------NO MEAN IMPROVEMENT FROM LOW TO HGH LANES')
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('----------------------------------------------STEP 1')
    # Step 1: Clearly state the null hypothesis
    Hypothesis_test = {'Ho'                    : 'Mean fractional improvement going from low-numbered lanes to \nhigh-numbered lanes is zero',
                       'Test statistic'        : 'Fractional improvement from low to high numbered lanes', 
                       'At least as extreme as': 'Greater than or equal to what was observed'}
    
    print('----------------------------------------------STEP 2')
    # Step 2: Define your test statistic
    competition_50_mts['frac'] = (competition_50_mts.LOW - competition_50_mts.HIGH) / competition_50_mts.LOW
    f = competition_50_mts.frac.values
    # Mean fractional improvement
    f_mean = f.mean()
    print(f"{f_mean:,.5f}")
    
    # Shift f: f_shift
    f_shift = f - f.mean()
    
    #print('------------------------------------------STEP 3 y 4')
    ## Step 3: Generate many sets of simulated data assuming the null hypothesis is true
    ## Step 4: Compute the test statistic for each simulated data set
    #bs_reps = most.draw_bs_reps(f_shift, np.mean, size)
    
    print('----------------------------------------------STEP 3') #Lesson 46
    # Step 3: Generate many sets of simulated data assuming the null hypothesis is true
    bs_reps = np.array([np.random.choice(f_shift, size=len(f_shift), replace=True) for i in range(size)])
        
    print('----------------------------------------------STEP 4')
    # Step 4: Compute the test statistic for each simulated data set
    bs_reps = np.mean(bs_reps, axis=1)
    
    print('----------------------------------------------STEP 5')
    # Step 5: The p-value is the fraction of your simulated data sets for which the test statistic is at least as extreme as for the real data
    p_val = np.mean(bs_reps >= f_mean)
    
    msg = "Hypothesis Test 1\n" +\
          "Ho: {}.\n\n".format(Hypothesis_test['Ho']) +\
          "Test statistic observed: {:,.4f}. \n({}).\n\n".format(f_mean, Hypothesis_test['Test statistic']) +\
          "p-value: {:,.5f}. {}.\n".format(p_val, Hypothesis_test['At least as extreme as']) +\
          "Ho is {}.".format(np.where(p_val<0.01, 'rejected', 'accepted'))
    print(msg)
    
    print('--------------------Visualizing the permutation test')
    #Visualize th p-value to help understanding - Lesson: 20.4.1
    fig, ax = plt.subplots(figsize=figsize)
    most.plotting_test_Ho(ax, bs_reps, f_mean, msg = msg, 
                          x_label='Fractional improvement from low to high lanes', 
                          title='FINA 2013 - Hypothesis 1\nWhether a swimmer is in a high-numbered lane or a low-numbered lane\nhas no bearing on the swim time',
                          params_title=title_param, greater=True)
    plt.subplots_adjust(left=.1, right=.95, bottom=.15, top=.8, hspace=None, wspace=None);
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    
    print('\n\n----HYPOTHESIS No.2---------------------------------')
    print('----LANE NUMBER HAS NO BEARING AT ALL ON PERFORMANCE')
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('----------------------------------------------STEP 1')
    # Step 1: Clearly state the null hypothesis
    Hypothesis_test = {'Ho'                    : 'Lane number has no bearing at all on performance\n' +\
                                                 '(In other words, there is no difference in performance between high and low lanes)',
                       'Test statistic'        : 'Fractional improvement from low to high numbered lanes', 
                       'At least as extreme as': 'Greater than or equal to what was observed'}
    
    print('----------------------------------------------STEP 2')
    # Step 2: Define your test statistic
    swimtime_low_lanes = competition_50_mts.LOW.values
    swimtime_high_lanes = competition_50_mts.HIGH.values
    
    # Mean fractional improvement
    f = (swimtime_low_lanes - swimtime_high_lanes) / swimtime_low_lanes
    f_mean = f.mean()
    print(f"{f_mean:,.5f}")
    
    #print('------------------------------------------STEP 3 y 4')
    ## Step 3: Generate many sets of simulated data assuming the null hypothesis is true
    ## Step 4: Compute the test statistic for each simulated data set
    def swap_random(a, b):
        """Randomly swap entries in two arrays."""
        # Indices to swap
        swap_inds = np.random.random(size=len(a)) < .5
        # Make copies of arrays a and b for output
        a_out = np.copy(a)
        b_out = np.copy(b)
        # Swap values
        a_out[swap_inds] = b[swap_inds]
        b_out[swap_inds] = a[swap_inds]
        return a_out, b_out
    
    # Set up array of permutation replicates
    bs_reps = np.zeros(size)
    for i in range(size):
        # Generate a permutation sample
        low_lanes, high_lanes = swap_random(swimtime_low_lanes, swimtime_high_lanes)
        # Compute f from the permutation sample
        f = (low_lanes - high_lanes) / low_lanes
        # Compute and store permutation replicate
        bs_reps[i] = f.mean()
    
    print('----------------------------------------------STEP 5')
    # Step 5: The p-value is the fraction of your simulated data sets for which the test statistic is at least as extreme as for the real data
    p_val = np.mean(bs_reps >= f_mean)
    
    msg = "Hypothesis Test 2\n" +\
          "Ho: {}.\n\n".format(Hypothesis_test['Ho']) +\
          "Test statistic observed: {:,.4f}. \n({}).\n\n".format(f_mean, Hypothesis_test['Test statistic']) +\
          "p-value: {:,.5f}. {}.\n".format(p_val, Hypothesis_test['At least as extreme as']) +\
          "Ho is {}.".format(np.where(p_val<0.01, 'rejected', 'accepted'))
    print(msg)
    
    print('--------------------Visualizing the permutation test')
    #Visualize th p-value to help understanding - Lesson: 20.4.1
    fig, ax = plt.subplots(figsize=figsize)
    most.plotting_test_Ho(ax, bs_reps, f_mean, msg = msg, 
                          x_label='Fractional improvement from low to high lanes', 
                          title='FINA 2013 - Hypothesis 2\nLane number has no bearing at all on performance',
                          params_title=title_param, greater=True)
    plt.subplots_adjust(left=.1, right=.95, bottom=.15, top=.8, hspace=None, wspace=None);
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    
    
def Did_the_2015_event_have_this_problem(seed=SEED, size=100000):
    print("****************************************************")
    topic = "7. Did the 2015 event have this problem?"; print("** %s" % topic)
    print("****************************************************")
    
    print('----------------FINA 2015 - Reading the data (final)')
    condition = (FINA_2015.distance==50) &\
                (FINA_2015.stroke.isin(['BACK', 'BREAST', 'FLY', 'FREE'])) &\
                (FINA_2015['round'].isin(['SEM','FIN'])) &\
                (FINA_2015['lane'].isin([1,2,3,5,6,7,8,9])) 
    cols = ['athleteid', 'lastname', 'gender', 'stroke', 'distance', 'round', 'lane', 'swimtime']
    competition_50_mts = FINA_2015[condition][cols].drop_duplicates()
    competition_50_mts['lane'] = np.where(competition_50_mts.lane.values<5, 'LOW', 'HIGH')
    competition_50_mts = competition_50_mts.pivot_table(values='swimtime', index=cols[:5], columns='lane').dropna()
    
    print('------------------FINA 2015 - Fractional improvement')
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    competition_50_mts['frac'] = (competition_50_mts.LOW - competition_50_mts.HIGH) / competition_50_mts.LOW
    print(competition_50_mts, '\n\n')
    
    # Mean fractional improvement
    f_mean = competition_50_mts.frac.mean()
    f_median = np.median(competition_50_mts.frac)
    msg1 = f"Mean frac. improvement: {f_mean:.5f}\n" +\
           f"Median frac. improvement: {f_median:.5f}"
    print(msg1)
    
    print('--------------------------------------FINA 2015 - CI')
    # Get bootstrap reps of mean: bs_reps
    bs_reps = most.draw_bs_reps(competition_50_mts.frac, np.mean, size)
    
    # Compute confidence intervals: conf_int
    conf_int = np.percentile(bs_reps, [2.5, 97.5])
    
    # Report
    msg2 = "95% conf int of mean frac. improvement:\n[{0:.5f}, {1:.5f}].".format(*conf_int)
    print(msg2)
    
    print('------------------------------------FINA 2015 - ECDF')
    # Make x and y values for ECDF: x, y
    x, y = most.ecdf(competition_50_mts.frac)
    
    # Plot the ECDFs as dots
    fig, ax = plt.subplots()
    ax.plot(x, y, marker='.', ls='none')
    ax.axvline(f_mean, color='green', label='Average')
    ax.axvline(f_median, color='red', label='Median')
    ax.axhline(.5, color='black', ls='--', label='50% of the data')
    ax.axvspan(conf_int[0], conf_int[1], color='green', alpha=.2, label='CI of the mean')
    t = ax.text(0.4, 0.55, msg1+'\n\n'+msg2, transform=ax.transAxes, color='black', ha='left', va='bottom')
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='darkgray'))
    ax.set_xlabel('Fractional Improvement')
    ax.set_ylabel('ECDF')
    ax.set_title("FINA 2015\nFractional Improvement from Low Lanes to\nHigh Lanes in 50 mts competition", **title_param)
    ax.legend()
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.show()   
    
    print('---------FINA 2015 - Boostrapping of the mean and CI')
    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(f_median, color='red', label='Median')
    most.plotting_CI(ax, bs_reps, f_mean, msg1+'\n\n'+msg2, x_label='Fractional Improvement', 
                     title='FINA 2015\nMean fractional improvement from the low to high lanes,\nin a 50 mts competition along with a 95% CI of the mean', 
                     params_title=title_param, msg_ci=False)
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.show()   
    
        
    print('\n\n-----------FINA 2015 - HYPOTHESIS No.1--------------')
    print('-----------NO MEAN IMPROVEMENT FROM LOW TO HGH LANES')
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('----------------------------------------------STEP 1')
    # Step 1: Clearly state the null hypothesis
    Hypothesis_test = {'Ho'                    : 'Mean fractional improvement going from low-numbered lanes to \nhigh-numbered lanes is zero',
                       'Test statistic'        : 'Fractional improvement from low to high numbered lanes', 
                       'At least as extreme as': 'Greater than or equal to what was observed'}
    
    print('----------------------------------------------STEP 2')
    # Step 2: Define your test statistic
    competition_50_mts['frac'] = (competition_50_mts.LOW - competition_50_mts.HIGH) / competition_50_mts.LOW
    f = competition_50_mts.frac.values
    # Mean fractional improvement
    f_mean = f.mean()
    print(f"{f_mean:,.5f}")
    
    # Shift f: f_shift
    f_shift = f - f.mean()
    
    #print('------------------------------------------STEP 3 y 4')
    ## Step 3: Generate many sets of simulated data assuming the null hypothesis is true
    ## Step 4: Compute the test statistic for each simulated data set
    #bs_reps = most.draw_bs_reps(f_shift, np.mean, size)
    
    print('----------------------------------------------STEP 3') #Lesson 46
    # Step 3: Generate many sets of simulated data assuming the null hypothesis is true
    bs_reps = np.array([np.random.choice(f_shift, size=len(f_shift), replace=True) for i in range(size)])
        
    print('----------------------------------------------STEP 4')
    # Step 4: Compute the test statistic for each simulated data set
    bs_reps = np.mean(bs_reps, axis=1)
    
    print('----------------------------------------------STEP 5')
    # Step 5: The p-value is the fraction of your simulated data sets for which the test statistic is at least as extreme as for the real data
    p_val = np.mean(bs_reps >= f_mean)
    
    msg = "Hypothesis Test 1\n" +\
          "Ho: {}.\n\n".format(Hypothesis_test['Ho']) +\
          "Test statistic observed: {:,.4f}. \n({}).\n\n".format(f_mean, Hypothesis_test['Test statistic']) +\
          "p-value: {:,.5f}. {}.\n".format(p_val, Hypothesis_test['At least as extreme as']) +\
          "Ho is {}.".format(np.where(p_val<0.01, 'rejected', 'accepted'))
    print(msg)
    
    print('--------------------Visualizing the permutation test')
    #Visualize th p-value to help understanding - Lesson: 20.4.1
    fig, ax = plt.subplots(figsize=figsize)
    most.plotting_test_Ho(ax, bs_reps, f_mean, msg = msg, 
                          x_label='Fractional improvement from low to high lanes', 
                          title='FINA 2015 - Hypothesis 1\nWhether a swimmer is in a high-numbered lane or a low-numbered lane\nhas no bearing on the swim time',
                          params_title=title_param, greater=True)
    plt.subplots_adjust(left=.1, right=.95, bottom=.15, top=.8, hspace=None, wspace=None);
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    
    print('\n\n----FINA 2015 - HYPOTHESIS No.2---------------------')
    print('----LANE NUMBER HAS NO BEARING AT ALL ON PERFORMANCE')
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('----------------------------------------------STEP 1')
    # Step 1: Clearly state the null hypothesis
    Hypothesis_test = {'Ho'                    : 'Lane number has no bearing at all on performance\n' +\
                                                 '(In other words, there is no difference in performance between high and low lanes)',
                       'Test statistic'        : 'Fractional improvement from low to high numbered lanes', 
                       'At least as extreme as': 'Greater than or equal to what was observed'}
    
    print('----------------------------------------------STEP 2')
    # Step 2: Define your test statistic
    swimtime_low_lanes = competition_50_mts.LOW.values
    swimtime_high_lanes = competition_50_mts.HIGH.values
    
    # Mean fractional improvement
    f = (swimtime_low_lanes - swimtime_high_lanes) / swimtime_low_lanes
    f_mean = f.mean()
    print(f"{f_mean:,.5f}")
    
    #print('------------------------------------------STEP 3 y 4')
    ## Step 3: Generate many sets of simulated data assuming the null hypothesis is true
    ## Step 4: Compute the test statistic for each simulated data set
    def swap_random(a, b):
        """Randomly swap entries in two arrays."""
        # Indices to swap
        swap_inds = np.random.random(size=len(a)) < .5
        # Make copies of arrays a and b for output
        a_out = np.copy(a)
        b_out = np.copy(b)
        # Swap values
        a_out[swap_inds] = b[swap_inds]
        b_out[swap_inds] = a[swap_inds]
        return a_out, b_out
    
    # Set up array of permutation replicates
    bs_reps = np.zeros(size)
    for i in range(size):
        # Generate a permutation sample
        low_lanes, high_lanes = swap_random(swimtime_low_lanes, swimtime_high_lanes)
        # Compute f from the permutation sample
        f = (low_lanes - high_lanes) / low_lanes
        # Compute and store permutation replicate
        bs_reps[i] = f.mean()
    
    print('----------------------------------------------STEP 5')
    # Step 5: The p-value is the fraction of your simulated data sets for which the test statistic is at least as extreme as for the real data
    p_val = np.mean(bs_reps >= f_mean)
    
    msg = "Hypothesis Test 2\n" +\
          "Ho: {}.\n\n".format(Hypothesis_test['Ho']) +\
          "Test statistic observed: {:,.4f}. \n({}).\n\n".format(f_mean, Hypothesis_test['Test statistic']) +\
          "p-value: {:,.5f}. {}.\n".format(p_val, Hypothesis_test['At least as extreme as']) +\
          "Ho is {}.".format(np.where(p_val<0.01, 'rejected', 'accepted'))
    print(msg)
    
    print('--------------------Visualizing the permutation test')
    #Visualize th p-value to help understanding - Lesson: 20.4.1
    fig, ax = plt.subplots(figsize=figsize)
    most.plotting_test_Ho(ax, bs_reps, f_mean, msg = msg, 
                          x_label='Fractional improvement from low to high lanes', 
                          title='FINA 2015 - Hypothesis 2\nLane number has no bearing at all on performance',
                          params_title=title_param, greater=True)
    plt.subplots_adjust(left=.1, right=.95, bottom=.15, top=.8, hspace=None, wspace=None);
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    
    
def The_zigzag_effect(seed=SEED, size=SIZE):
    print("****************************************************")
    topic = "8. The zigzag effect"; print("** %s" % topic)
    print("****************************************************")
    
    print('------------------------FINA 2013 - Reading the data')
    condition = (FINA_2013.distance==1500) &\
                (FINA_2013.gender=='F') &\
                (FINA_2013.stroke=='FREE') &\
                (FINA_2013['round']=='FIN') &\
                (FINA_2013['lane'].isin([1,2,3,5,6,7,8,9])) 
    cols = ['athleteid', 'lastname', 'gender', 'stroke', 'distance', 'round', 'lane', 'split', 'splitswimtime']
    c2013_50_mts = FINA_2013[condition][cols].drop_duplicates()
    c2013_50_mts['lane'] = np.where(c2013_50_mts.lane.values<5, 'FINA 2013 - LOW', 'FINA 2013 - HIGH')
    
    print('------------------------FINA 2015 - Reading the data')
    condition = (FINA_2015.distance==1500) &\
                (FINA_2015.gender=='F') &\
                (FINA_2015.stroke=='FREE') &\
                (FINA_2015['round']=='FIN') &\
                (FINA_2015['lane'].isin([1,2,3,5,6,7,8,9])) 
    cols = ['athleteid', 'lastname', 'gender', 'stroke', 'distance', 'round', 'lane', 'split', 'splitswimtime']
    c2015_50_mts = FINA_2015[condition][cols].drop_duplicates()
    c2015_50_mts['lane'] = np.where(c2015_50_mts.lane.values<5, 'FINA 2015 - LOW', 'FINA 2015 - HIGH')
    
    
    print('----------------------Extract the needed information')
    mean_splits_2013 = c2013_50_mts.groupby(['lane', 'split']).splitswimtime.mean().reset_index()
    mean_splits_2015 = c2015_50_mts.groupby(['lane', 'split']).splitswimtime.mean().reset_index()
    
    
    print('-------------------------------------------------EDA')
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the splits for low/high lanes
    sns.lineplot('split', 'splitswimtime', data=mean_splits_2013, marker='o', hue='lane', palette='Blues', ax=ax)
    sns.lineplot('split', 'splitswimtime', data=mean_splits_2015, marker='o', hue='lane', palette='Reds', ax=ax)
    
    # Label axes and show plot
    ax.set_xlabel('split number')
    ax.set_ylabel('split time (s)')
    ax.set_title('FINA 2013 vs FINA 2015\n1500 meters distance competition', **title_param)
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    
def EDA_mean_differences_between_odd_and_even_splits(seed=SEED, size=SIZE):
    print("****************************************************")
    topic = "10. EDA: mean differences between odd and even splits"; print("** %s" % topic)
    print("****************************************************")
    
    print('------------------------FINA 2013 - Reading the data')
    condition = (FINA_2013.distance==1500) &\
                (FINA_2013.gender=='F') &\
                (FINA_2013.stroke=='FREE') &\
                (FINA_2013['round']=='FIN') &\
                (FINA_2013.lane.between(1,8)) &\
                (FINA_2013.split.between(3,14))
    cols = ['lane', 'athleteid', 'lastname', 'gender', 'stroke', 'distance', 'round', 'split', 'splitswimtime']
    c2013_1500_mts = FINA_2013[condition][cols].drop_duplicates()
    c2013_1500_mts['split'] = np.where(c2013_1500_mts.split%2==0, 'EVEN', 'ODD')
    
    c2013_1500_mts = c2013_1500_mts.pivot_table(values='splitswimtime', index=cols[:7], columns='split', aggfunc='mean').reset_index()
    c2013_1500_mts['fractional_difference'] = 2*(c2013_1500_mts.ODD - c2013_1500_mts.EVEN)/(c2013_1500_mts.ODD + c2013_1500_mts.EVEN)
    print(c2013_1500_mts)
    
    print('------------------------FINA 2015 - Reading the data')
    condition = (FINA_2015.distance==1500) &\
                (FINA_2015.gender=='F') &\
                (FINA_2015.stroke=='FREE') &\
                (FINA_2015['round']=='FIN') &\
                (FINA_2015.lane.between(1,8)) &\
                (FINA_2015.split.between(3,14))
    cols = ['lane', 'athleteid', 'lastname', 'gender', 'stroke', 'distance', 'round', 'split', 'splitswimtime']
    c2015_1500_mts = FINA_2015[condition][cols].drop_duplicates()
    c2015_1500_mts['split'] = np.where(c2015_1500_mts.split%2==0, 'EVEN', 'ODD')
    
    c2015_1500_mts = c2015_1500_mts.pivot_table(values='splitswimtime', index=cols[:7], columns='split', aggfunc='mean').reset_index()
    c2015_1500_mts['fractional_difference'] = 2*(c2015_1500_mts.ODD - c2015_1500_mts.EVEN)/(c2015_1500_mts.ODD + c2015_1500_mts.EVEN)
    print(c2015_1500_mts)
    
    print('-------------------------------------------------EDA')
    slope_2013, intercept_2013 = np.polyfit(c2013_1500_mts.lane.values, c2013_1500_mts.fractional_difference.values, 1)
    slope_2015, intercept_2015 = np.polyfit(c2015_1500_mts.lane.values, c2015_1500_mts.fractional_difference.values, 1)
    
    x_2013, y_2013 = c2013_1500_mts.lane.values, c2013_1500_mts.lane.values*slope_2013 + intercept_2013
    x_2015, y_2015 = c2015_1500_mts.lane.values, c2015_1500_mts.lane.values*slope_2015 + intercept_2015
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the splits for low/high lanes
    sns.scatterplot('lane', 'fractional_difference', data=c2013_1500_mts, color='blue', marker='o', label='2013', ax=ax)
    sns.scatterplot('lane', 'fractional_difference', data=c2015_1500_mts, color='red',  marker='o', label='2015', ax=ax)
    
    ax.plot(x_2013, y_2013, color='blue')
    ax.plot(x_2015, y_2015, color='red')
    
    plt.axhline(0, ls='--', lw=.5, color='black')
    t = ax.text(0.1, 0.9, "EDA has exposed a strong slope in 2013 compared to 2015!", transform=ax.transAxes, color='black', ha='left', va='top')
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='darkgray'))
    # Label axes and show plot
    ax.set_xlabel('lane')
    ax.set_ylabel('frac. diff. (odd - even)')
    ax.set_title('FINA 2013 vs FINA 2015\n1500 meters distance competition - Fractional difference in Split times', **title_param)
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    
    
def How_does_the_current_effect_depend_on_lane_position(seed=SEED, size=SIZE):
    print("****************************************************")
    topic = "11. How does the current effect depend on lane position?"; print("** %s" % topic)
    print("****************************************************")
    
    print('------------------------FINA 2013 - Reading the data')
    condition = (FINA_2013.distance==1500) &\
                (FINA_2013.gender=='F') &\
                (FINA_2013.stroke=='FREE') &\
                (FINA_2013['round']=='FIN') &\
                (FINA_2013.lane.between(1,8)) &\
                (FINA_2013.split.between(3,14))
    cols = ['lane', 'athleteid', 'lastname', 'gender', 'stroke', 'distance', 'round', 'split', 'splitswimtime']
    c2013_1500_mts = FINA_2013[condition][cols].drop_duplicates()
    c2013_1500_mts['split'] = np.where(c2013_1500_mts.split%2==0, 'EVEN', 'ODD')
    
    c2013_1500_mts = c2013_1500_mts.pivot_table(values='splitswimtime', index=cols[:7], columns='split', aggfunc='mean').reset_index()
    c2013_1500_mts['fractional_difference'] = 2*(c2013_1500_mts.ODD - c2013_1500_mts.EVEN)/(c2013_1500_mts.ODD + c2013_1500_mts.EVEN)
    print(c2013_1500_mts)
    
    lanes = c2013_1500_mts.lane.values
    f_13   = c2013_1500_mts.fractional_difference.values
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('-----------------------------------Linear Regression')
    # Perform regression
    slope, intercept = np.polyfit(lanes, f_13, 1)
    
    print('---------------------------------Confidence Interval')
    # Compute bootstrap replicates
    bs_reps_slope, bs_reps_int = most.draw_bs_pairs_linreg(lanes, f_13, size=size)
    
    # Compute confidence interval
    ci_slope = np.percentile(bs_reps_slope, [2.5, 97.5])
    ci_intercept = np.percentile(bs_reps_int, [2.5, 97.5])
    
    # Print the slowdown per split
    msg = f"Slope: {slope:.5f}.\n" +\
           "95% conf int of slope: [{:.5f}, {:.5f}].\n\n".format(*ci_slope) +\
          f"Intercept: {intercept:.5f}.\n" +\
           "95% conf int of intercept: [{:.5f}, {:.5f}].".format(*ci_intercept)
    print(msg)
    
    print('----------------------------------Visualize the data')
    msg = f"{msg}\n\n" +\
          f"The slope is a fractional difference of about {slope:.1%} per lane.\n" +\
           "This is quite a substantial difference at this elite level of\n" +\
           "swimming where races can be decided by tiny differences."
    # Plot the data with regressions line
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(lanes, f_13, color='blue', marker='o')
    
    # x-values for plotting regression lines
    x = np.array([1,8])
    
    # Plot 100 bootstrap replicate lines
    for i in range(100):
        ax.plot(x, bs_reps_slope[i] * x + bs_reps_int[i], color='red', alpha=0.2, linewidth=0.5)
    
    ax.set_title('2013 FINA 1500 mts distance competition\nFractional Difference in split times', **title_param)
    fig.suptitle(topic, **suptitle_param)
    t = ax.text(0.5, 0.05, msg, transform=ax.transAxes, color='black', ha='left', va='bottom')
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='darkgray'))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    
    
def Hypothesis_test_can_this_be_by_chance(seed=SEED, size=SIZE):
    print("****************************************************")
    topic = "12. Hypothesis test: can this be by chance?"; print("** %s" % topic)
    print("****************************************************")
    
    print('------------------------FINA 2013 - Reading the data')
    condition = (FINA_2013.distance==1500) &\
                (FINA_2013.gender=='F') &\
                (FINA_2013.stroke=='FREE') &\
                (FINA_2013['round']=='FIN') &\
                (FINA_2013.lane.between(1,8)) &\
                (FINA_2013.split.between(3,14))
    cols = ['lane', 'athleteid', 'lastname', 'gender', 'stroke', 'distance', 'round', 'split', 'splitswimtime']
    c2013_1500_mts = FINA_2013[condition][cols].drop_duplicates()
    c2013_1500_mts['split'] = np.where(c2013_1500_mts.split%2==0, 'EVEN', 'ODD')
    
    c2013_1500_mts = c2013_1500_mts.pivot_table(values='splitswimtime', index=cols[:7], columns='split', aggfunc='mean').reset_index()
    c2013_1500_mts['fractional_difference'] = 2*(c2013_1500_mts.ODD - c2013_1500_mts.EVEN)/(c2013_1500_mts.ODD + c2013_1500_mts.EVEN)
    #c2013_1500_mts = c2013_1500_mts.groupby('lane')[['fractional_difference']].mean()
    print(c2013_1500_mts)
    
    lanes = c2013_1500_mts.lane.values
    f_13   = c2013_1500_mts.fractional_difference.values
    
    print('\n\nFINA 2013 - HYPOTHESIS-------------------------------')
    print('LANE ASSIGNMENT HAS NO BEARING AT FRAC.DIFF IN SPLITS')
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('----------------------------------------------STEP 1')
    # Step 1: Clearly state the null hypothesis
    Hypothesis_test = {'Ho'                    : 'Lane assignment has nothing to do with the mean fractional \ndifference between even and odd splits',
                       'Test statistic'        : 'Fractional difference in split times', 
                       'At least as extreme as': 'Greater than or equal to what was observed'}
    
    print('----------------------------------------------STEP 2')
    # Step 2: Define your test statistic
    rho = most.pearson_r(lanes, f_13)
    
    print('------------------------------------------STEP 3 y 4')
    # Step 3: Generate many sets of simulated data assuming the null hypothesis is true
    # Step 4: Compute the test statistic for each simulated data set
    
    # Initialize permutation reps: perm_reps_rho
    perm_reps_rho = np.zeros(size)
    # Make permutation reps
    for i in range(size):
        # Scramble the lanes array: scrambled_lanes
        scrambled_lanes = np.random.permutation(lanes)
        # Compute the Pearson correlation coefficient
        perm_reps_rho[i] = most.pearson_r(scrambled_lanes, f_13)
    
    print('----------------------------------------------STEP 5')
    # Step 5: The p-value is the fraction of your simulated data sets for which the test statistic is at least as extreme as for the real data
    p_val = np.mean(perm_reps_rho >= rho)

    msg = "Hypothesis Test\n" +\
          "Ho: {}.\n\n".format(Hypothesis_test['Ho']) +\
          "Test statistic observed: {:,.4f}. \n({}).\n\n".format(rho, Hypothesis_test['Test statistic']) +\
          "p-value: {:,.4f}. {}.\n".format(p_val, Hypothesis_test['At least as extreme as']) +\
          "Ho is {}.".format(np.where(p_val<0.01, 'rejected', 'accepted'))
    print(msg)
    
    print('--------------------Visualizing the permutation test')
    fig, ax = plt.subplots(figsize=figsize)
    most.plotting_test_Ho(ax, perm_reps_rho, rho, msg = msg, 
                          x_label='Pearson Correlation in the permutation test', 
                          title='Permutation test',
                          params_title=title_param, greater=True)
    plt.subplots_adjust(left=.1, right=.95, bottom=.15, top=.9, hspace=None, wspace=None);
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    
def Recap_of_swimming_analysis(seed=SEED, size=SIZE):
    print("****************************************************")
    topic = "13. Recap of swimming analysis"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Explore')
    print('---------------------------------------------Explore')
    
    
        
def main():
    print("****************************************************")
    print("** BEGIN                                          **")
    
    Introduction_to_the_current_controversy()
    ECDF_of_improvement_from_low_to_high_lanes()
    Hypothesis_test_Does_lane_assignment_affect_performance()
    Did_the_2015_event_have_this_problem()
    The_zigzag_effect()
    EDA_mean_differences_between_odd_and_even_splits()
    How_does_the_current_effect_depend_on_lane_position()
    Hypothesis_test_can_this_be_by_chance()
    Recap_of_swimming_analysis()
    
    print("****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    pd.reset_option("display.max_columns")
    plt.style.use('default')