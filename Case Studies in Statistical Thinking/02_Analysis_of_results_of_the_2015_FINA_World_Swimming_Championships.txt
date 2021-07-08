# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 2: Analysis of results of the 2015 FINA World Swimming Championships
    In this chapter, you will practice your EDA, parameter estimation, and 
    hypothesis testing skills on the results of the 2015 FINA World Swimming 
    Championships.
Source: https://learn.datacamp.com/courses/case-studies-in-statistical-thinking
Libraries:
    pip install dc_stat_think
    ##C:\Anaconda3\Lib\site-packages\dc_stat_think
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
FINA_2015 = pd.read_csv('2015_FINA.csv', skiprows=4)

###############################################################################
## Main part of the code
###############################################################################
def Introduction_to_swimming_data(seed=SEED, size=SIZE):
    print("****************************************************")
    topic = "1. Introduction to swimming data"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------Explore 2015 FINA World Championships')
    print(FINA_2015.head(2))
    print(FINA_2015.info())
    print(f"Shape: {FINA_2015.shape}.")
    print(f"Gender ({FINA_2015.gender.nunique()}): ", sorted(FINA_2015.gender.unique()))
    print(f"Country ({FINA_2015.code.nunique()})")#": ", sorted(FINA_2015.code.unique()))
    print(f"Events ({FINA_2015.eventid.nunique()})")#: ", sorted(FINA_2015.eventid.unique()))
    print(f"Heats ({FINA_2015.heat.nunique()}): ", sorted(FINA_2015.heat.unique()))
    print(f"Lanes ({FINA_2015.lane.nunique()}): ", sorted(FINA_2015.lane.unique()))
    print(f"Split ({FINA_2015.split.nunique()}): ", sorted(FINA_2015.split.unique()))
    print(f"Split distances ({FINA_2015.splitdistance.nunique()})")#": ", sorted(FINA_2015.splitdistance.unique()))
    print(f"Round ({FINA_2015['round'].nunique()}): ", sorted(FINA_2015['round'].unique()))
    print(f"Distance ({FINA_2015.distance.nunique()}): ", sorted(FINA_2015.distance.unique()))
    print(f"Relay count ({FINA_2015.relaycount.nunique()}): ", sorted(FINA_2015.relaycount.unique()))
    print(f"Stroke ({FINA_2015.stroke.nunique()}): ", sorted(FINA_2015.stroke.unique()))
    #F_FINA_2015 = FINA_2015[FINA_2015.gender=='F']
    #M_FINA_2015 = FINA_2015[FINA_2015.gender=='M']
    
    print('-----------------------------Some visual exploration')
    # Plot the ECDFs
    fig, axes = plt.subplots(1,2, figsize=(12.1, 4))
    
    ax = axes[0]
    cols = ['athleteid', 'gender', 'code', 'eventid', 'heat', 'lane', 'round', 'distance', 'stroke', 'swimtime']
    df_FINA_2015 = FINA_2015[cols].drop_duplicates().dropna()
    sns.scatterplot(x='distance', y='swimtime', hue='gender', data=df_FINA_2015, alpha=.3, ax=ax)
    ax.set_xticks(sorted(df_FINA_2015.distance.unique()))
    ax.tick_params(axis='x', rotation=90)                                         #Set rotation atributte
    ax.set_xlabel('Distance')
    ax.set_ylabel('Swimming time (min.)')
    ax.set_title('2015 FINA World Championships', **title_param)
    
    ax = axes[1]
    condition = (FINA_2015['round']=='PRE') & (FINA_2015.stroke=='FREE') & (FINA_2015.distance==200) & (FINA_2015.gender=='M')
    cols = ['athleteid', 'gender', 'code', 'eventid', 'heat', 'lane', 'round', 'distance', 'stroke', 'swimtime']
    FreeStyle = FINA_2015[condition][cols].drop_duplicates().dropna()
    sns.boxplot(x='gender', y='swimtime', data=FreeStyle, ax=ax)
    ax.set_xlabel('Gender')
    ax.set_ylabel('Swimming time (min.)')
    ax.set_title("200m Freestyle", **title_param)
    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=.2, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    
    
def Graphical_EDA_of_mens_200_free_heats(seed=SEED, size=SIZE):
    print("****************************************************")
    topic = "2. Graphical EDA of men's 200 free heats"; print("** %s" % topic)
    print("****************************************************")
    
    print('------------------------------------Reading the data')
    condition = (FINA_2015['round']=='PRE') & (FINA_2015.stroke=='FREE') & (FINA_2015.distance==200) & (FINA_2015.gender=='M')
    cols = ['athleteid', 'gender', 'code', 'eventid', 'heat', 'lane', 'round', 'distance', 'stroke', 'swimtime']
    mens_200_free_heats = FINA_2015[condition][cols].drop_duplicates().dropna().swimtime.values
    
    print('------------------------------------------------ECDF')
    msg = 'We see that fast swimmers are below 115 seconds, \n' +\
          'with a smattering of slow swimmers past that, \n' +\
          'including one very slow swimmer.'
    print(msg)
    
    # Generate x and y values for ECDF: x, y
    x, y = most.ecdf(mens_200_free_heats)
    
    # Plot the ECDF as dots
    fig, ax = plt.subplots()
    ax.plot(x, y, marker='.', ls='none')
    t = ax.text(0.35, 0.17, msg, transform=ax.transAxes, color='black', ha='left', va='top')
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='darkgray'))
    ax.set_xlabel('times (s)')
    ax.set_ylabel('ECDF')
    ax.set_title("Mens 200m Freestyle", **title_param)
    
    fig.suptitle(topic, **suptitle_param)
    #plt.subplots_adjust(left=None, bottom=.2, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    
    
def n200_m_free_time_with_confidence_interval(seed=SEED, size=SIZE):
    print("****************************************************")
    topic = "3. 200 m free time with confidence interval"; print("** %s" % topic)
    print("****************************************************")
    
    print('------------------------------------Reading the data')
    condition = (FINA_2015['round']=='PRE') & (FINA_2015.stroke=='FREE') & (FINA_2015.distance==200) & (FINA_2015.gender=='M')
    cols = ['athleteid', 'gender', 'code', 'eventid', 'heat', 'lane', 'round', 'distance', 'stroke', 'swimtime']
    mens_200_free_heats = FINA_2015[condition][cols].drop_duplicates().dropna().swimtime.values
    
    print('-----------------Getting the CI from mean and median')
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Compute mean and median swim times
    mean_time = mens_200_free_heats.mean()
    median_time = np.median(mens_200_free_heats)
    
    # Draw 10,000 bootstrap replicates of the mean and median
    bs_reps_mean = most.draw_bs_reps(mens_200_free_heats, np.mean, size)
    bs_reps_median = most.draw_bs_reps(mens_200_free_heats, np.median, size)
    
    # Compute the 95% confidence intervals
    conf_int_mean = np.percentile(bs_reps_mean, [2.5, 92.5])
    conf_int_median = np.percentile(bs_reps_median, [2.5, 92.5])
    
    # Print the result to the screen
    msg = "mean time: {:.2f} sec.\n".format(mean_time) + \
          "95% conf int of mean: [{:.2f}, {:.2f}] sec.\n\n".format(*conf_int_mean) + \
          "median time: {:.2f} sec.\n".format(median_time) + \
          "95% conf int of median: [{:.2f}, {:.2f}] sec.\n\n".format(*conf_int_median) + \
          "The mean swim time is longer than the median because of \nthe effect of the very slow swimmers."
    print(msg)
    
    print('----------------------------------Visualizing the CI')
    # Generate x and y values for ECDF: x, y
    x, y = most.ecdf(mens_200_free_heats)
    
    # Plot the ECDF as dots
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, marker='.', ls='none', label='Swim time')
    ax.axvline(mean_time, color='red')
    ax.axvspan(xmin=conf_int_mean[0], xmax=conf_int_mean[1], color='red', alpha=.2, label='CI Mean')
    ax.axvline(median_time, color='green')
    ax.axvspan(xmin=conf_int_median[0], xmax=conf_int_median[1], color='green', alpha=.2, label='CI Median')
    t = ax.text(0.35, 0.17, msg, transform=ax.transAxes, color='black', ha='left', va='bottom')
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='darkgray'))
    ax.set_xlabel('times (s)')
    ax.set_ylabel('ECDF')
    ax.legend()
    ax.set_title("Mens 200m Freestyle", **title_param)
    
    fig.suptitle(topic, **suptitle_param)
    #plt.subplots_adjust(left=None, bottom=.2, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    
    
def Do_swimmers_go_faster_in_the_finals(seed=SEED, size=SIZE):
    print("****************************************************")
    topic = "4. Do swimmers go faster in the finals?"; print("** %s" % topic)
    print("****************************************************")
    
    print('------------------------------------Reading the data')
    condition = (FINA_2015.gender=='F') & (FINA_2015.distance==200)
    cols = ['athleteid', 'lastname', 'firstname', 'birthdate', 'gender', 'name', 'code', 'eventid', 'heat', 'lane', 'points', 'swimtime', 'round', 'distance', 'stroke']
    fem_short_distance = FINA_2015[condition][cols].drop_duplicates()
    print(fem_short_distance.shape)
    #fem_short_distance.to_csv('fina_fem_200.csv')
    print(fem_short_distance.head(2))
    print(fem_short_distance.info())
    print(f"Stroke ({FINA_2015.stroke.nunique()}): ", sorted(FINA_2015.stroke.unique()))
    print(f"Round ({FINA_2015['round'].nunique()}): ", sorted(FINA_2015['round'].unique()))
    
    print('---------------Visualizing the ECDF for each strokes')
    fsd = {}
    fsd['back']   = fem_short_distance[fem_short_distance.stroke=='BACK'].swimtime.values
    fsd['breast'] = fem_short_distance[fem_short_distance.stroke=='BREAST'].swimtime.values
    fsd['fly']    = fem_short_distance[fem_short_distance.stroke=='FLY'].swimtime.values
    fsd['free']   = fem_short_distance[fem_short_distance.stroke=='FREE'].swimtime.values
    #fsd['medley'] = fem_short_distance[fem_short_distance.stroke=='MEDLEY'].swimtime.values
    
    fig, ax = plt.subplots()
    
    for key in fsd:
        # Generate x and y values for ECDF: x, y
        x, y = most.ecdf(fsd[key])
        ax.plot(x, y, marker='.', ls='none', label=f'{key} style')
    
    ax.set_xlabel('times (s)')
    ax.set_ylabel('ECDF')
    ax.set_title("Female Short Distances", **title_param)
    ax.legend()
    fig.suptitle(topic, **suptitle_param)
    #plt.subplots_adjust(left=None, bottom=.2, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    
    
def EDA_finals_versus_semifinals(seed=SEED, size=SIZE):
    print("****************************************************")
    topic = "5. EDA: finals versus semifinals"; print("** %s" % topic)
    print("****************************************************")
    
    strokes = ['BACK', 'BREAST', 'FLY', 'FREE']
    
    print('------------------------------------Reading the data')
    condition = (FINA_2015['round'].isin(['FIN','SEM'])) & (FINA_2015.gender=='F') & \
                (FINA_2015.stroke.isin(strokes)) & (FINA_2015.distance.isin([50, 100, 200]))
    cols = ['athleteid', 'lastname', 'stroke', 'distance', 'round', 'swimtime']
    fem_short_distance = FINA_2015[condition][cols].drop_duplicates()
    #fem_short_distance.loc[fem_short_distance['round']=='FIN', 'eventid'] = fem_short_distance.loc[fem_short_distance['round']=='FIN', 'eventid']+100 
    fem_short_distance = fem_short_distance.pivot_table(
                            index   = ['stroke', 'distance', 'athleteid', 'lastname'], 
                            columns = 'round',  values  = 'swimtime').dropna()
    
    print('------------------------------Fractional improvement')
    # Compute fractional difference in time between finals and semis
    fem_short_distance['frac_imrpovement'] = (fem_short_distance.SEM - fem_short_distance.FIN) / fem_short_distance.SEM
    print(fem_short_distance)
    
    topic = "6. Parameter estimates of difference between finals and semifinals"; print("** %s" % topic)
    print('--------------------------------------------------CI')
    # Mean fractional time difference: f_mean
    f_mean = fem_short_distance.frac_imrpovement.mean()
    f_median = np.median(fem_short_distance.frac_imrpovement)
    
    # Get bootstrap reps of mean: bs_reps
    bs_reps = most.draw_bs_reps(fem_short_distance.frac_imrpovement, np.mean, size)
    
    # Compute confidence intervals: conf_int
    conf_int = np.percentile(bs_reps, [2.5, 97.5])
    
    # Report
    msg = f"mean frac. diff.: {f_mean:.5f}.\n" +\
          f"median frac. diff.: {f_median:.5f}.\n\n" +\
           "95% conf int of mean frac. diff.: [{0:.5f}, {1:.5f}].".format(*conf_int)
    print(msg)
    
    print('------------------------------------------------ECDF')
    # Generate x and y values for the ECDF: x, y
    x, y = most.ecdf(fem_short_distance.frac_imrpovement.values)
    
    fig, ax = plt.subplots()
    ax.plot(x, y, marker='.', ls='none')
    ax.axvline(f_mean, color='green', label='Average')
    ax.axvline(f_median, color='red', label='Median')
    ax.axhline(.5, color='black', ls='--', label='50% of the data')
    ax.axvspan(conf_int[0], conf_int[1], color='green', alpha=.2, label='CI of the mean')
    ax.set_xlabel('Fractional Improvement')
    ax.set_ylabel('ECDF')
    ax.set_title("Fractional Improvement from Semifinals to Finals in\nFemale Short Distances", **title_param)
    ax.legend()
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.show()   
    
    print('------------Plotting boostrapping of the mean and CI')
    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(f_median, color='red', label='Median')
    most.plotting_CI(ax, bs_reps, f_mean, msg, x_label='Fractional Improvement', 
                     title='Mean fractional improvement from the semifinals to finals,\nalong with a 95% confidence interval of the mean', 
                     params_title=title_param, msg_ci=False)
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.show()   
       
    
    
def How_to_do_the_permutation_test(seed=SEED, size=1000):
    print("****************************************************")
    topic = "7. How to do the permutation test"; print("** %s" % topic)
    print("****************************************************")
    
    print('------------------------------------Reading the data')
    strokes = ['BACK', 'BREAST', 'FLY', 'FREE']
    condition = (FINA_2015['round'].isin(['FIN','SEM'])) & (FINA_2015.gender=='F') & \
                (FINA_2015.stroke.isin(strokes)) & (FINA_2015.distance.isin([50, 100, 200]))
    cols = ['athleteid', 'lastname', 'stroke', 'distance', 'round', 'swimtime']
    fem_short_distance = FINA_2015[condition][cols].drop_duplicates()
    fem_short_distance = fem_short_distance.pivot_table(
                            index   = ['stroke', 'distance', 'athleteid', 'lastname'], 
                            columns = 'round',  values  = 'swimtime').dropna()
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('----------------------------------------------STEP 1')
    # Step 1: Clearly state the null hypothesis
    Hypothesis_test = {'Ho'                    : 'There is no difference in performance between the semifinals and finals',
                       'Test statistic'        : 'Mean value of fractional improvement from semifinals to finals', 
                       'At least as extreme as': 'Greater than or equal to what was observed'}
    
    print('----------------------------------------------STEP 2')
    # Step 2: Define your test statistic
    fem_short_distance['frac_imrpovement'] = (fem_short_distance.SEM - fem_short_distance.FIN) / fem_short_distance.SEM
    f_mean = fem_short_distance.frac_imrpovement.mean()
    
    print('------------------------------------------STEP 3 y 4')
    # Step 3: Generate many sets of simulated data assuming the null hypothesis is true
    # Step 4: Compute the test statistic for each simulated data set
    topic = "8. Generating permutation samples"; print("** %s" % topic)
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
    
    topic = "9. Hypothesis test: Do women swim the same way in semis and finals?"; print("** %s" % topic)
    # Set up array of permutation replicates
    perm_reps = np.zeros(size)
    for i in range(size):
        # Generate a permutation sample
        semi_perm, final_perm = swap_random(fem_short_distance.SEM.values, fem_short_distance.FIN.values)
        # Compute f from the permutation sample
        f = (semi_perm - final_perm) / semi_perm
        # Compute and store permutation replicate
        perm_reps[i] = f.mean()
    
    print('----------------------------------------------STEP 5')
    # Step 5: The p-value is the fraction of your simulated data sets for which the test statistic is at least as extreme as for the real data
    p_val = np.mean(perm_reps >= f_mean)
    
    msg = "Permutation Test\n" +\
          "Ho: {}.\n\n".format(Hypothesis_test['Ho']) +\
          "Test statistic observed: {:,.4f}. \n({}).\n\n".format(f_mean, Hypothesis_test['Test statistic']) +\
          "p-value: {:,.4f}. {}.\n".format(p_val, Hypothesis_test['At least as extreme as']) +\
          "Ho is {}.".format(np.where(p_val<0.01, 'rejected', 'accepted'))
    print(msg)
    
    print('--------------------Visualizing the permutation test')
    #Visualize th p-value to help understanding - Lesson: 20.4.1
    fig, ax = plt.subplots(figsize=figsize)
    most.plotting_test_Ho(ax, perm_reps, f_mean, msg = msg, 
                          x_label='Mean value of fractional improvement from semifinals to finals', 
                          title='Permutation test',
                          params_title=title_param, greater=True)
    plt.subplots_adjust(left=.1, right=.95, bottom=.15, top=.9, hspace=None, wspace=None);
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    
    
def How_does_the_performance_of_swimmers_decline_over_long_events(seed=SEED, size=SIZE):
    print("****************************************************")
    topic = "10. How does the performance of swimmers decline over long events?"; print("** %s" % topic)
    print("****************************************************")
    
    print('------------------------------------Reading the data')
    condition = (FINA_2015.gender=='F') & (FINA_2015.distance==800) &\
                (FINA_2015.lastname.isin(['ASHWOOD','LEDECKY'])) &\
                (FINA_2015['round']=='FIN')
    cols = ['athleteid', 'lastname', 'gender', 'stroke', 'distance', 'round', 'split', 'splitswimtime']
    fem_800_mts = FINA_2015[condition][cols].drop_duplicates()
    print(fem_800_mts.head(2))
    
    print('----------------------------------------Slowing down')
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.lineplot('split', 'splitswimtime', data=fem_800_mts, marker='o', hue='lastname')
    
    ax.set_xlabel('split number')
    ax.set_ylabel('split time (seg)')
    ax.set_title('800 meters swims', **title_param)
    fig.suptitle(topic, **suptitle_param)
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    
    
def EDA_Plot_all_your_data(seed=SEED, size=SIZE):
    print("****************************************************")
    topic = "11. EDA: Plot all your data"; print("** %s" % topic)
    print("****************************************************")
    
    print('----------------------------Reading the data (final)')
    condition = (FINA_2015.gender=='F') & (FINA_2015.distance==800) &\
                (FINA_2015['round'] == 'FIN') & (FINA_2015.split.between(3,14)) 
    cols = ['athleteid', 'lastname', 'gender', 'stroke', 'distance', 'round', 'split', 'splitswimtime']
    fem_800_mts = FINA_2015[condition][cols].drop_duplicates()
    print(fem_800_mts.lastname.unique())
    print(fem_800_mts.head(2))
    
    mean_splits = fem_800_mts.groupby('split').splitswimtime.mean()
    
    print('---------------------------------------------EDA')
    fig, ax = plt.subplots(figsize=figsize)
    # Plot the splits for each swimmer
    sns.lineplot('split', 'splitswimtime', data=fem_800_mts, marker='o', 
                 hue='lastname', palette='GnBu_d', ax=ax)
    
    # Plot the mean split times
    ax.plot(mean_splits.index, mean_splits, color='red', marker='.', linewidth=3, markersize=12, label='Mean split time')
    
    # Label axes and show plot
    ax.set_xlabel('split number')
    ax.set_ylabel('split time (s)')
    ax.legend()
    ax.set_title('800 meters swims final', **title_param)
    fig.suptitle(topic, **suptitle_param)
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    print('-------------------Reading the data (FIRST HEAT)')
    condition = (FINA_2015.gender=='F') & (FINA_2015.distance==800) &\
                (FINA_2015['round'] == 'PRE') & (FINA_2015.split.between(3,14)) 
    cols = ['athleteid', 'lastname', 'gender', 'stroke', 'distance', 'round', 'split', 'splitswimtime']
    fem_800_mts = FINA_2015[condition][cols].drop_duplicates()
    print(fem_800_mts.lastname.unique())
    print(fem_800_mts.head(2))
    
    mean_splits = fem_800_mts.groupby('split').splitswimtime.mean()
    
    print('---------------------------------------------EDA')
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the splits for each swimmer
    sns.lineplot('split', 'splitswimtime', data=fem_800_mts, marker='o', 
                 hue='lastname', palette='GnBu_d', ax=ax)
    
    # Plot the mean split times
    ax.plot(mean_splits.index, mean_splits, color='red', marker='.', linewidth=3, markersize=12, label='Mean split time')
    
    # Label axes and show plot
    ax.set_xlabel('split number')
    ax.set_ylabel('split time (s)')
    ax.legend().set_visible(False)
    ax.set_title('800 meters swims', **title_param)
    fig.suptitle(topic, **suptitle_param)
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    
    
def Linear_regression_of_average_split_time(seed=SEED, size=SIZE):
    print("****************************************************")
    topic = "12. Linear regression of average split time"; print("** %s" % topic)
    print("****************************************************")
    
    print('----------------------------Reading the data (final)')
    condition = (FINA_2015.gender=='F') & (FINA_2015.distance==800) &\
                (FINA_2015['round'] == 'PRE') & (FINA_2015.split.between(3,14)) 
    cols = ['athleteid', 'lastname', 'gender', 'stroke', 'distance', 'round', 'split', 'splitswimtime']
    fem_800_mts = FINA_2015[condition][cols].drop_duplicates()
    fem_800_mts = fem_800_mts.pivot_table(values='splitswimtime', index=cols[:6], columns='split')
    print(fem_800_mts.head(2))
    
    split_number = fem_800_mts.columns.values
    split_mean = fem_800_mts.mean().values
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('-----------------------------------Linear Regression')
    # Perform regression
    slowdown, split_3 = np.polyfit(split_number, split_mean, 1)

    # Compute pairs bootstrap
    bs_slowdown, bs_split_3 = most.draw_bs_pairs_linreg(split_number, split_mean, size=size)
    
    # Compute confidence interval
    ci_slowdown = np.percentile(bs_slowdown, [2.5, 97.5])
    ci_split_3 = np.percentile(bs_split_3, [2.5, 97.5])
    
    # Print the slowdown per split
    msg = f"mean slowdown: {slowdown:.3f} sec./split.\n" +\
           "95% conf int of mean slowdown: [{:.3f}, {:.3f}] sec./split.\n\n".format(*ci_slowdown) +\
          f"split constant: {split_3:.3f} sec./split.\n" +\
           "95% conf int of constant split: [{:.3f}, {:.3f}] sec./split.".format(*ci_split_3)
    print(msg)
    
    print('----------------------------------Visualize the data')
    # Plot the data with regressions line
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(split_number, split_mean, marker='.', linestyle='none', label='Recorded data')
    ax.plot(split_number, slowdown * split_number + split_3, ls='-', color='red', label='Model')
    ax.fill_between(x=split_number, 
                    y1=ci_slowdown[0]*split_number + ci_split_3[0], 
                    y2=ci_slowdown[1]*split_number + ci_split_3[1], color='red', alpha = 0.1) # Set city interval color to desired and lower opacity
    
    # Label axes and show plot
    ax.set_xlabel('split number')
    ax.set_ylabel('split time (s)')
    ax.legend().set_visible(False)
    ax.set_title('Slowdown detectiion in 800 meters swims competitions', **title_param)
    fig.suptitle(topic, **suptitle_param)
    t = ax.text(0.05, 0.95, msg, transform=ax.transAxes, color='black', ha='left', va='top')
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='darkgray'))
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    
    
def Hypothesis_test_are_they_slowing_down(seed=SEED, size=SIZE):
    print("****************************************************")
    topic = "13. Hypothesis test: are they slowing down?"; print("** %s" % topic)
    print("****************************************************")
    
    print('----------------------------Reading the data (final)')
    condition = (FINA_2015.gender=='F') & (FINA_2015.distance==800) &\
                (FINA_2015['round'] == 'PRE') & (FINA_2015.split.between(3,14)) 
    cols = ['athleteid', 'lastname', 'gender', 'stroke', 'distance', 'round', 'split', 'splitswimtime']
    fem_800_mts = FINA_2015[condition][cols].drop_duplicates()
    fem_800_mts = fem_800_mts.pivot_table(values='splitswimtime', index=cols[:6], columns='split')
    
    split_number = fem_800_mts.columns.values
    split_mean = fem_800_mts.mean().values
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('----------------------------------------------STEP 1')
    # Step 1: Clearly state the null hypothesis
    Hypothesis_test = {'Ho'                    : "Swimmer's split time is not at all correlated with the distance they are at in the swim",
                       'Test statistic'        : 'Pearson correlation coefficient', 
                       'At least as extreme as': 'Greater than or equal to what was observed'}
    
    print('----------------------------------------------STEP 2')
    # Step 2: Define your test statistic
    rho = most.pearson_r(split_number, split_mean)
    
    print('------------------------------------------STEP 3 y 4')
    # Step 3: Generate many sets of simulated data assuming the null hypothesis is true
    # Step 4: Compute the test statistic for each simulated data set
    perm_reps_rho = np.zeros(size)
    # Make permutation reps
    for i in range(size):
        # Scramble the split number array
        scrambled_split_number = np.random.permutation(split_number)
        # Compute the Pearson correlation coefficient
        perm_reps_rho[i] = most.pearson_r(scrambled_split_number, split_mean)
    
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
    #Visualize th p-value to help understanding - Lesson: 20.4.1
    fig, ax = plt.subplots(figsize=figsize)
    most.plotting_test_Ho(ax, perm_reps_rho, rho, msg = msg, 
                          x_label='Pearson Correlation in the permutation test', 
                          title='Permutation test',
                          params_title=title_param, greater=True)
    plt.subplots_adjust(left=.1, right=.95, bottom=.15, top=.9, hspace=None, wspace=None);
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    
        
def main():
    print("\n\n****************************************************")
    print("** BEGIN                                          **")
    
    Introduction_to_swimming_data()
    Graphical_EDA_of_mens_200_free_heats()
    n200_m_free_time_with_confidence_interval()
    Do_swimmers_go_faster_in_the_finals()
    EDA_finals_versus_semifinals()
    How_to_do_the_permutation_test()
    How_does_the_performance_of_swimmers_decline_over_long_events()
    EDA_Plot_all_your_data()
    Linear_regression_of_average_split_time()
    Hypothesis_test_are_they_slowing_down()
    
    print("\n\n****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    pd.reset_option("display.max_columns")
    plt.style.use('default')