# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 1: Fish_sleep and bacteria growth: A review of Statistical Thinking I and II
    To begin, you'll use two data sets from Caltech researchers to rehash the key points 
    of Statistical Thinking I and II to prepare you for the following case studies!
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

from collections import OrderedDict

###############################################################################
## Preparing the environment
###############################################################################
#Global variables
suptitle_param = dict(color='darkblue', fontsize=11)
title_param    = {'color': 'darkred', 'fontsize': 14, 'weight': 'bold'}
plot_param     = {'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                  'legend.fontsize': 8, 'font.size': 6}
figsize        = (12.1, 5.9)
SEED           = 42
SIZE           = 10000

# Global configuration
plt.rcParams.update(**plot_param)
np.random.seed(SEED)
sns.set()


###############################################################################
## Reading the data
###############################################################################
zebrafish = pd.read_csv('gandhi_et_al_bouts.csv', skiprows=4)
bacillus_subtilis = pd.read_csv('park_bacterial_growth.csv', skiprows=2)

###############################################################################
## Main part of the code
###############################################################################
def EDA_Plot_ECDFs_of_active_bout_length(seed=SEED, size=SIZE):
    print("****************************************************")
    topic = "2. EDA: Plot ECDFs of active bout length"; print("** %s" % topic)
    print("****************************************************")
    
    bout_lengths_wt = zebrafish[zebrafish.genotype=='wt'].bout_length.values
    bout_lengths_mut = zebrafish[zebrafish.genotype=='mut'].bout_length.values
    
    # Generate x and y values for plotting ECDFs
    x_wt, y_wt = most.ecdf(bout_lengths_wt)
    x_mut, y_mut = most.ecdf(bout_lengths_mut)
    
    # Plot the ECDFs
    fig, ax = plt.subplots()
    ax.plot(x_wt, y_wt, marker='.', linestyle='none')
    ax.plot(x_mut, y_mut, marker='.', linestyle='none')
    
    # Make a legend, label axes, and show plot
    ax.legend(('wt', 'mut'))
    ax.set_xlabel('Active bout length (min)\n[Number of consecutive minutes with activity]')
    ax.set_ylabel('ECDF')
    ax.set_title('Wild Type vs Mutant in Zebrafish', **title_param)
    ax.grid(True)
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=.2, right=None, top=None, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    
    
def Parameter_estimation_active_bout_length(seed=SEED, size=SIZE):
    print("****************************************************")
    topic = "5. Parameter estimation: active bout length"; print("** %s" % topic)
    print("****************************************************")
    
    # From chapter 2
    bout_lengths_wt = zebrafish[zebrafish.genotype=='wt'].bout_length.values
    bout_lengths_mut = zebrafish[zebrafish.genotype=='mut'].bout_length.values
    x_wt, y_wt = most.ecdf(bout_lengths_wt)
    x_mut, y_mut = most.ecdf(bout_lengths_mut)
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Compute mean active bout length
    mean_fish = {}
    mean_fish['wt'] = bout_lengths_wt.mean()
    mean_fish['mut'] = bout_lengths_mut.mean()
    
    # Draw bootstrap replicates
    bs_reps_wt = most.draw_bs_reps(bout_lengths_wt, np.mean, size=size)
    bs_reps_mut = most.draw_bs_reps(bout_lengths_mut, np.mean, size=size)
    
    # Compute 95% confidence intervals
    conf_int = {}
    conf_int['wt'] = np.percentile(bs_reps_wt, [2.5, 97.5])
    conf_int['mut'] = np.percentile(bs_reps_mut, [2.5, 97.5])
    
    # Print the results
    print("""
    wt:  mean = {0:.3f} min., conf. int. = [{1:.1f}, {2:.1f}] min.
    mut: mean = {3:.3f} min., conf. int. = [{4:.1f}, {5:.1f}] min.
    """.format(mean_fish['wt'], *conf_int['wt'], mean_fish['mut'], *conf_int['mut']))
    
    # Plot the ECDFs
    fig, ax = plt.subplots()
    ax.plot(x_wt, y_wt, marker='.', color='blue', linestyle='none', label='Wild Type')
    ax.plot(x_mut, y_mut, marker='.', color='red', linestyle='none', label='Mutant')
    ax.axvline(mean_fish['wt'], color='darkblue', lw=.5, label='Mean Wild Type')
    ax.axvline(mean_fish['mut'], color='darkred', lw=.5, label='Mean Mutant Confidence')
    ax.axvspan(conf_int['wt'][0], conf_int['wt'][1], color='blue', alpha=.3, label='Mean Wild Type Confidence Interval')
    ax.axvspan(conf_int['mut'][0], conf_int['mut'][1], color='red', alpha=.3, label='Mean Mutant Confidence Interval')
    ax.set(xlim=(0, 70), ylim=(0, 1.05)) #To avoid the outlier in the mutant series
    # Make a legend, label axes, and show plot
    ax.legend()
    ax.set_xlabel('Active bout length (min)\n[Number of consecutive minutes with activity]')
    ax.set_ylabel('ECDF')
    ax.set_title('Wild Type vs Mutant in Zebrafish', **title_param)
    ax.grid(True)
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=.2, right=None, top=None, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    
    
def Permutation_and_bootstrap_hypothesis_tests(seed=SEED, size=SIZE):
    print("****************************************************")
    topic = "6. Permutation and bootstrap hypothesis tests"; print("** %s" % topic)
    print("****************************************************")
    
    # From chapter 2
    bout_lengths_wt = zebrafish[zebrafish.genotype=='wt'].bout_length.values
    bout_lengths_mut = zebrafish[zebrafish.genotype=='mut'].bout_length.values
    x_wt, y_wt = most.ecdf(bout_lengths_wt)
    x_mut, y_mut = most.ecdf(bout_lengths_mut)
    
    # From chapter 5
    np.random.seed(seed) 
    mean_fish = {}
    mean_fish['wt'] = bout_lengths_wt.mean()
    mean_fish['mut'] = bout_lengths_mut.mean()
    bs_reps_wt = most.draw_bs_reps(bout_lengths_wt, np.mean, size=size)
    bs_reps_mut = most.draw_bs_reps(bout_lengths_mut, np.mean, size=size)
    conf_int = {}
    conf_int['wt'] = np.percentile(bs_reps_wt, [2.5, 97.5])
    conf_int['mut'] = np.percentile(bs_reps_mut, [2.5, 97.5])
    
    
    print('---------------------------------Interval Confidence')
    # Plot the ECDFs
    fig, ax = plt.subplots()
    for key, color in zip(mean_fish, ['blue','red']):
        ax.hlines(y=key, xmin=conf_int[key][0], xmax=conf_int[key][1], linewidth=3, color=color) 
        ax.scatter(mean_fish[key], key, color=color)
    ax.set_xlabel('Mean Bout Length (min)\n[Number of consecutive minutes with activity]')
    ax.set_ylabel('Fish Species')
    ax.set_ylim(-1,2)
    ax.set_title('Interval Confidence', **title_param)
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=.4, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    
    # Compute 95% confidence intervals
    bout_lengths_het = zebrafish[zebrafish.genotype=='het'].bout_length.values
    mean_fish['het'] = bout_lengths_het.mean()
    conf_int['het'] = np.percentile(most.draw_bs_reps(bout_lengths_het, np.mean, size=size), [2.5, 97.5])
    x_het, y_het = most.ecdf(bout_lengths_het)
    
    fish_ci = zebrafish.groupby('genotype')[['bout_length']].mean()
    # Print the results
    print("""
    het: mean = {0:.3f} min., conf. int. = [{1:.1f}, {2:.1f}] min.
    """.format(mean_fish['het'], *conf_int['het']))
    
    # Compute 95% confidence intervals
    conf_int = OrderedDict(sorted(conf_int.items()))
    fish_ci['lower'] = np.reshape(list(conf_int.values()), (fish_ci.shape[0],2))[:,0]
    fish_ci['upper'] = np.reshape(list(conf_int.values()), (fish_ci.shape[0],2))[:,1]
    fish_ci['color'] = ['green', 'red', 'blue']
    # Plot the ECDFs
    fig, ax = plt.subplots()
    ax.hlines(y=fish_ci.index, xmin=fish_ci.lower, xmax=fish_ci.upper, linewidth=3, color=fish_ci.color) 
    ax.scatter(fish_ci.bout_length, fish_ci.index, color=fish_ci.color)
    ax.set_xlabel('Mean Bout Length (min)\n[Number of consecutive minutes with activity]')
    ax.set_ylabel('Zebrafish Species')
    ax.set_ylim(-1,3)
    ax.set_title('Interval Confidence', **title_param)
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=.2, right=None, top=.8, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    
    print('---------------------------------------------ECDF')
    # Plot the ECDFs
    fig, ax = plt.subplots()
    ax.plot(x_wt, y_wt, marker='.', color='blue', linestyle='none', label='Wild Type')
    ax.plot(x_mut, y_mut, marker='.', color='red', linestyle='none', label='Mutant')
    ax.plot(x_het, y_het, marker='.', color='green', linestyle='none', label='Heterozygote')
    ax.axvline(mean_fish['wt'], color='darkblue', lw=.5, label='Mean Wild Type')
    ax.axvline(mean_fish['mut'], color='darkred', lw=.5, label='Mean Mutant Confidence')
    ax.axvline(mean_fish['het'], color='darkgreen', lw=.5, label='Mean Heterozygote Confidence')
    ax.axvspan(conf_int['wt'][0], conf_int['wt'][1], color='blue', alpha=.3, label='Mean Wild Type Confidence Interval')
    ax.axvspan(conf_int['mut'][0], conf_int['mut'][1], color='red', alpha=.3, label='Mean Mutant Confidence Interval')
    ax.axvspan(conf_int['het'][0], conf_int['het'][1], color='green', alpha=.3, label='Mean Heterozygote Confidence Interval')
    ax.set(xlim=(0, 70), ylim=(0, 1.05)) #To avoid the outlier in the mutant series
    # Make a legend, label axes, and show plot
    ax.legend()
    ax.set_xlabel('Active bout length (min)\n[Number of consecutive minutes with activity]')
    ax.set_ylabel('ECDF')
    ax.set_title('Zebrafish', **title_param)
    ax.grid(True)
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=.2, right=None, top=None, wspace=None, hspace=None); #To set the margins 
    plt.show()
    
    
def Permutation_test_wild_type_versus_heterozygote(seed=SEED, size=SIZE):
    print("****************************************************")
    topic = "7. Permutation test: wild type versus heterozygote"; print("** %s" % topic)
    print("****************************************************")
    
    # From chapter 2
    bout_lengths_wt = zebrafish[zebrafish.genotype=='wt'].bout_length.values
    # From chapter 6
    bout_lengths_het = zebrafish[zebrafish.genotype=='het'].bout_length.values
    
    print('-------------------------------------Hypothesis test')
    print('----------------------------------Using most library')
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Step 1: Clearly state the null hypothesis
    Hypothesis_test = {'Ho'                    : 'The active bout lengths of wild type and heterozygotic fish are identically distributed',
                       'Test statistic'        : 'Difference in mean active bout length between heterozygotes and wild type', 
                       'At least as extreme as': 'Greater than or equal to what was observed'}
    
    # Step 2: Define your test statistic
    diff_means_obs = most.diff_of_means(bout_lengths_het, bout_lengths_wt)
    
    # Step 3: Generate many sets of simulated data assuming the null hypothesis is true
    # Step 4: Compute the test statistic for each simulated data set
    perm_reps = most.draw_perm_reps(bout_lengths_het, bout_lengths_wt, most.diff_of_means, size=size)
    
    # Step 5: The p-value is the fraction of your simulated data sets for which the test statistic is at least as extreme as for the real data
    p_val = np.mean(perm_reps >= diff_means_obs)
    
    msg = "Permutation Test\n" +\
          "Ho: {}.\n\n".format(Hypothesis_test['Ho']) +\
          "Test statistic observed: {:,.4f}. \n({}).\n\n".format(diff_means_obs, Hypothesis_test['Test statistic']) +\
          "p-value: {:,.4f}. {}.\n".format(p_val, Hypothesis_test['At least as extreme as']) +\
          "Ho is {}.".format(np.where(p_val<0.01, 'rejected', 'accepted'))
    print(msg)
    
    #Visualize th p-value to help understanding - Lesson: 20.4.1
    fig, ax = plt.subplots(figsize=figsize)
    most.plotting_test_Ho(ax, perm_reps, diff_means_obs, msg = msg, 
                          x_label='Difference in permutation samples', title='Hypothesis test',
                          params_title=title_param, greater=True)
    plt.subplots_adjust(left=.1, right=.95, bottom=.15, top=.9, hspace=None, wspace=None);
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    
    print('--------------------------------Using lesson 46.4.10')
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Step 1: Clearly state the null hypothesis
    Hypothesis_test = {'Ho'                    : 'The active bout lengths of wild type and heterozygotic fish are identically distributed',
                       'Test statistic'        : 'Difference in mean active bout length between heterozygotes and wild type', 
                       'At least as extreme as': 'Greater than or equal to what was observed'}
    
    # Step 2: Define your test statistic
    diff_means_obs = bout_lengths_het.mean() - bout_lengths_wt.mean()
    
    # Step 3: Generate many sets of simulated data assuming the null hypothesis is true
    data = np.concatenate([bout_lengths_het, bout_lengths_wt])
    perm = np.array([np.random.permutation(data) for i in range(size)])
    permuted_het = perm[:, :len(bout_lengths_het)]
    permuted_wt = perm[:, len(bout_lengths_het):]

    # Step 4: Compute the test statistic for each simulated data set
    perm_reps = np.mean(permuted_het, axis=1) - np.mean(permuted_wt, axis=1)
    
    # Step 5: The p-value is the fraction of your simulated data sets for which the test statistic is at least as extreme as for the real data
    p_val = np.mean(perm_reps >= diff_means_obs)
    
    msg = "Permutation Test\n" +\
          "Ho: {}.\n\n".format(Hypothesis_test['Ho']) +\
          "Test statistic observed: {:,.4f}. \n({}).\n\n".format(diff_means_obs, Hypothesis_test['Test statistic']) +\
          "p-value: {:,.4f}. {}.\n".format(p_val, Hypothesis_test['At least as extreme as']) +\
          "Ho is {}.".format(np.where(p_val<0.01, 'rejected', 'accepted'))
    print(msg)
    
    
    
def Bootstrap_hypothesis_test(seed=SEED, size=SIZE):
    print("****************************************************")
    topic = "8. Bootstrap hypothesis test"; print("** %s" % topic)
    print("****************************************************")
    
    # From chapter 2
    bout_lengths_wt = zebrafish[zebrafish.genotype=='wt'].bout_length.values
    # From chapter 6
    bout_lengths_het = zebrafish[zebrafish.genotype=='het'].bout_length.values
    
    print("The permutation test has a pretty restrictive hypothesis, that the heterozygotic \n" + \
          "and wild type bout lengths are identically distributed. " +\
          "Let's make another aproximation!... \n\n")
    
    print('-------------------------------------Hypothesis test')
    print('----------------------------------Using most library')
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Step 1: Clearly state the null hypothesis
    Hypothesis_test = {'Ho'                    : 'Means bout lengths of wild type and heterozygotic fish are equal \n(making no assumptions about the distributions).',
                       'Test statistic'        : 'Difference in mean active bout length between heterozygotes and wild type', 
                       'At least as extreme as': 'Greater than or equal to what was observed'}
    
    # Step 2: Define your test statistic
    diff_means_obs = most.diff_of_means(bout_lengths_het, bout_lengths_wt)
    
    # Step 3: Generate many sets of simulated data assuming the null hypothesis is true
    #### Concatenate arrays: bout_lengths_concat
    bout_lengths_concat = np.concatenate([bout_lengths_wt, bout_lengths_het])
    #### Compute mean of all bout_lengths: mean_bout_length
    mean_bout_length = bout_lengths_concat.mean()
    #### Generate shifted arrays
    wt_shifted = bout_lengths_wt - bout_lengths_wt.mean() + mean_bout_length
    het_shifted = bout_lengths_het - bout_lengths_het.mean() + mean_bout_length
    #### Compute 10,000 bootstrap replicates from shifted arrays
    bs_reps_wt = most.draw_bs_reps(wt_shifted, np.mean, size)
    bs_reps_het = most.draw_bs_reps(het_shifted, np.mean, size)
    
    # Step 4: Compute the test statistic for each simulated data set
    bs_reps = bs_reps_het - bs_reps_wt

    # Step 5: The p-value is the fraction of your simulated data sets for which the test statistic is at least as extreme as for the real data
    p_val = np.mean(bs_reps >= diff_means_obs)
    
    msg = "Bootstrap Test\n" +\
          "Ho: {}.\n\n".format(Hypothesis_test['Ho']) +\
          "Test statistic observed: {:,.4f}. \n({}).\n\n".format(diff_means_obs, Hypothesis_test['Test statistic']) +\
          "p-value: {:,.4f}. {}.\n".format(p_val, Hypothesis_test['At least as extreme as']) +\
          "Ho is {}.".format(np.where(p_val<0.01, 'rejected', 'accepted'))
    print(msg)
    
    #Visualize th p-value to help understanding - Lesson: 20.4.1
    fig, ax = plt.subplots(figsize=figsize)
    most.plotting_test_Ho(ax, bs_reps, diff_means_obs, msg = msg, 
                          x_label='Difference in permutation samples', title='Hypothesis test',
                          params_title=title_param, greater=True)
    plt.subplots_adjust(left=.1, right=.95, bottom=.15, top=.9, hspace=None, wspace=None);
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    
    print('--------------------------------Using lesson 46.4.10')
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Step 1: Clearly state the null hypothesis
    Hypothesis_test = {'Ho'                    : 'Means bout lengths of wild type and heterozygotic fish are equal \n(making no assumptions about the distributions).',
                       'Test statistic'        : 'Difference in mean active bout length between heterozygotes and wild type', 
                       'At least as extreme as': 'Greater than or equal to what was observed'}
    
    # Step 2: Define your test statistic
    diff_means_obs = most.diff_of_means(bout_lengths_het, bout_lengths_wt)
    
    # Step 3: Generate many sets of simulated data assuming the null hypothesis is true
    #### Concatenate arrays: bout_lengths_concat
    bout_lengths_concat = np.concatenate([bout_lengths_wt, bout_lengths_het])
    #### Compute mean of all bout_lengths: mean_bout_length
    mean_bout_length = bout_lengths_concat.mean()
    #### Generate shifted arrays
    wt_shifted = bout_lengths_wt - bout_lengths_wt.mean() + mean_bout_length
    het_shifted = bout_lengths_het - bout_lengths_het.mean() + mean_bout_length
    #### Compute 10,000 bootstrap replicates from shifted arrays
    bs_reps_wt = most.draw_bs_reps(wt_shifted, np.mean, size)
    bs_reps_het = most.draw_bs_reps(het_shifted, np.mean, size)
    # Make boostraping
    boostraped_wt = np.array([np.random.choice(wt_shifted, size=len(wt_shifted), replace=True) for i in range(size)])
    boostraped_het = np.array([np.random.choice(het_shifted, size=len(het_shifted), replace=True) for i in range(size)])
    
    # Step 4: Compute the test statistic for each simulated data set
    bs_reps = np.mean(boostraped_het, axis=1) - np.mean(boostraped_wt, axis=1)
    
    # Step 5: The p-value is the fraction of your simulated data sets for which the test statistic is at least as extreme as for the real data
    p_val = np.mean(bs_reps >= diff_means_obs)
    
    msg = "Bootstrap Test\n" +\
          "Ho: {}.\n\n".format(Hypothesis_test['Ho']) +\
          "Test statistic observed: {:,.4f}. \n({}).\n\n".format(diff_means_obs, Hypothesis_test['Test statistic']) +\
          "p-value: {:,.4f}. {}.\n".format(p_val, Hypothesis_test['At least as extreme as']) +\
          "Ho is {}.".format(np.where(p_val<0.01, 'rejected', 'accepted'))
    print(msg)
    
    
    
def Linear_regressions_and_pairs_bootstrap(seed=SEED, size=SIZE):
    print("****************************************************")
    topic = "9. Linear regressions and pairs bootstrap"; print("** %s" % topic)
    print("****************************************************")
    
    print('------------------------------------Reading the data')
    t = bacillus_subtilis['time (hr)'].values
    bac_area = bacillus_subtilis['bacterial area (sq. microns)']
    
    print('------------------------------------Bacterial growth')
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    ax = axes[0, 0]
    ax.plot(t, bac_area, marker='.', linestyle='none', ms=1.5)
    ax.set_xlabel('Time (hr)')
    ax.set_ylabel('Area in microns')
    ax.set_title("Bacterial growth", **title_param)
    
    ax = axes[0, 1]
    ax.semilogy(t, bac_area, marker='.', linestyle='none', ms=1.5)
    ax.set_xlabel('Time (hr)')
    ax.set_ylabel('Area in microns')
    ax.set_title("Bacterial growth (log scale)", **title_param)
    
    ax = axes[1, 0]
    # Linear regression
    slope, intercept = np.polyfit(t, bac_area, 1)
    t_theor = np.array([0, 14])
    bac_area_theor = slope * t_theor + intercept
    
    ax.plot(t, bac_area, marker='.', linestyle='none', ms=1.5, label='Recorded Data')
    ax.plot(t_theor, bac_area_theor, lw=1, label='Theorical model')
    ax.set_xlabel('Time (hr)')
    ax.set_ylabel('Area in microns')
    ax.legend()
    ax.set_title("Bacterial growth - Linear Regression", **title_param)
    
    ax = axes[1, 1]
    # Linear regression
    slope, intercept = np.polyfit(t, np.log(bac_area), 1)
    t_theor = np.array([0, 14])
    bac_area_theor = np.exp(slope * t_theor + intercept)
    
    ax.semilogy(t, bac_area, marker='.', linestyle='none', ms=1.5, label='Recorded Data')
    ax.plot(t_theor, bac_area_theor, lw=1, label='Theorical model')
    ax.set_xlabel('Time (hr)')
    ax.set_ylabel('Area in microns')
    ax.legend()
    ax.set_title("Bacterial growth (log scale) - Linear Regression", **title_param)
    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, right=None, bottom=.15, top=None, hspace=.7, wspace=.4);
    plt.show()
    
    
    
def Assessing_the_growth_rate(seed=SEED, size=SIZE):
    print("****************************************************")
    topic = "10. Assessing the growth rate"; print("** %s" % topic)
    print("****************************************************")
    
    print('------------------------------------Reading the data')
    t = bacillus_subtilis['time (hr)'].values
    bac_area = bacillus_subtilis['bacterial area (sq. microns)']
    
    print('------------------------------------Bacterial growth')
    # Compute logarithm of the bacterial area: log_bac_area
    log_bac_area = np.log(bac_area)
    
    # Compute the slope and intercept: growth_rate, log_a0
    growth_rate, log_a0 = np.polyfit(t, log_bac_area, 1)
    
    # Draw 10,000 pairs bootstrap replicates: growth_rate_bs_reps, log_a0_bs_reps
    growth_rate_bs_reps, log_a0_bs_reps = most.draw_bs_pairs_linreg(t, log_bac_area, size=size)
                
    # Compute confidence intervals: growth_rate_conf_int
    growth_rate_conf_int = np.percentile(growth_rate_bs_reps, [2.5, 97.5])
    
    # Print the result to the screen
    msg = "Growth rate: {:.4f} sq. µm/hour.".format(growth_rate)
    print(msg +\
          "\n95% conf int: [{:.4f}, {:.4f}] sq. µm/hour.".format(*growth_rate_conf_int))
    
    #Visualize the ci
    fig, ax = plt.subplots()
    most.plotting_CI(ax, growth_rate_bs_reps, growth_rate, msg, 
                     x_label='Growth rate', title='Bacillys subtilis growth per Hour', params_title=title_param)
    #plt.subplots_adjust(left=.1, right=.95, bottom=.15, top=.9, hspace=None, wspace=None);
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    
    
def Plotting_the_growth_curve(seed=SEED, size=100):
    print("****************************************************")
    topic = "11. Plotting the growth curve"; print("** %s" % topic)
    print("****************************************************")
    
    print('------------------------------------Reading the data')
    t = bacillus_subtilis['time (hr)'].values
    bac_area = bacillus_subtilis['bacterial area (sq. microns)']
    
    print('-------------------------Applying logical regression')
    # Compute logarithm of the bacterial area: log_bac_area
    log_bac_area = np.log(bac_area)
    
    # Compute the slope and intercept: growth_rate, log_a0
    growth_rate, log_a0 = np.polyfit(t, log_bac_area, 1)
    
    print('-----------------------------------------Boostraping')
    # Draw 10,000 pairs bootstrap replicates: growth_rate_bs_reps, log_a0_bs_reps
    growth_rate_bs_reps, log_a0_bs_reps = most.draw_bs_pairs_linreg(t, log_bac_area, size=size)
    
    print('----------------------------------Visualize the data')
    # Plot data points in a semilog-y plot with axis labeles
    fig, ax = plt.subplots()
    ax.semilogy(t, bac_area, marker='.', linestyle='none', label='Recorded Data1')
    
    # Plotting the theorical model
    t_bs = np.array([0, 14])
    bac_area_theor = np.exp(growth_rate * t_bs + log_a0)
    ax.plot(t_bs, bac_area_theor, color='red', lw=.5, label='Theorical model CI')
    
    # Plot the first 100 bootstrap lines
    for i in range(100):
        y = np.exp(growth_rate_bs_reps[i] * t_bs + log_a0_bs_reps[i])
        ax.semilogy(t_bs, y, linewidth=1, alpha=.05, color='red')
    
    ax.set_xlabel('Time (hr)')
    ax.set_ylabel('Area in microns')
    ax.legend()
    ax.set_title("Bacterial growth (log scale) - Linear Regression", **title_param)
    
    plt.show()
    
    
        
def main():
    print("****************************************************")
    print("** BEGIN                                          **")
    
    EDA_Plot_ECDFs_of_active_bout_length()
    
    Parameter_estimation_active_bout_length()
    Permutation_and_bootstrap_hypothesis_tests()
    Permutation_test_wild_type_versus_heterozygote()
    Bootstrap_hypothesis_test()
    Linear_regressions_and_pairs_bootstrap()
    Assessing_the_growth_rate()
    Plotting_the_growth_curve()

    
    print("****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    plt.style.use('default')