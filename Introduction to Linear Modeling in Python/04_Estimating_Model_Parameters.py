# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 03:10:21 2020

@author: jacesca@gmail.com
Subject: Practicing Statistics Interview Questions in Python
Chapter 4: Estimating Model Parameters
    In our final chapter, we introduce concepts from inferential statistics, 
    and use them to explore how maximum likelihood estimation and bootstrap 
    resampling can be used to estimate linear model parameters. We then apply 
    these methods to make probabilistic statements about our confidence in the 
    model parameters. 
"""

###############################################################################
## Importing libraries
###############################################################################
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from statsmodels.formula.api import ols #Create a Model from a formula and dataframe.


###############################################################################
## Preparing the environment
###############################################################################
SEED = 42
np.random.seed(SEED) 
    
def gaussian_model(x, mu, sigma):
    """Define gaussian model function"""
    coeff_part = 1/(np.sqrt(2 * np.pi * sigma**2))
    exp_part = np.exp( - (x - mu)**2 / (2 * sigma**2) )
    return coeff_part*exp_part
    
def compute_loglikelihood(data, mu, sigma):
    """Compute loglikelihood of a guessed mu"""
    probs = np.zeros(len(data))
    for n, elem in enumerate(data):
        probs[n] = gaussian_model(elem, mu, sigma)
    return(np.sum(np.log(probs)))
          
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    n = len(data) # Number of data points: n
    x = np.sort(data) # x-data for the ECDF: x
    y = np.arange(1, n+1) / n # y-data for the ECDF: y
    return x, y

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    corr_mat = np.corrcoef(x, y) # Compute correlation matrix: corr_mat
    return corr_mat[0,1]         # Return entry [0,1]

def bootstrap_replicate_1d(data, func, replace=True):
    """Generate bootstrap replicate of 1D data."""
    return func(np.random.choice(data, size=len(data), replace=replace))

def draw_bs_reps(data, func, size=1, replace=True):
    """Draw bootstrap replicates."""
    bs_replicates = np.zeros(size) # Initialize array of replicates: bs_replicates
    for i in range(size): # Generate replicates
        bs_replicates[i] = bootstrap_replicate_1d(data, func, replace)
    return bs_replicates

def draw_bs_pairs_linreg(x, y, size=1, replace=True):
    """Perform pairs bootstrap for linear regression."""
    inds = np.arange(len(x)) # Set up array of indices to sample from: inds
    bs_slope_reps = np.zeros(size) # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_intercept_reps = np.zeros(size)
    for i in range(size): # Generate replicates
        bs_inds = np.random.choice(inds, size=len(inds), replace=replace)
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1)
    return bs_slope_reps, bs_intercept_reps 

def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""
    diff = data_1.mean()-data_2.mean() # The difference of means of data_1, data_2: diff
    return diff

def diff_before_get_means(data_1, data_2):
    """Difference in means of two arrays."""
    size_sample = len(data_1) if len(data_1)<len(data_2) else len(data_2)
    diff = data_1[:size_sample]-data_2[:size_sample] # The difference of means of data_1, data_2: diff
    return diff.mean()

def bootstrap_replicate_2d(data1, data2, func, size=1, replace=True):
    """Perform pairs bootstrap for linear regression."""
    bs_func = np.zeros(size) # Initialize replicates: bs_slope_reps, bs_intercept_reps
    for i in range(size): # Generate replicates
        bs_data1 = np.random.choice(data1, size=len(data1), replace=replace)
        bs_data2 = np.random.choice(data2, size=len(data2), replace=replace)
        bs_func[i] = func(bs_data1, bs_data2)
        #print(bs_func[i], 'Test Statistic: mean={:0.2f}, stdev={:0.2f}'.format(bs_func.mean(), bs_func.std()))
    return bs_func 
    
def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""
    data = np.concatenate((data1, data2)) # Concatenate the data sets: data
    permuted_data = np.random.permutation(data) # Permute the concatenated array: permuted_data
    perm_sample_1 = permuted_data[:len(data1)] # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_2 = permuted_data[len(data1):]
    return perm_sample_1, perm_sample_2

def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""
    perm_replicates = np.zeros(size) # Initialize array of replicates: perm_replicates
    for i in range(size):
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2) # Generate permutation sample
        perm_replicates[i] = func(perm_sample_1, perm_sample_2) # Compute the test statistic
    return perm_replicates

def plotting_boostraps_ecdf(ax, sample, bs_sample, x_label='', title='', params_title={}):
    """Visualizing bootstrap samples."""
    #Getting the ecdf
    x_theor, y_theor = ecdf(sample)
    x, y = ecdf(bs_sample)
    
    ax.plot(x_theor, y_theor, label='population model')
    ax.plot(x, y, marker='.', alpha=0.7, linestyle='none', label='boostrap sample')
    ax.axvline(np.mean(sample), ls='--', color='black', label='estimated mean')
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.legend()
    ax.set_xlabel(x_label)
    ax.set_ylabel('CDF')
    ax.set_title("Boostrap samples" + title, **params_title)
    ax.grid(True)    
   
def plotting_boostraps_samples(ax, data, x_label='', title='', params_title={}, replace=True, seed=SEED):
    """To visualize bootstrap samples"""
    np.random.seed(seed) # Seed random number generator
    
    size = len(data)
    for i in range(25):
        # Generate bootstrap sample: bs_sample
        bs_sample = np.random.choice(data, size=size, replace=True)
        
        # Compute and plot ECDF from bootstrap sample
        x, y = ecdf(bs_sample)
        ax.plot(x, y, marker='.', linestyle='none', color='gray', alpha=0.1)
        #ax.axvline(np.mean(bs_sample), ls='-', lw=.5, color='gray', alpha=1)

    # Compute and plot ECDF from original data
    x, y = ecdf(data)
    ax.plot(x, y, marker='.', ms=.25)
    #ax.axvline(np.mean(data), ls='--', lw=.5, color='red')
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.set_xlabel(x_label)
    ax.set_ylabel('ECDF')
    ax.set_title("Boostrap samples" + title, **params_title)
    ax.grid(True) 
    
def plotting_boostraps_hist(ax, bs_sample, 
                            x_label='', title='', params_title={}, bins=None,
                            pos_x_conf_legend=0.03, pos_y_conf_legend=0.97):
    """Plot the histogram of the replicates and its confidence interval."""
    # Compute the 95% confidence interval: conf_int
    conf_int = np.percentile(bs_sample, [2.5, 97.5])
    
    # add a 'best fit' line
    mu = bs_sample.mean()
    sigma = bs_sample.std()
    
    # Plot the histogram of the replicates
    ax.hist(bs_sample, bins=bins, rwidth=.9, density=True, label='boostrap sample')
    best_fit = gaussian_model(bs_sample, mu, sigma)
    ax.plot(bs_sample, best_fit, linestyle=" ", ms=1, marker="o", color='darkgreen')
    ax.axvline(bs_sample.mean(), color='red', linestyle='dashed', linewidth=2, label='boostrap mean')
    ax.axvspan(conf_int[0], conf_int[1], color='grey', alpha=0.2, label='confidence interval')
    ax.set_xlabel(x_label)
    ax.set_ylabel('PDF')
    t = ax.text(pos_x_conf_legend, pos_y_conf_legend, 
                "95% confidence\ninterval:\n[{:,.2f}, {:,.2f}].".format(conf_int[0],conf_int[1]), 
                transform=ax.transAxes, color='black', ha='left', va='top')
    t.set_bbox(dict(facecolor='antiquewhite', alpha=0.8, edgecolor='antiquewhite'))
    ax.set_title("Mean expected (Boostrap samples)" + title, **params_title)
    ax.legend(loc='upper right')
    ax.grid(True)    
    
def plotting_test_Ho(ax, bs_sample, effect_size, ho, test_for, conclusion,
                     x_label='', title='', params_title={}, bins=20, significance=0.01,
                     pos_x_conf_legend=0.03, pos_y_conf_legend=0.97):
    """Plot the histogram of the replicates and its confidence interval."""
    # Compute the 95% confidence interval: conf_int
    conf_int = np.percentile(bs_sample, [2.5, 97.5])
    
    # add a 'best fit' line
    mu = bs_sample.mean()
    sigma = bs_sample.std()
    p = np.sum(bs_sample >= effect_size)*1.0 / len(bs_sample)
    
    msg = "95% confidence interval: [{:,.2f}, {:,.2f}].".format(conf_int[0], conf_int[1]) + \
          '\n\n{}\n{}\n\np-value = {}\n{}'.format(ho, test_for, p, conclusion) + \
          "\n\nThe p-value tells you that there is about a {:.2%} to accept the hipothesis Ho.".format(p) + \
          "\nHipotesis Ho is {}.".format("rejected" if p<significance else "accepted") + \
          "\nA p-value below {} means 'statistically significant'.".format(significance)
    
    # Plot the histogram of the replicates
    ax.hist(bs_sample, bins=bins, rwidth=.9, density=True, color='red', label='replicates sample')
    best_fit = gaussian_model(bs_sample, mu, sigma)
    ax.plot(bs_sample, best_fit, linestyle=" ", ms=.5, marker="o", color='darkred')
    ax.axvline(bs_sample.mean(), color='darkred', linestyle='dashed', linewidth=2, label='replicates mean')
    ax.axvspan(conf_int[0], conf_int[1], color='red', alpha=0.1, label='confidence interval')
    
    ax.axvline(effect_size, color='darkblue', lw=2, label='effect size')
    min_x, max_x = ax.get_xlim()
    ax.axvspan(effect_size, max_x, color='gray', alpha=0.5, label='p-value')
    
    ax.set_xlim(min_x, max_x)
    ax.set_xlabel(x_label)
    ax.set_ylabel('PDF')
    t = ax.text(pos_x_conf_legend, pos_y_conf_legend, msg, 
                transform=ax.transAxes, color='black', ha='left', va='top')
    t.set_bbox(dict(facecolor='white', alpha=0.95, edgecolor='darkgray'))
    ax.set_title("Mean expected (Boostrap samples)" + title, **params_title)
    ax.legend(loc='upper right')
    ax.grid(True)    
    
def plotting_test_Ho_with_boostrap(ax, bs_empirical, bs_sample, ho, test_for, conclusion,
                                   title='', params_title={}, bins=50, significance=0.01,
                                   pos_x_conf_legend=0.03, pos_y_conf_legend=0.97, legend_loc='upper right'):
    """Plot the histogram of the test statistic and boostrap shuffled and the validation of Ho."""
    # Compute the 95% confidence interval: conf_int
    ci_empirical = np.percentile(bs_empirical, [2.5, 97.5])
    ci_sample = np.percentile(bs_sample, [2.5, 97.5])
    
    # add a 'best fit' line
    mu_empirical = bs_empirical.mean(); sigma_empirical = bs_empirical.std();
    mu_sample = bs_sample.mean(); sigma_sample = bs_sample.std();
    
    # Get the p-value
    empirical_test = bs_empirical.mean()
    p = np.sum(bs_sample >= empirical_test)*1.0 / len(bs_sample)
    
    msg = "95% ci statistic test: [{:,.2f}, {:,.2f}].".format(ci_empirical[0], ci_empirical[1]) + \
          "\n95% ci replicates test: [{:,.2f}, {:,.2f}].".format(ci_sample[0], ci_sample[1]) + \
          '\n\n{}\n{}\n\np-value = {}\n{}'.format(ho, test_for, p, conclusion) + \
          "\n\nThe p-value tells you that there is about a {:.2%} to accept the hipothesis Ho.".format(p) + \
          "\nHipotesis Ho is {}.".format("rejected" if p<significance else "accepted") + \
          "\nA p-value below {} means 'statistically significant'.".format(significance)
    # To sinc the bins
    n, bins = (0, 20) if bins==None else np.histogram(np.concatenate((bs_empirical, bs_sample)), bins=int(bins*1.5))
    
    # Plot the histogram of the replicates
    #ax.hist(bs_empirical, bins=bins, rwidth=.9, density=True, color='blue', alpha=.7, label='Empirical statistic test')
    #empirical_fit = gaussian_model(bs_empirical, mu_empirical, sigma_empirical)
    #ax.plot(bs_empirical, empirical_fit, linestyle=" ", ms=1, marker="o", color='darkblue')
    _, fit, _ = ax.hist(bs_empirical, bins=bins, rwidth=.9, density=True, color='blue', alpha=.6, label='Empirical statistic test')
    empirical_fit = gaussian_model(fit, mu_empirical, sigma_empirical)
    ax.plot(fit, empirical_fit, ls='--', lw=.5, color='darkblue')
    #ax.axvline(bs_empirical.mean(), color='darkblue', linestyle='dashed', linewidth=2)
    ax.axvspan(ci_empirical[0], ci_empirical[1], color='blue', alpha=0.1)
    
    #ax.hist(bs_sample, bins=bins, rwidth=.9, density=True, color='red', alpha=.7, label='Replicates statistic test')
    #sample_fit = gaussian_model(bs_sample, mu_sample, sigma_sample)
    #ax.plot(bs_sample, sample_fit, linestyle=" ", ms=1, marker="o", color='darkred')
    _, fit, _ = ax.hist(bs_sample, bins=bins, rwidth=.9, density=True, color='red', alpha=.6, label='Replicates statistic test')
    sample_fit = gaussian_model(fit, mu_sample, sigma_sample)
    ax.plot(fit, sample_fit, ls='--', lw=.5, color='darkred')
    #ax.axvline(bs_sample.mean(), color='darkred', linestyle='dashed', linewidth=2)
    ax.axvspan(ci_sample[0], ci_sample[1], color='red', alpha=0.1)
    
    min_x, max_x = ax.get_xlim()
    ax.axvline(mu_empirical, color='darkblue', lw=2, label='effect size')
    ax.axvspan(mu_empirical, max_x, color='gray', alpha=0.5, label='p-value')
    
    ax.set_xlim(min_x, max_x)
    ax.set_xlabel('Test Statistic Values')
    ax.set_ylabel('PDF')
    t = ax.text(pos_x_conf_legend, pos_y_conf_legend, msg, 
                transform=ax.transAxes, color='black', ha='left', va='top')
    t.set_bbox(dict(facecolor='white', alpha=0.95, edgecolor='darkgray'))
    ax.set_title("Test Statistic Distribution" + title, **params_title)
    ax.legend(loc=legend_loc)#'lower center')
    ax.grid(True)    

def plotting_bs_linreg(ax, x, y, bs_intercept, bs_slope,
                       x_label='X data', y_label='Y data', title='', params_title={},
                       pos_x_conf_legend=0.03, pos_y_conf_legend=0.97):
    """Plotting bootstrap regressions"""
    # Compute the 95% confidence interval: conf_int
    conf_int_slope = np.percentile(bs_slope, [2.5, 97.5]) 
    conf_int_intercept = np.percentile(bs_intercept, [2.5, 97.5]) 
    
    # Plot the data
    ax.plot(x, y, linestyle=" ", marker='.')
    
    num_linreg_to_plot = 100 #len(bs_slope)
    # Plot the bootstrap lines
    for i in range(num_linreg_to_plot):
        ax.plot(x, bs_intercept[i] + bs_slope[i]*x, linewidth=.5, alpha=.5, color='red')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    t = ax.text(pos_x_conf_legend, pos_y_conf_legend, 
                "95% confidence\ninterval.\nSlope:\n[{:,.2f}, {:,.2f}].\nIntercept:\n[{:,.2f}, {:,.2f}].\nPearson cor:\n{:,.4f}.".format(
                    conf_int_slope[0],conf_int_slope[1],conf_int_intercept[0],conf_int_intercept[1],pearson_r(x, y)), 
                transform=ax.transAxes, color='black', ha='left', va='top')
    t.set_bbox(dict(facecolor='antiquewhite', alpha=0.8, edgecolor='antiquewhite'))
    ax.set_title("Expected Linear Regression" + title, **params_title)
    ax.legend(['population model', 'sampling'], loc='lower right')
    ax.grid(True)    
    

###############################################################################
## Main part of the code
###############################################################################
def Inferential_Statistics_Concepts(population, seed=SEED):
    print("****************************************************")
    topic = "1. Inferential Statistics Concepts"; print("** %s\n" % topic)
    
    # Set random seed for reproducibility
    np.random.seed(SEED)
        
    size_sample = 31
    pop_size = len(population)
    pop_mean = population.mean()
    pop_std = population.std()    
    
    ###########################################################################
    print("** Taking one sample:")
    ###########################################################################
    # Draw a Random Sample from a Population
    sample = np.random.choice(population, size=size_sample)
    sample_size = len(sample)
    sample_mean = sample.mean()
    sample_std = sample.std()
    
    print("Population size:",  pop_size,    "Mean: ", pop_mean,    "Std:", pop_std)
    print("Sample     size: ", sample_size, "Mean:",  sample_mean, "Std:", sample_std)
    
    # Global configuration of graphs
    plt.rcParams.update({'axes.labelsize': 6, 
                         'xtick.labelsize': 6, 'ytick.labelsize': 6, 
                         'legend.fontsize': 6})
    params_title = dict(fontsize=8, color='maroon', loc='left')
    
    fig, axis = plt.subplots(3,2, figsize=(12, 5.5))
    fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
    
    ax = axis[0,0]
    ax.set_title("Daily High Temperature in August\n" + \
                 "Population size: {:3.0f}, Mean: {:8.4f}, Std: {:6.4f}\n".format(
                     pop_size, pop_mean, pop_std) + \
                 "Sample size: {:3.0f}, Mean: {:8.4f}, Std: {:6.4f}".format(
                     sample_size, sample_mean, sample_std), **params_title )
    _, bins, _ = ax.hist(population, bins=10, rwidth=.9, alpha=.5, label='population')
    ax.hist(sample, bins=bins, rwidth=.6, alpha=.5, label='sample')
    ax.set_xlabel('Temperature Bins (deg F°)')
    ax.set_ylabel('Count of Day per Bin')
    ax.legend()
    
    ax = axis[0,1]
    ax.set_title(">>> Normalized", **params_title)
    _, bins, _ = ax.hist(population, bins=10, rwidth=.9, alpha=.5, density=True, label='population')
    ax.hist(sample, bins=bins, rwidth=.6, alpha=.5, density=True, label='sample')
    ax.set_xlabel('Temperature Bins (deg F°)')
    ax.set_ylabel('Fraction of Total Days')
    ax.legend()
    
        
    
    ###########################################################################
    print("\n** Taking two more samples:")
    ###########################################################################
    # Draw two Random Sample from a Population
    sample1 = np.random.choice(population, size=size_sample)
    sample2 = np.random.choice(population, size=size_sample)
    
    sample1_size = len(sample1);   sample2_size = len(sample2);
    sample1_mean = sample1.mean(); sample2_mean = sample2.mean();
    sample1_std = sample1.std();   sample2_std = sample2.std()

    print("Population size:",  pop_size,     "Mean: ", pop_mean,    "Std:", pop_std)
    print("Sample 1   size: ", sample1_size, "Mean: ", sample1_mean,"Std:", sample1_std)
    print("Sample 2   size: ", sample2_size, "Mean: ", sample2_mean,"Std:", sample2_std)
    
    ax = axis[1,0]
    ax.set_title("Daily High Temperature in August\n" + \
                 "Population size: {:3.0f}, Mean: {:8.4f}, Std: {:6.4f}\n".format(
                     pop_size, pop_mean, pop_std) + \
                 "Sample1 size: {:3.0f}, Mean: {:8.4f}, Std: {:6.4f}\n".format(
                     sample1_size, sample1_mean, sample1_std) + \
                 "Sample2 size: {:3.0f}, Mean: {:8.4f}, Std: {:6.4f}".format(
                     sample2_size, sample2_mean, sample2_std), **params_title)
    _, bins, _ = ax.hist(population, bins=10, rwidth=.9, alpha=.5, label='population')
    ax.hist(sample1, bins=bins, rwidth=.6, alpha=.5, label='sample 1')
    ax.hist(sample2, bins=bins, rwidth=.3, alpha=.5, label='sample 2')
    ax.set_xlabel('Temperature Bins (deg F°)')
    ax.set_ylabel('Count of Day per Bin')
    ax.legend()
    
    ax = axis[1,1]
    ax.set_title(">>> Normalized", **params_title)
    _, bins, _ = ax.hist(population, bins=10, rwidth=.9, alpha=.5, density=True, label='population')
    ax.hist(sample1, bins=bins, rwidth=.6, alpha=.5, density=True, label='sample1')
    ax.hist(sample2, bins=bins, rwidth=.3, alpha=.5, density=True, label='sample1')
    ax.set_xlabel('Temperature Bins (deg F°)')
    ax.set_ylabel('Fraction of Total Days')
    ax.legend()
    
    
    
    ###########################################################################
    print("\n** Resampling as iteration:")
    ###########################################################################
    num_samples = 300
    distribution_of_means = np.zeros(num_samples)
    distribution_of_std = np.zeros(num_samples)
    for ns in range(num_samples):
        sample = np.random.choice(population, size_sample)
        distribution_of_means[ns] = sample.mean()
        distribution_of_std[ns] = sample.std()
        
    # Sample Distribution Statistics
    mean_of_means = np.mean(distribution_of_means)
    #std_of_means = np.std(distribution_of_means)
    mean_of_std = np.mean(distribution_of_std)
    #std_of_std = np.std(distribution_of_std)
    
    print("Population size  :",  pop_size,    "Mean:  ", pop_mean,     "Std:", pop_std)
    print("Number of samples:", num_samples,  "Mean: ", mean_of_means, "Std:", mean_of_std)
    
    ax = axis[2,0]
    ax.set_title("Daily High Temperature in August\n" + \
                 "Population size: {:3.0f}, Mean: {:8.4f}, Std: {:6.4f}\n".format(
                     pop_size, pop_mean, pop_std) + \
                 "Number of samples: {:3.0f}, Mean: {:8.4f}, Std: {:6.4f}".format(
                     num_samples, mean_of_means, mean_of_std), **params_title)
    _, bins, _ = ax.hist(population, bins=10, rwidth=.9, alpha=.5, label='population')
    ax.hist(distribution_of_means, bins=bins, rwidth=.6, alpha=.5, label='boostrap sample')
    ax.set_xlabel('Temperature Bins (deg F°)')
    ax.set_ylabel('Count of Day per Bin')
    ax.legend()
    
    ax = axis[2,1]
    ax.set_title(">>> Normalized", **params_title)
    _, bins, _ = ax.hist(population, bins=10, rwidth=.9, alpha=.5, density=True, label='population')
    ax.hist(distribution_of_means, bins=bins, rwidth=.6, alpha=.5, density=True, label='boostrap sample')
    ax.set_xlabel('Temperature Bins (deg F°)')
    ax.set_ylabel('Fraction of Total Days')
    ax.legend()
    
    plt.subplots_adjust(left=None, bottom=.1, right=None, top=.85, wspace=.3, hspace=1.2);
    plt.show()
    plt.style.use('default')
    
def Sample_Statistics_versus_Population(population, seed=SEED):
    print("****************************************************")
    topic = "2. Sample Statistics versus Population"; print("** %s\n" % topic)
    
    size_sample = 31
    
    # Compute the population statistics
    print("Population size: {:3.0f}, mean {:.1f}, stdev {:.2f}".format( len(population), population.mean(), population.std() ))
    
    # Set random seed for reproducibility
    np.random.seed(SEED)
    
    # Construct a sample by randomly sampling 31 points from the population
    sample = np.random.choice(population, size=size_sample)
    
    # Compare sample statistics to the population statistics
    print("    Sample size: {:3.0f}, mean {:.1f}, stdev {:.2f}".format( len(sample), sample.mean(), sample.std() ))
    
def Variation_in_Sample_Statistics(population, seed=SEED):
    print("****************************************************")
    topic = "3. Variation in Sample Statistics"; print("** %s\n" % topic)
    
    num_samples = 300
    size_sample = 31
    
    # Set random seed for reproducibility
    np.random.seed(SEED)
    
    # Initialize two arrays of zeros to be used as containers
    means = np.zeros(num_samples)
    stdevs = np.zeros(num_samples)
    
    # For each iteration, compute and store the sample mean and sample stdev
    for ns in range(num_samples):
        sample = np.random.choice(population, size_sample)
        means[ns] = sample.mean()
        stdevs[ns] = sample.std()
        
    # Compute the population statistics
    print("Population mean {:.1f}, stdev {:.2f}".format( population.mean(), population.std() ))
    
    # Compute and print the mean() and std() for the sample statistic distributions
    print("Means : center = {:>6.2f}, spread = {:>6.2f}".format(means.mean(), means.std()))
    print("Stdevs: center = {:>6.2f}, spread = {:>6.2f}".format(stdevs.mean(), stdevs.std()))
    
def Visualizing_Variation_of_a_Statistic(population, seed=SEED):
    print("****************************************************")
    topic = "4. Visualizing Variation of a Statistic"; print("** %s\n" % topic)
    
    # Set random seed for reproducibility
    np.random.seed(SEED)
    
    num_samples = 300
    size_sample = 31
    bins = 10
    
    pop_size = len(population)
    pop_mean = population.mean()
    pop_std = population.std()    
    
    
    
    # Specific function
    def get_sample_statistics(population, num_samples, size_sample):
        # Initialize two arrays of zeros to be used as containers
        means = np.zeros(num_samples)
        stdevs = np.zeros(num_samples)
        
        # For each iteration, compute and store the sample mean and sample stdev
        for ns in range(num_samples):
            sample = np.random.choice(population, size_sample)
            means[ns] = sample.mean()
            stdevs[ns] = sample.std()
        return means, stdevs
    
    
    # Generate sample distribution and associated statistics
    means, stdevs = get_sample_statistics(population, num_samples, size_sample)
    
    
    # Global configuration of graphs
    plt.rcParams.update({'axes.labelsize': 9, 
                         'xtick.labelsize': 9, 'ytick.labelsize': 9, 
                         'legend.fontsize': 9})
    params_title = dict(fontsize=8, color='maroon')
    
    # Plot the distribution of means, and the distribution of stdevs
    fig, axis = plt.subplots(1, 3, figsize=(12.1, 3.5))
    fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
    
    ax = axis[0]
    ax.set_title("Daily High Temperature in August\n" + \
                 "Population size: {:3.0f}, Mean: {:8.4f}, Std: {:6.4f}".format(
                     pop_size, pop_mean, pop_std), **params_title )
    ax.hist(population, bins=bins, rwidth=.9, alpha=.5, label='population')
    ax.set_xlabel('Temperature Bins (deg F°)')
    ax.set_ylabel('Count of Day per Bin')
    ax.legend()
    
    
    ax = axis[1]
    ax.set_title("Distribution of the Means:\n" + \
                 "center: {:8.4f}, spread: {:6.4f}".format(
                     means.mean(), means.std()), **params_title)
    ax.hist(means, bins=bins, rwidth=.9, alpha=.5, color='green', label='Means sample')
    ax.set_xlabel('Values of Means')
    ax.set_ylabel('Bin counts of Means')
    ax.legend()
    
    
    ax = axis[2]
    ax.set_title("Distribution of the Stedevs:\n" + \
                 "center: {:8.4f}, spread: {:6.4f}".format(
                     stdevs.mean(), stdevs.std()), **params_title)
    ax.hist(stdevs, bins=bins, rwidth=.9, alpha=.5, color='red', label='Stdvs sample')
    ax.set_xlabel('Values of Stdevs')
    ax.set_ylabel('Bin counts of Stdevs')
    ax.legend()
    
    plt.subplots_adjust(left=None, bottom=.12, right=None, top=.8, wspace=.3);
    plt.show()
    plt.style.use('default')
    
def Model_Estimation_and_Likelihood(df, seed=SEED):
    print("****************************************************")
    topic = "5. Model Estimation and Likelihood"; print("** %s\n" % topic)
    
    ###########################################################################
    print("** Estimation:")
    ###########################################################################
    
    # Compute sample statistics
    mean_sample = df.sample_distances.mean()
    std_sample = df.sample_distances.std()
    print("   Sample size: {}, mean: {:.4f}, stdev: {:.4f}".format(df.shape[0], mean_sample, std_sample))
    
    population_model = gaussian_model(df.sample_distances, mu=mean_sample, sigma=std_sample)
    
    # Plot the model
    plt.rcParams.update({'axes.labelsize': 9, 
                         'xtick.labelsize': 9, 'ytick.labelsize': 9, 
                         'legend.fontsize': 9})
    params_title = dict(fontsize=9, color='maroon')
    bins = 20
    fig, axis = plt.subplots(1,3, figsize=(12.1, 3.5))
    fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
    
    ax = axis[0]
    ax.set_title("Sample Distances", **params_title )
    ax.hist(df.sample_distances, bins=bins, rwidth=.9, alpha=.5, density=True)
    ax.set_xlabel('Distance bin values')
    ax.set_ylabel('Distances bin counts')
    
    ax = axis[1]
    ax.set_title("Gaussian model, mu={:.4f}, sigma={:.4f}".format(mean_sample, std_sample), **params_title )
    ax.plot(df.sample_distances, population_model, linestyle=" ", ms=2, marker="o", color='red')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Probability density')
    
    ax = axis[2]
    ax.set_title("Data and Model", **params_title )
    ax.plot(df.sample_distances, population_model, linestyle=" ", ms=2, marker="o", color='red', label='population model')
    ax.hist(df.sample_distances, bins=bins, rwidth=.9, alpha=.5, density=True, label='sample distances')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Population vs Sample')
    ax.legend()
    
    plt.subplots_adjust(left=None, bottom=.15, right=None, top=.85, hspace=None, wspace=.5);
    plt.show()
    plt.style.use('default')
    
    
    
    ###########################################################################
    print("\n** Likelihood from Probabilities:")
    ###########################################################################
    # For each sample point, compute a probability
    probs = np.zeros(len(df.sample_distances))
    for n, distance in enumerate(df.sample_distances):
        probs[n] = gaussian_model(distance, mu=mean_sample, sigma=std_sample)
    
    likelihood = np.prod(probs)
    loglikelihood = np.sum(np.log(probs))
    
    print('   Model likelihood: ', likelihood)
    print('   Model loglikelihood: ', loglikelihood)
    
    
    
    ###########################################################################
    print("\n** Maximum Likelihood Estimation:")
    ###########################################################################
    # Create an array of mu guesses, centered on sample_mean, spread out +/- by sample_stdev
    low_guess = mean_sample - 2*std_sample
    high_guess = mean_sample + 2*std_sample
    mu_guesses = np.linspace(low_guess, high_guess, 101)

    # Compute the loglikelihood for each model created from each guess value
    loglikelihoods = np.zeros(len(mu_guesses))
    for n, mu_guess in enumerate(mu_guesses):
        loglikelihoods[n] = compute_loglikelihood(df.sample_distances, mu=mu_guess, sigma=std_sample)

    # Find the best guess by using logical indexing, the print and plot the result
    max_loglikelihood = np.max(loglikelihoods)
    min_loglikelihood = np.min(loglikelihoods)
    best_mu = mu_guesses[loglikelihoods==max_loglikelihood][0]
    max_mu = mu_guesses.max()
    min_mu = mu_guesses.min()
    print('Maximum loglikelihood = {} found for best mu guess = {}'.format(max_loglikelihood, best_mu))
    
    # Plot the loglikelihoods
    plt.rcParams.update({'axes.labelsize': 9, 
                         'xtick.labelsize': 9, 'ytick.labelsize': 9, 
                         'legend.fontsize': 9})
    params_title = dict(fontsize=14, color='maroon')
    
    fig, ax = plt.subplots()
    fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
    ax.set_title("Max Log Likelihood = {}\nwas found at Mu = {}".format(max_loglikelihood, best_mu), **params_title )
    ax.set_xlim(min_mu, max_mu);
    ax.set_ylim(min_loglikelihood, max_loglikelihood+100)
    ax.plot(mu_guesses, loglikelihoods)
    ax.plot(best_mu, max_loglikelihood, linestyle=" ", ms=10, marker="o", color='red')
    ax.vlines(best_mu, min_loglikelihood, max_loglikelihood, ls='--', color='gray', alpha=.5)
    ax.hlines(max_loglikelihood, min_mu, best_mu, ls='--', color='gray', alpha=.5)
    ax.set_xlabel('Guesses for Mu')
    ax.set_ylabel('Log Likelihoods')
    plt.subplots_adjust(left=.15, bottom=.15, right=None, top=.8, hspace=None, wspace=None);
    plt.show()
    plt.style.use('default')
    
    
    """
    ###########################################################################
    print("\n** Finding the model:")
    ###########################################################################
    # Fit the model, based on the form of the formula
    df_model = pd.DataFrame({'x': df.index,
                             'y': df.sample_distances.cumsum()}) #
    model_fit = ols(formula="y ~ x", data=df_model).fit()
    
    # Extract the model parameters and associated "errors" or uncertainties
    a0 = model_fit.params['Intercept']
    a1 = model_fit.params['x']
    e0 = model_fit.bse['Intercept']
    e1 = model_fit.bse['x']
    
    # Defining the y values in the model
    df_model['ym'] = a0 + a1*df_model.x
    
    # Print the results
    print("   Model found: Y = {:,.4f} + {:,.4f} X\n".format(a0, a1))
    print('   For     slope a1 = {:,.4f}, the uncertainty in a1 is {:,.4f}'.format(a1, e1))
    print('   For intercept a0 = {:,.4f}, the uncertainty in a0 is {:,.4f}\n'.format(a0, e0))
    
    residuals = df_model.y - df_model.ym
    rss = np.square(residuals).sum()
    rmse = np.sqrt(np.mean(np.square(residuals)))
    r_squared = np.square(np.corrcoef(df_model.y, df_model.ym)[0,1])
    print('   Building evaluation:')
    print('                        RSS = {:,.4f}'.format(rss))
    print("   Goodness-of-Fit:")
    print('                       RMSE = {:,.4f}'.format(rmse))
    print('                  R-squared = {:,.4f}'.format(r_squared))
    """
    
def Estimation_of_Population_Parameters(df, seed=SEED):
    print("****************************************************")
    topic = "6. Estimation of Population Parameters"; print("** %s\n" % topic)
    
    # Compute the mean and standard deviation of the sample_distances
    sample_mean = np.mean(df.sample_distances)
    sample_stdev = np.std(df.sample_distances)
    print("Sample size: {}, mean: {:.4f}, stdev: {:.4f}".format(df.shape[0], sample_mean, sample_stdev))
    
    # Use the sample mean and stdev as estimates of the population model parameters mu and sigma
    population_model = gaussian_model(df.sample_distances, mu=sample_mean, sigma=sample_stdev)
    
    
    # Plot the model
    #plt.rcParams.update({'axes.labelsize': 9, 
    #                     'xtick.labelsize': 9, 'ytick.labelsize': 9, 
    #                     'legend.fontsize': 9})
    params_title = dict(fontsize=20, color='maroon')
    bins = 20
    fig, ax = plt.subplots()
    fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
    ax.set_title("Data and Model", **params_title )
    ax.plot(df.sample_distances, population_model, linestyle=" ", ms=2, marker="o", color='red', label='population model')
    ax.hist(df.sample_distances, bins=bins, rwidth=.9, alpha=.5, density=True, label='sample distances')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Population vs Sample')
    ax.legend()
    plt.subplots_adjust(left=None, bottom=.15, right=None, top=.85, hspace=None, wspace=None);
    plt.show()
    plt.style.use('default')
    
def Maximizing_Likelihood_Part_1(df, seed=SEED):
    print("****************************************************")
    topic = "7. Maximizing Likelihood, Part 1"; print("** %s\n" % topic)
        
    # Compute the mean and standard deviation of the sample_distances
    sample_mean = np.mean(df.sample_distances)
    sample_stdev = np.std(df.sample_distances)
    print("Sample size: {}, mean: {:.4f}, stdev: {:.4f}".format(df.shape[0], sample_mean, sample_stdev))
    
    # For each sample distance, compute the probability modeled by the parameter guesses
    probs = np.zeros(len(df.sample_distances))
    for n, distance in enumerate(df.sample_distances):
        probs[n] = gaussian_model(distance, mu=sample_mean, sigma=sample_stdev)
    
    # Compute and print the log-likelihood as the sum() of the log() of the probabilities
    loglikelihood = np.sum(np.log(probs))
    print('For guesses mu={:0.2f} and sigma={:0.2f}, the loglikelihood={:0.2f}'.format(sample_mean, sample_stdev, loglikelihood))
    
def Maximizing_Likelihood_Part_2(df, seed=SEED):
    print("****************************************************")
    topic = "8. Maximizing Likelihood, Part 2"; print("** %s\n" % topic)
    
    # Compute the mean and standard deviation of the sample_distances
    sample_mean = np.mean(df.sample_distances)
    sample_stdev = np.std(df.sample_distances)
    print("Sample size: {}, mean: {:.4f}, stdev: {:.4f}".format(df.shape[0], sample_mean, sample_stdev))
    
    # Create an array of mu guesses, centered on sample_mean, spread out +/- by sample_stdev
    low_guess = sample_mean - 2*sample_stdev
    high_guess = sample_mean + 2*sample_stdev
    mu_guesses = np.linspace(low_guess, high_guess, 101)

    # Compute the loglikelihood for each model created from each guess value
    loglikelihoods = np.zeros(len(mu_guesses))
    for n, mu_guess in enumerate(mu_guesses):
        loglikelihoods[n] = compute_loglikelihood(df.sample_distances, mu=mu_guess, sigma=sample_stdev)

    # Find the best guess by using logical indexing, the print and plot the result
    max_loglikelihood = np.max(loglikelihoods)
    min_loglikelihood = np.min(loglikelihoods)
    best_mu = mu_guesses[loglikelihoods==max_loglikelihood][0]
    max_mu = mu_guesses.max()
    min_mu = mu_guesses.min()
    print('Maximum loglikelihood = {} found for best mu guess = {}'.format(max_loglikelihood, best_mu))
    
    # Plot the loglikelihoods
    plt.rcParams.update({'axes.labelsize': 9, 
                         'xtick.labelsize': 9, 'ytick.labelsize': 9, 
                         'legend.fontsize': 9})
    params_title = dict(fontsize=14, color='maroon')
    
    fig, ax = plt.subplots()
    fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
    ax.set_title("Max Log Likelihood = {}\nwas found at Mu = {}".format(max_loglikelihood, best_mu), **params_title )
    ax.set_xlim(min_mu, max_mu);
    ax.set_ylim(min_loglikelihood, max_loglikelihood+100)
    ax.plot(mu_guesses, loglikelihoods)
    ax.plot(best_mu, max_loglikelihood, linestyle=" ", ms=10, marker="o", color='red')
    ax.vlines(best_mu, min_loglikelihood, max_loglikelihood, ls='--', color='gray', alpha=.5)
    ax.hlines(max_loglikelihood, min_mu, best_mu, ls='--', color='gray', alpha=.5)
    ax.set_xlabel('Guesses for Mu')
    ax.set_ylabel('Log Likelihoods')
    plt.subplots_adjust(left=.15, bottom=.15, right=None, top=.8, hspace=None, wspace=None);
    plt.show()
    plt.style.use('default')
    
def Model_Uncertainty_and_Sample_Distributions(daily_temp, seed=SEED):
    print("****************************************************")
    topic = "9. Model Uncertainty and Sample Distributions"; print("** %s\n" % topic)
    
    ###########################################################################
    print("** First part: Making only thee differents samples")
    ###########################################################################
    # Seed random number generator
    np.random.seed(SEED) 
    
    # Plot the data
    plt.rcParams.update({'axes.labelsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 
                         'legend.fontsize': 8})
    params_title = dict(fontsize=17, color='maroon')
    fig, axis = plt.subplots(1, 2, figsize=(10,4))
    fig.suptitle(topic, fontsize=14, color='darkblue', weight='bold')
    bins = 10
    
    ax = axis[0]
    # Plot the histogram of the data
    ax.hist(daily_temp.temperature_F, bins=bins, rwidth=.8, label='Population, mean:{:,.4f}'.format(daily_temp.temperature_F.mean()))
    ax.set_xlabel('Temperature Bins [deg F]')
    ax.set_ylabel('Counts of Days per Bin')
    ax.set_title("Daily High Temperatures in August", **params_title)
    ax.legend()
    ax.grid(True)    
    
    ax = axis[1]
    # Plot the histogram of the num_samples
    num_samples = 3
    for i in range(num_samples):
        sample = np.random.choice(daily_temp.temperature_F, size=len(daily_temp.temperature_F), replace=True)
        ax.hist(sample, bins=bins, rwidth=.8, alpha=.25, label='Sample {}, mean:{:,.4f}'.format(i,sample.mean()))
    ax.set_xlabel('Temperature Bins [deg F]')
    ax.set_ylabel('Counts of Days per Bin')
    ax.set_title("{num} Samples, {num} Means".format(num=num_samples), **params_title)
    ax.legend()
    ax.grid(True)    
    
    plt.subplots_adjust(left=None, right=None, bottom=None, top=.8, hspace=None, wspace=None);
    plt.show()
    plt.style.use('default')
    
    
    ###########################################################################
    print("** Second part: Modeling the uncertainty and sample distributions")
    ###########################################################################
    # Seed random number generator
    np.random.seed(SEED) 
    
    # Use sample as model for population
    population_model = daily_temp.temperature_F
    num_resamples = 300
    
    # Get bootstrap replicates of means
    bootstrap_means = draw_bs_reps(population_model, np.mean, num_resamples)

    # Compute the mean of the bootstrap resample distribution
    estimate_temperature = np.mean(bootstrap_means)
    
    # Compute standard deviation of the bootstrap resample distribution
    estimate_uncertainty = np.std(bootstrap_means)
    
    print('For the estimate temperature mean = {:,.4f}, the uncertainty is {:,.4f}'.format(
            estimate_temperature, estimate_uncertainty))
    
    
    # Plot the data
    plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                         'legend.fontsize': 7, 'font.size': 8})
    params_title = dict(fontsize=10, color='maroon')
    fig, axis = plt.subplots(2, 2, figsize=(12,5.5))
    fig.suptitle(topic, fontsize=14, color='darkblue', weight='bold')
    
    ax = axis[0,0]
    ax.set_title('Sample Daily High Temperature', **params_title)
    ax.plot(daily_temp.index, population_model, linestyle=" ", ms=1, marker="o")
    ax.set_xlabel('Days')
    ax.set_ylabel('Temperature F°')
    ax.grid()
    
    ax=axis[1,0]
    title = " - Daily High Temperature" + \
            "\npopulation size: {:.0f}, mean: {:.4f}, std: {:.4f}".format(len(population_model), np.mean(population_model), np.std(population_model)) + \
            "\nDistribution of the Means:" + \
            "\ncenter: {:.4f}, spread: {:.4f}".format(bootstrap_means.mean(), bootstrap_means.std())
    plotting_boostraps_ecdf(ax, population_model, bootstrap_means, 
                            x_label='Daily temp', title=title, params_title=params_title)
    
    ax=axis[0,1]
    plotting_boostraps_hist(ax, bootstrap_means, 
                            x_label='Mean Temp', params_title=params_title)
    
    ax=axis[1,1]
    plotting_boostraps_samples(ax, population_model, x_label='Daily temp', params_title=params_title, seed=seed)
    
    plt.subplots_adjust(left=.05, right=.97, bottom=None, top=.9, hspace=.8, wspace=.2);
    plt.show()
    plt.style.use('default')
    
def Bootstrap_and_Standard_Error(national_park, seed=SEED):
    print("****************************************************")
    topic = "10. Bootstrap and Standard Error"; print("** %s\n" % topic)
    
    # Seed random number generator
    np.random.seed(SEED) 
    
    # We use the sample_Data as the population model
    population_model = national_park.distance
    
    # Resample the population_model 100 times, computing the mean each sample
    num_resamples = 1000
    bootstrap_means = draw_bs_reps(population_model, np.mean, size=num_resamples)
    
    # Compute and print the mean, stdev of the resample distribution of means
    distribution_mean = np.mean(bootstrap_means)
    standard_error = np.std(bootstrap_means)
    print('Bootstrap Distribution: center={:0.1f}, spread={:0.1f}'.format(distribution_mean, standard_error))
    
    # Plot the bootstrap resample distribution of means
    plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                         'legend.fontsize': 6, 'font.size': 8})
    params_title = dict(fontsize=10, color='maroon')
    fig, axis = plt.subplots(1,3, figsize=(12.1,4))
    fig.suptitle(topic, fontsize=14, color='darkblue', weight='bold')
    
    ax = axis[0]
    ax.set_title('Sample Location Data', **params_title)
    ax.plot(national_park.time, population_model, linestyle=" ", ms=1, marker="o")
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Distance [inches]')
    ax.grid()
    
    ax=axis[1]
    title = "\nVariations in daily distance traveled in the National Park" + \
            "\npopulation size: {:.0f}, mean: {:.4f}, std: {:.4f}".format(len(population_model), np.mean(population_model), np.std(population_model)) + \
            "\nDistribution of the Means:\n" + \
            "center: {:.4f}, spread: {:.4f}".format(bootstrap_means.mean(), bootstrap_means.std())
    plotting_boostraps_ecdf(ax, population_model, bootstrap_means, 
                            x_label='Variations', title=title, params_title=params_title)
    
    ax=axis[2]
    plotting_boostraps_hist(ax, bootstrap_means, 
                            x_label='Mean Variations', params_title=params_title)
    
    plt.subplots_adjust(left=.05, right=.95, bottom=None, top=.7, hspace=None, wspace=.3);
    plt.show()
    plt.style.use('default')
    
def Estimating_Speed_and_Confidence(national_park, seed=SEED):
    print("****************************************************")
    topic = "11. Estimating Speed and Confidence"; print("** %s\n" % topic)
    
    num_resamples = 1000
    
    ###########################################################################
    print("** Using the lesson long method:")
    ###########################################################################
    # Seed random number generator
    np.random.seed(SEED) 
    
    # Resample each preloaded population, and compute speed distribution
    distances = national_park.distance.values
    times = national_park.time
    # Resample each preloaded population, and compute speed distribution
    population_inds = np.arange(0, 99, dtype=int)
    resample_speeds = np.zeros(num_resamples)
    for nr in range(num_resamples):
        sample_inds = np.random.choice(population_inds, size=100, replace=True)
        sample_inds.sort()
        sample_distances = distances[sample_inds]
        sample_times = times[sample_inds]
        a1, a0 = np.polyfit(sample_times, sample_distances, 1)
        resample_speeds[nr] = a1
    
    # Compute effect size and confidence interval, and print
    speed_estimate = np.mean(resample_speeds)
    ci_90 = np.percentile(resample_speeds, [5, 95])
    print('Speed Estimate = {:0.2f}, 90% Confidence Interval: {:0.2f}, {:0.2f} '.format(speed_estimate, ci_90[0], ci_90[1]))
    
    
    
    ###########################################################################
    print("\n** Using another method:")
    ###########################################################################
    
    # Fit the model, based on the form of the formula
    model_fit = ols(formula="distance ~ time", data=national_park).fit()
    
    # Extract the model parameters and associated "errors" or uncertainties
    a0 = model_fit.params['Intercept']
    a1 = model_fit.params['time']
    e0 = model_fit.bse['Intercept']
    e1 = model_fit.bse['time']
    
    # Print the results
    msg_model = "\nPopulation model: Y = {:.4f} + {:.4f} X".format(a0, a1) + \
                '\nFor slope = {:.04f}, the uncertainty is {:.02f}'.format(a1, e1) + \
                '\nFor intercept = {:.04f}, the uncertainty is {:.02f}'.format(a0, e0) 
    print(msg_model)
    
    
    x = national_park.time
    y = national_park.distance
    bs_slope, bs_intercept = draw_bs_pairs_linreg(x, y, size=num_resamples)
    
    # Compute effect size and confidence interval, and print
    speed_estimate = np.mean(bs_slope)
    ci_90 = np.percentile(bs_slope, [2.5, 97.5])
    print('\nSpeed Estimate = {:0.4f}, 95% Confidence Interval: {:0.2f}, {:0.2f} '.format(speed_estimate, ci_90[0], ci_90[1]))
    intercept_estimate = np.mean(bs_intercept)
    ci_90 = np.percentile(bs_intercept, [2.5, 97.5])
    print('Intercept Estimate = {:0.4f}, 95% Confidence Interval: {:0.2f}, {:0.2f} '.format(intercept_estimate, ci_90[0], ci_90[1]))
    
    
    # Plot the bootstrap resample distribution of means
    plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                         'legend.fontsize': 6, 'font.size': 8})
    params_title = dict(fontsize=10, color='maroon')
    fig, axis = plt.subplots(1, 3, figsize=(12.1,4))
    fig.suptitle(topic, fontsize=14, color='darkblue', weight='bold')
    
    ax = axis[0]
    plotting_bs_linreg(ax, x, y, bs_intercept, bs_slope,
                       x_label='Time [Hours]', y_label='Distance [Miles]', 
                       title=msg_model, params_title=params_title)
    
    ax=axis[1]
    plotting_boostraps_hist(ax, bs_slope, x_label='Mean Slope', 
                            title='\nSlope = {}'.format(speed_estimate), params_title=params_title)
    
    ax=axis[2]
    plotting_boostraps_hist(ax, bs_intercept, x_label='Mean Intercept', 
                            title='\nIntercept = {}'.format(intercept_estimate), params_title=params_title)
    
    plt.subplots_adjust(left=.05, right=.95, bottom=None, top=.75, hspace=None, wspace=.3);
    plt.show()
    plt.style.use('default')
    
    return x, y, model_fit, bs_intercept, bs_slope
    
def Visualize_the_Bootstrap(x, y, model_fit, bs_intercept, bs_slope, seed=SEED):
    print("****************************************************")
    topic = "12. Visualize the Bootstrap"; print("** %s\n" % topic)
    
    # Extract the model parameters and associated "errors" or uncertainties
    a0 = model_fit.params['Intercept']
    a1 = model_fit.params['time']
    e0 = model_fit.bse['Intercept']
    e1 = model_fit.bse['time']
    
    msg_model = "\nPopulation model: Y = {:.4f} + {:.4f} X".format(a0, a1) + \
                '\nFor slope = {:.04f}, the uncertainty is {:.02f}'.format(a1, e1) + \
                '\nFor intercept = {:.04f}, the uncertainty is {:.02f}'.format(a0, e0) 
    
    speed_estimate = np.mean(bs_slope)
    intercept_estimate = np.mean(bs_intercept)
    
    # Plot the bootstrap resample distribution of means
    plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                         'legend.fontsize': 6, 'font.size': 8})
    params_title = dict(fontsize=10, color='maroon')
    fig, axis = plt.subplots(1, 3, figsize=(12.1,4))
    fig.suptitle(topic, fontsize=14, color='darkblue', weight='bold')
    bins = 21
    
    ax = axis[0]
    plotting_bs_linreg(ax, x, y, bs_intercept, bs_slope,
                       x_label='Time [Hours]', y_label='Distance [Miles]', 
                       title=msg_model, params_title=params_title)
    
    ax=axis[1]
    plotting_boostraps_hist(ax, bs_slope, x_label='Mean Slope', bins=bins,
                            title='\nSlope = {}'.format(speed_estimate), params_title=params_title)
    
    ax=axis[2]
    plotting_boostraps_hist(ax, bs_intercept, x_label='Mean Intercept', bins=bins,
                            title='\nIntercept = {}'.format(intercept_estimate), params_title=params_title)
    
    plt.subplots_adjust(left=.05, right=.95, bottom=None, top=.75, hspace=None, wspace=.3);
    plt.show()
    plt.style.use('default')
    
def Model_Errors_and_Randomness(hiking_data, seed=SEED):
    print("****************************************************")
    topic = "13. Model Errors and Randomness"; print("** %s\n" % topic)
    
    x = hiking_data.time 
    y = hiking_data.distance
    
    short_travel = x < x.max()/2
    hiking_data['trip'] = np.where(hiking_data.distance < hiking_data.distance.max()/2, 'short', 'long')
    
    #x_short = x[short_travel].values
    y_short = y[short_travel].values
    #x_long = x[np.logical_not(short_travel)].values
    y_long = y[np.logical_not(short_travel)].values
    
    ###########################################################################
    print("** Exploratory part:")
    ###########################################################################
    # Plot the bootstrap resample distribution of means
    plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                         'legend.fontsize': 8, 'font.size': 6})
    params_title = dict(fontsize=10, color='maroon')
    fig, axis = plt.subplots(1, 3, figsize=(12.1,3))
    fig.suptitle(topic, fontsize=14, color='darkblue', weight='bold')
    
    ax=axis[0]
    ax.set_title('Hiking Trips, Final Distance and Time Duration', **params_title)
    ax.plot(x, y, linestyle=" ", ms=1, marker="o")
    ax.set_xlabel('Time duration [hours]')
    ax.set_ylabel('Distance traveled [miles]')
    ax.grid()
    
    ax=axis[1]
    my_color = np.where(short_travel, 'red', 'blue')
    ax.set_title('Hiking Trips, Grouped by Hike Duration', **params_title)
    ax.scatter(x, y, color=my_color, s=1)
    ax.set_xlabel('Time duration [hours]')
    ax.set_ylabel('Distance traveled [miles]')
    ax.grid()
    
    ax=axis[2]
    bins = np.linspace(0,25,26)
    ax.set_title('Hiking Trips, Grouped by Duration,\nBinned by Distance Traveled', **params_title)
    ax.hist(y_long, bins=bins, alpha=.5, rwidth=.9, label="Long")
    ax.hist(y_short, bins=bins, alpha=.5, rwidth=.5, label="Short")
    ax.set_xlabel('Distance Bins [miles]')
    ax.set_ylabel('Hike counts per Distance bin')
    ax.legend()
    ax.grid()
    
    plt.subplots_adjust(left=.05, right=.95, bottom=None, top=.8, hspace=None, wspace=.3);
    plt.show()
    plt.style.use('default')
    
    
    ###########################################################################
    print("\n** Testing and Validating:")
    ###########################################################################
    np.random.seed(42)
    size_test = 10000
    
    bs_diff_mean = bootstrap_replicate_2d(y_long, y_short, diff_of_means, size=size_test)
    empirical_diff_means = bs_diff_mean.mean() #diff_of_means(y_long, y_short) # Compute difference of mean impact force from experiment: empirical_diff_means
    
    perm_replicates = draw_perm_reps(y_long, y_short, diff_of_means, size=size_test) # Draw 10,000 permutation replicates: perm_replicates
    p = np.sum(perm_replicates >= empirical_diff_means)*1.0 / len(perm_replicates) # Compute p-value: p
    
    msg = "Ho: The difference observed between long and short trips is by chance" + \
          "\n\nWe computed the probability of getting at least a {:,.4f} miles ".format(empirical_diff_means) + \
          "\ndifference in distance traveled, under the hypotesis." + \
          '\n\np-value = {}'.format(p) + \
          "\n\nMean difference in distances between both group [long-short] = {}.".format(empirical_diff_means) + \
          "\nMean difference in distances after shuffle both group [long-short] = {}.".format(perm_replicates.mean()) + \
          "\n\nThe p-value tells you that there is about a {:.2%}".format(p) + \
          "\nchance that you would get the difference of means" + \
          "\nobserved in the experiment. " + \
          "\nHipotesis Ho is rejected." + \
          "\n\nA p-value below 0.01 means 'statistically significant'."
    print(msg)
    
    
    
    ###########################################################################
    print("\n** Making a standar procedure:")
    ###########################################################################
    # 1. Establish the Hipothesis
    Ho = "Ho: Time don't make a difference in the distance traveled"
    np.random.seed(42)
    size_test = 500
    
    plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                         'legend.fontsize': 8, 'font.size': 6})
    params_title = dict(fontsize=10, color='maroon')
    fig, axis = plt.subplots(1, 3, figsize=(12.1,3))
    fig.suptitle(topic, fontsize=14, color='darkblue', weight='bold')
    
    
    # 2. See the representation of the distribution of values
    ax = axis[0]
    ax.set_title('How Time affect Distance Traveled', **params_title)
    sns.swarmplot(x='trip', y='distance', data=hiking_data, ax=ax) # Make bee swarm plot
    ax.set_ylabel('Distance traveled [miles]')# Label axes
    ax.set_xlabel('Type of trip')
    ax.grid()
    
    
    # 3. Show distributions with respect to categories
    ax = axis[1]
    ax.set_title('Do the Time have unintended consecuence\nin Distance Traveled?', **params_title)
    sns.boxplot(x='trip', y='distance', data=hiking_data, ax=ax) # Make bee swarm plot
    ax.set_ylabel('Distance traveled [miles]')# Label axes
    ax.set_xlabel('Type of trip')
    ax.grid()
    
    
    # 4. Review the ECDF of the groups
    ax = axis[2]
    ax.set_title('Reviewing the ECDF of both groups', **params_title)
    xe_short, ye_short = ecdf(y_short) # Compute x,y values for ECDFs
    xe_long, ye_long = ecdf(y_long)
    ax.plot(xe_short, ye_short, marker='.', linestyle='none') # Plot the ECDFs
    ax.plot(xe_long, ye_long, marker='.', linestyle='none')
    ax.legend(('short', 'long')) # Add a legend
    ax.set_xlabel('miles traveled') # Label axes and show plot
    ax.set_ylabel('ECDF')
    
    plt.subplots_adjust(left=.05, right=.95, bottom=.15, top=.8, hspace=.5, wspace=.3);
    plt.show()
    plt.style.use('default')
    
    
    # 5. Test the Ho.
    # Do the part *** Testing and Validating ***
    plt.rcParams.update({'axes.labelsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 
                         'legend.fontsize': 9, 'font.size': 8})
    params_title = dict(fontsize=10, color='maroon')
    fig, ax = plt.subplots(1, 1, figsize=(12.1, 3))
    fig.suptitle(topic, fontsize=10, color='darkblue', weight='bold')
    
    test_for   = "We computed the probability of getting at least a {:,.2f} miles difference in distance".format(empirical_diff_means) + \
                 "\ntraveled, under the hypotesis."
    conclusion = "Mean difference in distances between both group [long-short] = {}.".format(empirical_diff_means) + \
                 "\nMean difference in distances after shuffle both group [long-short] = {}.".format(perm_replicates.mean())
    x_label = 'difference observed between Long and Short trips'
    title = '\nDo the Time have unintended consecuence in Distance Traveled?'
    plotting_test_Ho(ax, perm_replicates, empirical_diff_means, Ho, test_for, conclusion, 
                     x_label=x_label, title=title, params_title=params_title, 
                     pos_x_conf_legend=0.3, pos_y_conf_legend=.95)
    
    plt.subplots_adjust(left=.05, right=.95, bottom=.15, top=.8, hspace=None, wspace=None);
    plt.show()
    plt.style.use('default')
    
def Test_Statistics_and_Effect_Size(sample_times, sample_distances, seed=SEED):
    print("****************************************************")
    topic = "14. Test Statistics and Effect Size"; print("** %s\n" % topic)
    
    size_test = 10000
    Ho = 'Ho = "Short and long time durations have no effect on total distance traveled."'
    print(Ho)
    
    ###########################################################################
    print("\n** Using the lesson method:")
    ###########################################################################
    np.random.seed(42)
    # Create two poulations, sample_distances for early and late sample_times.
    group_duration_short = sample_distances[sample_times < 5]
    group_duration_long = sample_distances[sample_times > 5]
    
    # Then resample with replacement, taking 500 random draws from each population.
    np.random.seed(42)
    resample_short = np.random.choice(group_duration_short, size=size_test, replace=True)
    resample_long = np.random.choice(group_duration_long, size=size_test, replace=True)
    
    # Difference the resamples to compute a test statistic distribution, then compute its mean and stdev
    test_statistic_lesson = resample_long - resample_short
    effect_size_lesson = np.mean(test_statistic_lesson)
    standard_error = np.std(test_statistic_lesson)

    # Print and plot the results
    print('Test Statistic: mean={:0.2f}, stdev={:0.2f}'.format(effect_size_lesson, standard_error))
    
    # Plot the boostrap sample
    params_title = dict(fontsize=14, color='maroon')
    fig, ax = plt.subplots()
    fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
    bins = 20
    plotting_boostraps_hist(ax, test_statistic_lesson, x_label='Distance difference, late-early', bins=bins,
                            title='\nTest Statistic: mean = {:,.4f}, stdev = {:,.4f}'.format(effect_size_lesson, standard_error), params_title=params_title)
    plt.subplots_adjust(left=None, right=None, bottom=None, top=.8, hspace=None, wspace=None);
    plt.show()
    plt.style.use('default')
    
    ###########################################################################
    print("\n** Using the method learn in subject 20:")
    ###########################################################################
    np.random.seed(42)
    # Create two poulations, sample_distances for early and late sample_times.
    group_duration_short = sample_distances[sample_times < 5]
    group_duration_long = sample_distances[sample_times >= 5]
    
    # Then resample with replacement, taking 500 random draws from each population.
    test_statistic_method = bootstrap_replicate_2d(group_duration_long, group_duration_short, diff_before_get_means, size=size_test)
    effect_size_method = test_statistic_method.mean() #diff_of_means(y_long, y_short) # Compute difference of mean impact force from experiment: empirical_diff_means
    standard_error = test_statistic_method.std()
    
    # Print and plot the results
    print('Test Statistic: mean={:0.2f}, stdev={:0.2f}'.format(effect_size_method, standard_error))
    
    # Plot the boostrap sample
    params_title = dict(fontsize=14, color='maroon')
    fig, ax = plt.subplots()
    fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
    bins = 20
    plotting_boostraps_hist(ax, test_statistic_method, x_label='Distance difference, late-early', bins=bins,
                            title='\nTest Statistic: mean = {:,.4f}, stdev = {:,.4f}'.format(effect_size_method, standard_error), params_title=params_title)
    plt.subplots_adjust(left=None, right=None, bottom=None, top=.8, hspace=None, wspace=None);
    plt.show()
    plt.style.use('default')
    
def Null_Hypothesis(sample_times, sample_distances, seed=SEED):
    print("****************************************************")
    topic = "15. Null Hypothesis"; print("** %s\n" % topic)
    
    size_test = 10000
    Ho = 'Ho = "Short and long time durations have no effect on total distance traveled."'
    print(Ho)
    
    ###########################################################################
    print("** Using the lesson method:")
    ###########################################################################
    np.random.seed(42)
    # Create two poulations, sample_distances for early and late sample_times.
    group_duration_short = sample_distances[sample_times < 5]
    group_duration_long = sample_distances[sample_times > 5]
    
    # Shuffle the time-ordered distances, then slice the result into two populations.
    shuffle_bucket = np.concatenate((group_duration_short, group_duration_long))
    np.random.shuffle(shuffle_bucket)
    slice_index = len(shuffle_bucket)//2
    shuffled_half1 = shuffle_bucket[0:slice_index]
    shuffled_half2 = shuffle_bucket[slice_index:]
    
    # Create new samples from each shuffled population, and compute the test statistic
    resample_half1 = np.random.choice(shuffled_half1, size=size_test, replace=True)
    resample_half2 = np.random.choice(shuffled_half2, size=size_test, replace=True)
    test_statistic = resample_half2 - resample_half1
    
    # Compute and print the effect size
    effect_size = np.mean(test_statistic)
    print('Replicate Test Statistic (after shuffling), mean = {}'.format(effect_size))
    
    ###########################################################################
    print("\n** Using the method learn in subject 20:")
    ###########################################################################
    np.random.seed(42)
    # Create two poulations, sample_distances for early and late sample_times.
    group_duration_short = sample_distances[sample_times < 5]
    group_duration_long = sample_distances[sample_times >= 5]
    
    perm_replicates = draw_perm_reps(group_duration_long, group_duration_short, diff_before_get_means, size=size_test) # Draw 10,000 permutation replicates: perm_replicates
    effect_size = np.mean(perm_replicates)
    print('Replicate Test Statistic (after shuffling), mean = {}'.format(effect_size))
    
def Visualizing_Test_Statistics(sample_times, sample_distances, seed=SEED):
    print("****************************************************")
    topic = "16. Visualizing Test Statistics"; print("** %s\n" % topic)
    
    size_test = 10000
    Ho = 'Ho = "Short and long time durations have no effect on total distance traveled."'; print(Ho);
    
    ###########################################################################
    print("\n** Using the lesson method:")
    ###########################################################################
    np.random.seed(42)
    # Create two poulations, sample_distances for early and late sample_times.
    group_duration_short = sample_distances[sample_times < 5]
    group_duration_long = sample_distances[sample_times > 5]
    
    # From the unshuffled groups, compute the test statistic distribution
    resample_short = np.random.choice(group_duration_short, size=size_test, replace=True)
    resample_long = np.random.choice(group_duration_long, size=size_test, replace=True)
    test_statistic_unshuffled = resample_long - resample_short
    effect_size_unshuffled = test_statistic_unshuffled.mean() #diff_of_means(y_long, y_short) # Compute difference of mean impact force from experiment: empirical_diff_means
    standard_error_unshuffled = test_statistic_unshuffled.std()
    print('Test Statistic (unshuffled): mean={:0.2f}, stdev={:0.2f}'.format(effect_size_unshuffled, standard_error_unshuffled))
    
    # Shuffle two populations, cut in half, and recompute the test statistic
    shuffled_half1, shuffled_half2 = permutation_sample(group_duration_short, group_duration_long)
    resample_half1 = np.random.choice(shuffled_half1, size=size_test, replace=True)
    resample_half2 = np.random.choice(shuffled_half2, size=size_test, replace=True)
    test_statistic_shuffled = resample_half2 - resample_half1
    effect_size_shuffled = test_statistic_shuffled.mean() #diff_of_means(y_long, y_short) # Compute difference of mean impact force from experiment: empirical_diff_means
    standard_error_shuffled = test_statistic_shuffled.std()
    print('Replicate Test Statistic (after shuffling), mean={:0.2f}, stdev={:0.2f}'.format(effect_size_shuffled, standard_error_shuffled))
    
    # Compute and print the p-value
    p = np.sum(test_statistic_shuffled >= effect_size_unshuffled)*1.0 / len(test_statistic_shuffled) # Compute p-value: p
    print('\nJust one experiment, p-value = {}'.format(p))
    
    # Print and plot the results
    msg = 'Test Statistic: mean={:0.2f}, stdev={:0.2f}'.format(effect_size_unshuffled, standard_error_unshuffled) + \
          '\nReplicate Test Statistic: mean={:0.2f}, stdev={:0.2f}'.format(effect_size_shuffled, standard_error_shuffled) + \
          '\n\nP-value={}'.format(p)   
    
    # Plot the test result - first version
    plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                         'legend.fontsize': 8, 'font.size': 6})
    params_title = dict(fontsize=14, color='maroon')
    fig, ax = plt.subplots(1, 1, figsize=(12.1, 4))
    fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
    test_for   = "We computed the probability of getting at least a {:,.2f} miles difference in distance".format(effect_size_unshuffled) + \
                 "\ntraveled, under the hypotesis."
    plotting_test_Ho_with_boostrap(ax, test_statistic_unshuffled, test_statistic_shuffled, Ho, test_for, msg, 
                                   params_title=params_title, significance=0.15, pos_x_conf_legend=.03)
    plt.subplots_adjust(left=.05, right=.95, bottom=None, top=.8, hspace=None, wspace=None);
    plt.show()
    plt.style.use('default')
    
    # Plot the test result - second version
    plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                         'legend.fontsize': 8, 'font.size': 6})
    params_title = dict(fontsize=10, color='maroon')
    fig, ax = plt.subplots(1, 1, figsize=(12.1, 4))
    fig.suptitle(topic, fontsize=10, color='darkblue', weight='bold')
    x_label = 'difference observed between Long and Short trips'
    title = '\nDo the Time have unintended consecuence in Distance Traveled?'
    plotting_test_Ho(ax, test_statistic_shuffled, effect_size_unshuffled, Ho, test_for, msg, 
                     x_label=x_label, title=title, params_title=params_title, significance=0.15, 
                     pos_x_conf_legend=0.01, pos_y_conf_legend=.95)
    plt.subplots_adjust(left=.05, right=.95, bottom=.15, top=.8, hspace=None, wspace=None);
    plt.show()
    plt.style.use('default')
    
    ###########################################################################
    print("\n** Using the method learn in subject 20:")
    ###########################################################################
    np.random.seed(42)
    # Create two poulations, sample_distances for early and late sample_times.
    group_duration_short = sample_distances[sample_times < 5]
    group_duration_long = sample_distances[sample_times >= 5]
    
    # Then resample with replacement, taking 500 random draws from each population.
    bs_empirical = bootstrap_replicate_2d(group_duration_long, group_duration_short, diff_before_get_means, size=size_test)
    empirical_test = bs_empirical.mean() #diff_of_means(y_long, y_short) # Compute difference of mean impact force from experiment: empirical_diff_means
    empirical_std = bs_empirical.std()
    
    bs_replicates = draw_perm_reps(group_duration_long, group_duration_short, diff_before_get_means, size=size_test) # Draw 10,000 permutation replicates: perm_replicates
    replicates_test = np.mean(bs_replicates)
    replicates_std = bs_replicates.std()
    
    # Compute p-value
    p = np.sum(bs_replicates >= empirical_test)*1.0 / len(bs_replicates) # Compute p-value: p
    
    # Print and plot the results
    msg = 'Test Statistic: mean={:0.2f}, stdev={:0.2f}'.format(empirical_test, empirical_std) + \
          '\nReplicate Test Statistic: mean={:0.2f}, stdev={:0.2f}'.format(replicates_test, replicates_std) + \
          '\n\nP-value={}'.format(p)   
    print(msg)
    
    # Plot the test result - first version
    plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                         'legend.fontsize': 8, 'font.size': 6})
    params_title = dict(fontsize=14, color='maroon')
    fig, ax = plt.subplots(1, 1, figsize=(12.1, 4))
    fig.suptitle(topic, fontsize=17, color='darkblue', weight='bold')
    
    test_for   = "We computed the probability of getting at least a {:,.2f} miles difference in distance".format(empirical_test) + \
                 "\ntraveled, under the hypotesis."
    plotting_test_Ho_with_boostrap(ax, bs_empirical, bs_replicates, Ho, test_for, msg, 
                                   params_title=params_title, bins=None, 
                                   pos_x_conf_legend=.35, legend_loc='lower center')
    plt.subplots_adjust(left=.05, right=.95, bottom=None, top=.8, hspace=None, wspace=None);
    plt.show()
    plt.style.use('default')
    
    # Plot the test result - second version
    plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                         'legend.fontsize': 8, 'font.size': 6})
    params_title = dict(fontsize=10, color='maroon')
    fig, ax = plt.subplots(1, 1, figsize=(12.1, 4))
    fig.suptitle(topic, fontsize=10, color='darkblue', weight='bold')
    test_for   = "We computed the probability of getting at least a {:,.2f} miles difference in distance".format(empirical_test) + \
                 "\ntraveled, under the hypotesis."
    x_label = 'difference observed between Long and Short trips'
    title = '\nDo the Time have unintended consecuence in Distance Traveled?'
    plotting_test_Ho(ax, bs_replicates, empirical_test, Ho, test_for, msg, x_label=x_label, 
                     title=title, params_title=params_title, 
                     pos_x_conf_legend=0.3, pos_y_conf_legend=.95)
    plt.subplots_adjust(left=.05, right=.95, bottom=.15, top=.8, hspace=None, wspace=None);
    plt.show()
    plt.style.use('default')
    
def Visualizing_the_P_Value(seed=SEED):
    print("****************************************************")
    topic = "17. Visualizing the P-Value"; print("** %s\n" % topic)
    
    
    
def Course_Conclusion(seed=SEED):
    print("****************************************************")
    topic = "18. Course Conclusion"; print("** %s\n" % topic)



def main():
    print("****************************************************")
    print("** BEGIN                                          **")
    
    file = 'daily_temperature_in_august.csv'
    daily_temp   = pd.read_csv(file)
    
    Inferential_Statistics_Concepts(daily_temp.temperature_F.values)
    Sample_Statistics_versus_Population(daily_temp.temperature_F.values)
    Variation_in_Sample_Statistics(daily_temp.temperature_F.values)
    Visualizing_Variation_of_a_Statistic(daily_temp.temperature_F.values)
    
    file = 'sample_distances.csv'
    sample_distances   = pd.read_csv(file)
            
    Model_Estimation_and_Likelihood(sample_distances)
    Estimation_of_Population_Parameters(sample_distances)
    Maximizing_Likelihood_Part_1(sample_distances)
    Maximizing_Likelihood_Part_2(sample_distances)
    
    Model_Uncertainty_and_Sample_Distributions(daily_temp)
    
    file = 'national_park.csv'
    national_park   = pd.read_csv(file)
    national_park['time'] = national_park.index/100
        
    Bootstrap_and_Standard_Error(national_park)
    x, y, model_fit, bs_intercept, bs_slope = Estimating_Speed_and_Confidence(national_park)
    Visualize_the_Bootstrap(x, y, model_fit, bs_intercept, bs_slope)
    
    file = 'hiking_data.csv'
    hiking_data   = pd.read_csv(file)
        
    Model_Errors_and_Randomness(hiking_data)
    Test_Statistics_and_Effect_Size(hiking_data.time.values, hiking_data.distance.values)
    Null_Hypothesis(hiking_data.time.values, hiking_data.distance.values)
    
    Visualizing_Test_Statistics(hiking_data.time.values, hiking_data.distance.values)
    Visualizing_the_P_Value()
    Course_Conclusion()
    
    print("****************************************************")
    print("** END                                            **")
    print("****************************************************")



if __name__ == '__main__':
    main()
    plt.style.use('default')
