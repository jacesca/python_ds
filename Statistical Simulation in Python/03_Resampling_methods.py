# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 3: Resampling methods
    In this chapter, we will get a brief introduction to resampling methods and 
    their applications. We will get a taste of bootstrap resampling, jackknife 
    resampling, and permutation testing. After completing this chapter, students 
    will be able to start applying simple resampling methods for data analysis.
Source: https://learn.datacamp.com/courses/statistical-simulation-in-python
"""
###############################################################################
## Importing libraries
###############################################################################
import numpy as np
import pandas as pd

import statsmodels.regression.linear_model as sm


###############################################################################
## Preparing the environment
###############################################################################
# Global variables
SEED = 123
SIZE = 10000

# Global configuration
np.set_printoptions(formatter={'float': '{:,.4f}'.format})
np.random.seed(SEED) 
        
###############################################################################
## Reading the data
###############################################################################
# Setting data for wrench_lengths 
wrench_lengths = np.random.uniform(low=7, high=13, size=100)
    

###############################################################################
## Main part of the code
###############################################################################
def Probability_example(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "3 Probability example"; print("** %s" % topic)
    print("****************************************************")
    
    print('Consider a bowl filled with colored candies: ' + \
          'three blue, two green, and five yellow. ')
    print('Draw three candies at random, with replacement and without replacement. ')
    print('You want to know the probability of drawing a yellow candy on ' + \
          'the third draw given that the first candy was blue and the second ' + \
          'candy was green.\n\n')
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Set up the bowl
    success_rep, success_no_rep = np.zeros(size), np.zeros(size)
    bowl = list(np.repeat('b',3)) + list(np.repeat('g',2)) + list(np.repeat('y',5))

    for i in range(size):
        # Sample with and without replacement & increment success counters
        sample_rep = np.random.choice(bowl, size=3, replace=True)
        sample_no_rep = np.random.choice(bowl, size=3, replace=False)
        
        success_rep[i] = (sample_rep == ['b','g','y']).all()
        success_no_rep[i] = (sample_no_rep == ['b','g','y']).all()
        
    # Calculate probabilities
    prob_with_replacement = success_rep.mean()
    prob_without_replacement = success_no_rep.mean()
    
    print("Probability with replacement = {}, without replacement = {}.\n\n".format(prob_with_replacement, prob_without_replacement))
    
    
    
def Running_a_simple_bootstrap(wrench_lengths=wrench_lengths, size=1000, seed=SEED):
    print("****************************************************")
    topic = "5 Running a simple bootstrap"; print("** %s" % topic)
    print("****************************************************")
    
    print('Suppose you own a factory that produces wrenches. You want ' + \
          'to be able to characterize the average length of the wrenches ' + \
          'and ensure that they meet some specifications. ')
    print('Your factory produces thousands of wrenches every day, but ' + \
          'it is infeasible to measure the length of each wrench. However, ' + \
          'you have access to a representative sample of 100 wrenches. ' + \
          "Let's use bootstrapping to get the 95% confidence interval (CI) " + \
          'for the average lengths.\n\n')
        
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Set up the init variables
    len_samples, boostrap = len(wrench_lengths), np.zeros(size)
    
    # Draw some random sample with replacement and append mean
    for i in range(size):
        temp_sample = np.random.choice(wrench_lengths, size=len_samples, replace=True)
        boostrap[i] = temp_sample.mean()
        
    # Calculate bootstrapped mean and 95% confidence interval.
    mean_estimated = boostrap.mean()
    ci = np.percentile(boostrap, [2.5, 97.5])
    
    print("Bootstrapped Mean Length = {:,.4f}, 95% CI = {}.\n\n".format(mean_estimated, ci))
    
    
    
def Non_standard_estimators(size=1000, seed=SEED):
    print("****************************************************")
    topic = "6 Non-standard estimators"; print("** %s" % topic)
    print("****************************************************")
    
    print('Suppose you are studying the health of students. You are given the ' + \
          'height and weight of 1000 students and are interested in the median ' + \
          'height as well as the correlation between height and weight and the ' + \
          "associated 95% CI for these quantities. Let's use bootstrapping.")
    print('Calculate the 95% CI for both the median height as well as the ' + \
          'correlation between height and weight.\n\n')
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Set up the init variables
    len_samples = 1000
    df = pd.DataFrame({'Heights': np.sort(np.random.normal(loc=5.42, scale=2, size=len_samples)),
                       'Weights': np.sort(np.random.normal(loc=189.94, scale=73.36, size=len_samples))})
    boostrap_heights_median, boostrap_hw_corr = np.zeros(size), np.zeros(size)
    
    # Draw some random sample with replacement and append median
    for i in range(size):
        samples = df.sample(n=len_samples, replace=True)
        boostrap_heights_median[i] = samples.Heights.median()
        boostrap_hw_corr[i] = samples.Heights.corr(samples.Weights)
    
    # Calculate confidence intervals
    height_median_ci = np.percentile(boostrap_heights_median, [2.5, 97.5])
    height_weight_corr_ci = np.percentile(boostrap_hw_corr, [2.5, 97.5])
    
    print("Height Median CI = {} \nHeight Weight Correlation CI = {}.\n\n".format( height_median_ci, height_weight_corr_ci))
    
    
    
def Bootstrapping_regression(size=1000, seed=SEED):
    print("****************************************************")
    topic = "7 Bootstrapping regression"; print("** %s" % topic)
    print("****************************************************")
    
    print('Use bootstrapping to calculate the 95% CI of R2')
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('-------------------------------------------df.head()')
    len_samples = 1000
    df = pd.DataFrame({'y': np.sort(np.random.normal(loc=1.5, scale=.36, size=len_samples)),
                       'Intercept': np.repeat(1, len_samples),
                       'X1': np.sort(np.random.uniform(low=0, high=1, size=len_samples)),
                       'X2': np.sort(np.random.uniform(low=0, high=1, size=len_samples))})
    print(df.head())
    
    print('-----------------------------------reg_fit.summary()')
    reg_fit = sm.OLS(df['y'], df.iloc[:,1:]).fit()
    print(reg_fit.summary())
    
    print('---------------------------Boostraping to get 95% CI')
    # Set up the init variables
    boostrap = np.zeros(size)
    
    # Draw some random sample with replacement and append mean
    for i in range(size):
        sample = df.sample(n=len_samples, replace=True)
        boostrap[i] = sm.OLS(sample['y'], sample.iloc[:,1:]).fit().rsquared
        
    # Calculate bootstrapped mean and 95% confidence interval.
    ci = np.percentile(boostrap, [2.5, 97.5])
    print("R Squared 95% CI = {}.\n\n".format(ci))
    
    
    
def Basic_jackknife_estimation_mean(wrench_lengths=wrench_lengths):
    print("****************************************************")
    topic = "9 Basic jackknife estimation - mean"; print("** %s" % topic)
    print("****************************************************")
    
    print('You own a wrench factory and want to measure the average length ' + \
          'of the wrenches to ensure that they meet some specifications. ' + \
          'Your factory produces thousands of wrenches every day, but it is ' + \
          'infeasible to measure the length of each wrench. However, you have ' + \
          'access to a representative sample of 100 wrenches.')
    print("Let's use jackknife estimation to get the average lengths.\n\n")
    
    # Set up the init variables
    n = len(wrench_lengths)
    
    # Leave one observation out from wrench_lengths to get the jackknife sample and store the mean length
    mean_lengths, median_lengths, index = np.zeros(n), np.zeros(n), np.arange(n)
    for i in range(n):
        jk_sample = wrench_lengths[index != i]
        mean_lengths[i] = jk_sample.mean()
        median_lengths[i] = np.median(jk_sample)
        
    # The jackknife estimate is the mean of the mean lengths from each sample
    mean_lengths_jk = mean_lengths.mean()
    print("Jackknife estimate of the mean = {}.\n\n".format(mean_lengths_jk))
    
    
    
    topic = "10 Jackknife confidence interval for the median"; print("** %s" % topic)
    print('Returning to the wrench factory, you are now interested in ' + \
          'estimating the median length of the wrenches along with a 95% CI ' + \
          'to ensure that the wrenches are within tolerance.\n\n')
    
    # Calculate jackknife estimate and it's variance
    median_lengths_jk = median_lengths.mean()
    jk_std = np.sqrt((n-1)*median_lengths.var())
    
    # Assuming normality, calculate lower and upper 95% confidence intervals
    ci = [median_lengths_jk - 1.96*jk_std, median_lengths_jk + 1.96*jk_std]
    print("Jackknife 95% CI {}".format(ci))
    
    
    
def Generating_a_single_permutation(size=1000, seed=SEED):
    print("****************************************************")
    topic = "12 Generating a single permutation"; print("** %s" % topic)
    print("****************************************************")
    
    print('We want to see if there is any difference in the donations ' + \
          'generated by the two designs - A and B. Suppose that you have ' + \
          'been running both the versions for a few days and have generated ' + \
          '500 donations on A and 700 donations on B, stored in the variables ' + \
          'donations_A and donations_B.\n\n')
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Set up the init variables
    len_donations_A, len_donations_B = 500, 700
    donations_A = np.random.exponential(scale=5.92, size=len_donations_A)
    donations_B = np.random.exponential(scale=4.92, size=len_donations_B)
    
    # Concatenate the two arrays donations_A and donations_B into data
    data = np.concatenate([donations_A, donations_B])

    # Get a single permutation of the concatenated length
    data = np.random.permutation(data)

    # Calculate the permutated datasets and difference in means
    permuted_A = data[:len_donations_A]
    permuted_B = data[len_donations_A:]
    
    diff_in_means = permuted_A.mean() - permuted_B.mean()
    print("Difference in the permuted mean values = {:,.4f}.\n\n".format(diff_in_means))
    
    
    
    topic = "13 Hypothesis testing - Difference of means"; print("** %s" % topic)
    print('Now, we will generate a null distribution of the difference in ' + \
          'means and then calculate the p-value.')
    print('We calculate the p-value as twice the fraction of cases where the ' + \
          'difference is greater than or equal to the absolute value of the ' + \
          'test statistic (2-sided hypothesis). ')
    print('A p-value of less than say 0.05 could then determine statistical ' + \
          'significance.')
    print('For instance, if the test statistic falls within the 95% confidence ' + \
          'interval, we can say that there is no real difference between ' + \
          'donations A and B.\n\n')
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Concatenate the two arrays donations_A and donations_B into data
    data = np.concatenate([donations_A, donations_B])
    
    # Generate permutations equal to the number of repetitions
    perm = np.array([np.random.permutation(data) for i in range(size)])
    permuted_A_datasets = perm[:, :len_donations_A]
    permuted_B_datasets = perm[:, len_donations_A:]

    # Calculate the difference in means for each of the datasets
    samples = np.mean(permuted_A_datasets, axis=1) - np.mean(permuted_B_datasets, axis=1)
    ci = np.percentile(samples, [2.5, 97.5])    
    
    # Calculate the test statistic and p-value
    test_stat = np.mean(donations_A) - np.mean(donations_B)
    p_val = 2*np.mean(samples >= np.abs(test_stat))
    
    print("Test Statistical = {:,.4f}.".format(test_stat))    
    print("Permutation CI = {}.\n".format(ci))    
    print("p-value = {:,.4f}.".format(p_val))  
    print("Conclusion: Both donations groups are {}.\n\n".format(np.where(((test_stat >= ci[0]) & (test_stat <= ci[1])), 'equals', 'differents')))
    
    
    topic = "14 Hypothesis testing - Non-standard statistics"; print("** %s" % topic)
    print('Suppose that you are interested in understanding the distribution ' + \
          'of the donations received from websites A and B. You want to see if ' + \
          'there is a statistically significant difference in the median and ' + \
          'the 80th percentile of the donations.\n\n')
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Calculate the difference in 80th percentile and median for each of the permuted datasets (A and B)
    samples_percentile = np.percentile(permuted_A_datasets, 80, axis=1) - np.percentile(permuted_B_datasets, 80, axis=1)
    samples_median = np.median(permuted_A_datasets, axis=1) - np.median(permuted_B_datasets, axis=1)

    # Calculate the test statistic from the original dataset and corresponding p-values
    test_stat_percentile = np.percentile(donations_A, 80) - np.percentile(donations_B, 80)
    test_stat_median = np.median(donations_A) - np.median(donations_B)
    
    # Calculate the test statistic from the original dataset and corresponding p-values
    p_val_percentile = 2*np.mean(samples_percentile >= np.abs(test_stat_percentile))
    p_val_median = 2*np.mean(samples_median >= np.abs(test_stat_median))

    print("80th Percentile: test statistic = {:,.4f}, p-value = {:,.4f}".format(test_stat_percentile, p_val_percentile))
    print("Conclusion: Both donations groups are {}.\n".format(np.where(p_val_percentile<0.05, 'differents', 'equals')))
    
    print("Median: test statistic = {:,.4f}, p-value = {:,.4f}".format(test_stat_median, p_val_median))
    print("Conclusion: Both donations groups are {}.\n\n".format(np.where(p_val_median<0.05, 'differents', 'equals')))
    
    
        
def main():
    print("\n\n****************************************************")
    print("** BEGIN                                          **")
    
    Probability_example()
    
    Running_a_simple_bootstrap()
    Non_standard_estimators()
    Bootstrapping_regression()
    
    Basic_jackknife_estimation_mean()
    
    Generating_a_single_permutation()
    
    
    print("\n\n****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    
    # Return to default
    np.set_printoptions(formatter={'float': None})