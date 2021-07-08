# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 4: Personalization A/B Test
    Combine pandas with the powers of SQL to find out just how many problems 
    New Yorkers have with their housing. This chapter features introductory 
    SQL topics like WHERE clauses, aggregate functions, and basic joins.
Source: https://learn.datacamp.com/courses/streamlined-data-ingestion-with-pandas
"""
###############################################################################
## Importing libraries
###############################################################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

from scipy.stats import ttest_ind

###############################################################################
## Preparing the environment
###############################################################################
# Global configuration
plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                     'legend.fontsize': 6, 'font.size': 6})

#Global variables
suptitle_param = dict(color='darkblue', fontsize=9)
title_param    = {'color': 'darkred', 'fontsize': 10, 'weight': 'bold'}
figsize        = (12.1, 5.9)
SEED           = 42
size           = 10000


# Global Funtion
def retention_rate(df, col_groupby,  
                   col_user='user_id', col_retention='is_retained', col_convertion='converted'):
    """
    Return the retention rate of a set in agroup by structure.
    Parameters
    ----------
    df            : Dataframe with the data to analyze.
    col_groupby   : List with the columns_name to make the group by.
    col_user      : String with the name of column to identify users.
    col_retention : String with the name of ccolumn to identify retentions. The default is 'is_retained'.
    col_convertion: String with the name of column to identify conversions. The default is 'converted'.
    Returns
    -------
    retention_rate : Dataframe with the retention rates.
    """
    # Group by column_names and calculate retention
    retained = df[df[col_retention] == True].groupby(col_groupby)[col_user].nunique()
    # Group by column_names and calculate conversion
    converted = df[df[col_convertion] == True].groupby(col_groupby)[col_user].nunique()
    
    retention_rate = retained/converted
    retention_rate = retention_rate.fillna(0)
    return retention_rate

def conversion_rate(df, col_groupby,  
                   col_user='user_id', col_conversion='converted'):
    """
    Return the retention rate of a set in agroup by structure.
    Parameters
    ----------
    df            : Dataframe with the data to analyze.
    col_groupby   : List with the columns_name to make the group by.
    col_user      : String with the name of column to identify users.
    col_convertion: String with the name of column to identify conversions. The default is 'converted'.
    Returns
    -------
    retention_rate : Dataframe with the coversion rates.
    """
    # Group by column_names and calculate retention
    total_user = df.groupby(col_groupby)[col_user].nunique()
    # Group by column_names and calculate conversion
    conversed = df[df[col_conversion] == True].groupby(col_groupby)[col_user].nunique()
    conversion_rate = conversed/total_user
    conversion_rate = conversion_rate.fillna(0)
    return conversion_rate

def lift(a,b):
    # Calcuate the mean of a and b
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    # Calculate the lift using a_mean and b_mean
    return (b_mean - a_mean) / a_mean
    
  
#Permutation functions
def diff_converse(converse_A, converse_B): return converse_B.mean() - converse_A.mean()
def bootstrap_replicate_2d(data1, data2, func, size=1, replace=True):
    """Perform pairs bootstrap for linear regression."""
    bs_func = np.zeros(size) # Initialize replicates: bs_slope_reps, bs_intercept_reps
    for i in range(size): # Generate replicates
        bs_data1 = np.random.choice(data1, size=len(data1), replace=replace)
        bs_data2 = np.random.choice(data2, size=len(data2), replace=replace)
        bs_func[i] = func(bs_data1, bs_data2)
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
    perm_replicates = np.empty(size) # Initialize array of replicates: perm_replicates
    for i in range(size):
        perm_sample_1, perm_sample_2 = permutation_sample(data_1,data_2) # Generate permutation sample
        perm_replicates[i] = func(perm_sample_1, perm_sample_2) # Compute the test statistic
    return perm_replicates
def gaussian_model(x, mu, sigma):
    """Define gaussian model function"""
    coeff_part = 1/(np.sqrt(2 * np.pi * sigma**2))
    exp_part = np.exp( - (x - mu)**2 / (2 * sigma**2) )
    return coeff_part*exp_part
def plotting_test_Ho(ax, bs_sample, effect_size, msg, x_label='', title='', params_title=title_param,
                     pos_x_msg=0.01, pos_y_msg=0.97):
    """Plot the histogram of the replicates and its confidence interval."""
    # Compute the 95% confidence interval: conf_int
    conf_int = np.percentile(bs_sample, [2.5, 97.5])
    
    # add a 'best fit' line
    mu = bs_sample.mean()
    sigma = bs_sample.std()
    
    msg = msg + "\n95% confidence interval: [{:,.2f}, {:,.2f}].".format(conf_int[0], conf_int[1])
    
    # Plot the histogram of the replicates
    #best_fit = gaussian_model(np.sort(bs_sample), mu, sigma)
    #ax.plot(bs_sample, best_fit, linestyle=" ", ms=.5, marker="o", color='darkred')
    _, bins, _ = ax.hist(bs_sample, bins=50, rwidth=.9, density=True, color='red', label='replicates sample')
    #y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2)) # add a 'best fit' line
    y = gaussian_model(bins, mu, sigma)
    ax.plot(bins, y, '--', color='black')
    ax.axvline(bs_sample.mean(), color='darkred', linestyle='solid', linewidth=2, label='replicates mean')
    ax.axvspan(conf_int[0], conf_int[1], color='red', alpha=0.1, label='confidence interval')
    
    ax.axvline(effect_size, color='darkblue', lw=2, label='effect size')
    min_x, max_x = ax.get_xlim()
    ax.axvspan(effect_size, max_x, color='gray', alpha=0.5, label='p-value')
    
    ax.set_xlim(min_x, max_x)
    ax.set_xlabel(x_label)
    ax.set_ylabel('PDF')
    t = ax.text(pos_x_msg, pos_y_msg, msg, transform=ax.transAxes, color='black', ha='left', va='top')
    t.set_bbox(dict(facecolor='white', alpha=0.95, edgecolor='darkgray'))
    ax.set_title(title, **params_title)
    ax.legend(loc='upper right')
    ax.grid(True)    
def plotting_test_Ho_with_boostrap(ax, bs_empirical, bs_sample, msg, x_label='', title='', 
                                   params_title=title_param, pos_x_msg=0.01, pos_y_msg=0.97):
    """Plot the histogram of the test statistic and boostrap shuffled and the validation of Ho."""
    # Compute the 95% confidence interval: conf_int
    ci_empirical = np.percentile(bs_empirical, [2.5, 97.5])
    ci_sample = np.percentile(bs_sample, [2.5, 97.5])
    
    # Add a 'best fit' line
    mu_empirical = bs_empirical.mean(); sigma_empirical = bs_empirical.std();
    mu_sample = bs_sample.mean(); sigma_sample = bs_sample.std();
    
    # Plot the histogram of the replicates
    _, fit, _ = ax.hist(bs_empirical, bins=50, rwidth=.9, density=True, color='blue', alpha=.6, label='Empirical statistic test')
    empirical_fit = gaussian_model(fit, mu_empirical, sigma_empirical)
    ax.plot(fit, empirical_fit, ls='--', color='darkblue')
    ax.axvspan(ci_empirical[0], ci_empirical[1], color='blue', alpha=0.1)
    
    _, fit, _ = ax.hist(bs_sample, bins=50, rwidth=.9, density=True, color='red', alpha=.6, label='Replicates statistic test')
    sample_fit = gaussian_model(fit, mu_sample, sigma_sample)
    ax.plot(fit, sample_fit, ls='--', color='darkred')
    ax.axvspan(ci_sample[0], ci_sample[1], color='red', alpha=0.1)
    
    min_x, max_x = ax.get_xlim()
    ax.axvline(mu_empirical, color='darkblue', lw=2, label='effect size')
    ax.axvspan(mu_empirical, max_x, color='gray', alpha=0.5, label='p-value')
    
    ax.set_xlim(min_x, max_x)
    ax.set_xlabel('Test Statistic Values')
    ax.set_ylabel('PDF')
    t = ax.text(pos_x_msg, pos_y_msg, msg, transform=ax.transAxes, color='black', ha='left', va='top')
    t.set_bbox(dict(facecolor='white', alpha=0.95, edgecolor='darkgray'))
    ax.set_title(title, **params_title)
    ax.legend(loc='upper right')
    ax.grid(True)    


###############################################################################
## Reading the data
###############################################################################
marketing = pd.read_csv('marketing_dataset_2.csv', index_col=0,
                        parse_dates=['date_served', 'date_subscribed', 'date_canceled'],
                        dtype={'converted': bool,
                               'is_retained': bool})

###############################################################################
## Main part of the code
###############################################################################
print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "1. A/B testing for marketing"; print("** %s" % topic)
print("****************************************************")
topic = "2. Determining key metrics"; print("** %s" % topic)
print("****************************************************")
print('---------------------------------------------Test allocation')
email = marketing[marketing['marketing_channel'] == 'Email']
allocation = email.groupby(['variant'])['user_id'].nunique()
print(allocation.head())

print("****************************************************")
topic = "3. Test allocation"; print("** %s" % topic)
print("****************************************************")
fig, ax = plt.subplots(figsize=figsize)
allocation.plot(kind='bar', ax=ax)
ax.set_xlabel(allocation.index.name)
ax.set_ylabel('No. de Participantes')
ax.set_title('Personalization Test Alloccation', **title_param)    
fig.suptitle(topic, **suptitle_param)
plt.subplots_adjust(left=None, bottom=.2, right=None, top=None, wspace=None, hspace=None); #To set the margins 
plt.show()

print("****************************************************")
topic = "4. Comparing conversion rates"; print("** %s" % topic)
print("****************************************************")
print('---------------------------------------------Setting up our data to evaluate the test')
# Group by user_id and variant
subscribers = email.groupby(['user_id', 'variant'])['converted'].max()
print(subscribers.head())
subscribers = subscribers.unstack(level='variant')
print(subscribers)

print('---------------------------------------------Setting up our data to evaluate the test')
# Drop missing values from the control column
control = subscribers['control'].dropna()
print(control.head())
# Drop missing values from the personalization column
personalization = subscribers['personalization'].dropna()
print(personalization.head())

print('---------------------------------------------Conversion rates')
print("Control conversion rate:", np.mean(control))
print("Personalization conversion rate:", np.mean(personalization))

    
    
print("****************************************************")
topic = "5. Calculating lift & significance testing"; print("** %s" % topic)
print("****************************************************")
topic = "6. Creating a lift function"; print("** %s" % topic)
print("****************************************************")

print('---------------------------------------------Calculating lift')
lift_value = lift(control, personalization)
print("Lift: ", str(round(lift_value*100, 2)) + '%')

print("****************************************************")
topic = "7. Evaluating statistical significance"; print("** %s" % topic)
print("****************************************************")
np.random.seed(SEED) 

print('---------------------------------------------T-test in PythonT-test in Python')
t = ttest_ind(control, personalization)
print(t)
print("""
A p-value less than 0.05 is typiically considered statistically 
significant at 95% significance level.

Since the p-value here is indeed less than 0.05, we can be 
confident that the difference in conversion rates is 95%
statistically significant.
""")
    
print('---------------------------------------------p-value manual - Lesson: 20.4.1')
np.random.seed(SEED) 

Ho = 'Ho: The converse rate is not affected by the email personalization.\n\n'

diff_converse_observed = diff_converse(control, personalization)
txt_Statistic = 'Test Statistic observed: \n' +\
                f'Converse difference between personalized and control mails: {diff_converse_observed:0.4f}.\n\n'

perm_replicates = draw_perm_reps(control, personalization, diff_converse, size=size)
p_value = np.sum(perm_replicates >= diff_converse_observed)*1.0 / size
txt_pvalue = f'p-value: {p_value}\n\n'
conclusion = "Ho is {} because the p-value is {} than 0.05 (95% significance).".\
             format(np.where(p_value<0.05,'rejected', 'accepted'), 
                    np.where(p_value<0.05,'less','greater'))
print(Ho + txt_Statistic + txt_pvalue + conclusion)

#Visualize th p-value to help understanding
fig, ax = plt.subplots(figsize=figsize)
plotting_test_Ho(ax, perm_replicates, diff_converse_observed, 
                 msg = Ho + txt_Statistic + txt_pvalue + conclusion, 
                 x_label='Converse difference in permutation samples', title='Lesson: 20.4.1')
plt.subplots_adjust(left=.05, right=.95, bottom=.15, top=.9, hspace=None, wspace=None);
fig.suptitle(topic, **suptitle_param)
plt.show()

print('---------------------------------------------p-value manual - Lesson: 45.4.16')
np.random.seed(SEED) 

Ho = 'Ho: The converse rate is not affected by the email personalization.\n\n'

bs_statistic = bootstrap_replicate_2d(control, personalization, diff_converse, size=size)
diff_converse_observed = bs_statistic.mean() #diff_of_means(y_long, y_short) # Compute difference of mean impact force from experiment: empirical_diff_means

txt_Statistic = 'Test Statistic observed: \n' +\
                f'Converse difference between personalized and control mails: {diff_converse_observed:0.4f}.\n\n'


perm_replicates = draw_perm_reps(control, personalization, diff_converse, size=size)
p_value = np.sum(perm_replicates >= diff_converse_observed)*1.0 / size
txt_pvalue = f'p-value: {p_value}\n\n'
conclusion = "Ho is {} because the p-value is {} than 0.05 (95% significance).".\
             format(np.where(p_value<0.05,'rejected', 'accepted'), 
                    np.where(p_value<0.05,'less','greater'))
print(Ho + txt_Statistic + txt_pvalue + conclusion)


# Visualize th p-value to help understanding
fig, ax = plt.subplots(figsize=figsize)
plotting_test_Ho(ax, perm_replicates, diff_converse_observed, 
                 msg = Ho + txt_Statistic + txt_pvalue + conclusion, 
                 x_label='Converse difference in permutation samples', title='Lesson: 45.4.16')
fig.suptitle(topic, **suptitle_param)
plt.subplots_adjust(left=.05, right=.95, bottom=.15, top=.9, hspace=None, wspace=None);
plt.show()


# Visualize the p-value and distribution of the replicates samples
fig, ax = plt.subplots(figsize=figsize)
plotting_test_Ho_with_boostrap(ax, bs_statistic, perm_replicates, 
                               msg = Ho + txt_Statistic + txt_pvalue + conclusion,
                               x_label='Converse difference', title='Lesson: 45.4.16')
plt.subplots_adjust(left=.05, right=.95, bottom=None, top=.9, hspace=None, wspace=None);
fig.suptitle(topic, **suptitle_param)
plt.show()



print("****************************************************")
topic = "8. A/B testing & segmentation"; print("** %s" % topic)
print("****************************************************")
np.random.seed(SEED) 
Ho = 'Ho: The converse rate is not affected by the email personalization.\n\n'

print('---------------------------------------------Personalization test segmented by language')
fig, axes = plt.subplots(2, 2, figsize=figsize)
for ax, language in zip(axes.flatten(), np.unique(marketing['language_displayed'].values)):
    print(f'\n---------------------------------------------{language.upper()}')
    
    # Isolate the relevant data
    language_data = marketing[(marketing['marketing_channel'] == 'Email') & (marketing['language_displayed'] == language)]
    
    # Isolate subscribers
    subscribers = language_data.groupby(['user_id', 'variant'])['converted'].max()
    subscribers = subscribers.unstack(level=1)
    
    # Isolate control and personalization
    control = subscribers['control'].dropna()
    personalization = subscribers['personalization'].dropna()
    
    print('lift:', lift(control, personalization))
    print('t-statistic:', ttest_ind(control, personalization), '\n')

    # p-value manual - Lesson: 45.4.16
    bs_statistic = bootstrap_replicate_2d(control, personalization, diff_converse, size=size)
    diff_converse_observed = bs_statistic.mean() #diff_of_means(y_long, y_short) # Compute difference of mean impact force from experiment: empirical_diff_means

    txt_Statistic = 'Test Statistic observed: \n' +\
                    f'Converse difference between personalized and control mails: {diff_converse_observed:0.4f}.\n\n'

    perm_replicates = draw_perm_reps(control, personalization, diff_converse, size=size)
    p_value = np.sum(perm_replicates >= diff_converse_observed)*1.0 / size
    txt_pvalue = f'p-value: {p_value}\n'
    conclusion = "Ho is {} because the p-value is {} than 0.05 (95% significance).".\
                 format(np.where(p_value<0.05,'rejected', 'accepted'), 
                        np.where(p_value<0.05,'less','greater'))
    print(Ho + txt_Statistic + txt_pvalue + conclusion)
    
    # Visualize th p-value to help understanding
    plotting_test_Ho(ax, perm_replicates, diff_converse_observed, 
                     msg = Ho + txt_Statistic + txt_pvalue + conclusion, 
                     x_label='Converse difference in permutation samples', title=language.upper())

fig.suptitle(topic, **suptitle_param)
plt.subplots_adjust(left=.05, right=.95, bottom=.15, top=.9, hspace=.5, wspace=None);
plt.show()


print("****************************************************")
topic = "9. Building an A/B test segmenting function"; print("** %s" % topic)
print("****************************************************")

print('---------------------------------------------Creating ab_segmentation function')
def ab_segmentation(segment, Ho, seed=SEED):
    np.random.seed(SEED) 
    # Build a for loop for each subsegment in marketing
    for subsegment in np.unique(marketing[segment].values):
        print(f'---------------------------------------------{subsegment}', )
        
        # Limit marketing to email and subsegment
        email = marketing[(marketing['marketing_channel'] == 'Email') & 
                          (marketing[segment] == subsegment)]
        subscribers = email.groupby(['user_id', 'variant'])['converted'].max()
        subscribers = pd.DataFrame(subscribers.unstack(level=1)) 
        control = subscribers['control'].dropna()
        personalization = subscribers['personalization'].dropna()
        
        print('lift:', lift(control, personalization)) 
        print('t-statistic:', ttest_ind(control, personalization), '\n')
    
        # p-value manual - Lesson: 45.4.16
        bs_statistic = bootstrap_replicate_2d(control, personalization, diff_converse, size=size)
        diff_converse_observed = bs_statistic.mean() #diff_of_means(y_long, y_short) # Compute difference of mean impact force from experiment: empirical_diff_means
        
        txt_Statistic = 'Test Statistic observed: \n' +\
                        f'Converse difference between personalized and control mails: {diff_converse_observed:0.4f}.\n'
        
        perm_replicates = draw_perm_reps(control, personalization, diff_converse, size=size)
        p_value = np.sum(perm_replicates >= diff_converse_observed)*1.0 / size
        txt_pvalue = f'p-value: {p_value}\n'
        conclusion = "Ho is {} because the p-value is {} than 0.05 (95% significance).\n\n".\
                     format(np.where(p_value<0.05,'rejected', 'accepted'), 
                            np.where(p_value<0.05,'less','greater'))
        print(Ho + txt_Statistic + txt_pvalue + conclusion)
        
        
print('---------------------------------------------Creating ab_segmentation_plot function')
def ab_segmentation_plot(segment, Ho, ncols=2, seed=SEED):
    np.random.seed(SEED) 
    
    set_group = np.unique(marketing[segment].values)
    nrows = math.ceil(len(set_group)/ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    # Build a for loop for each subsegment in marketing
    for i, (ax, subsegment) in enumerate(zip(axes, set_group)):
        print(f'---------------------------------------------{subsegment}', )
        
        # Limit marketing to email and subsegment
        email = marketing[(marketing['marketing_channel'] == 'Email') & 
                          (marketing[segment] == subsegment)]
        subscribers = email.groupby(['user_id', 'variant'])['converted'].max()
        subscribers = pd.DataFrame(subscribers.unstack(level=1)) 
        control = subscribers['control'].dropna()
        personalization = subscribers['personalization'].dropna()
        
        print('lift:', lift(control, personalization)) 
        print('t-statistic:', ttest_ind(control, personalization), '\n')
    
        # p-value manual - Lesson: 45.4.16
        bs_statistic = bootstrap_replicate_2d(control, personalization, diff_converse, size=size)
        diff_converse_observed = bs_statistic.mean() #diff_of_means(y_long, y_short) # Compute difference of mean impact force from experiment: empirical_diff_means

        txt_Statistic = 'Test Statistic observed: \n' +\
                        f'Converse difference between personalized and control mails: {diff_converse_observed:0.4f}.\n'

        perm_replicates = draw_perm_reps(control, personalization, diff_converse, size=size)
        p_value = np.sum(perm_replicates >= diff_converse_observed)*1.0 / size
        txt_pvalue = f'p-value: {p_value}\n'
        conclusion = "Ho is {} because the p-value is {} than 0.05 (95% significance).".\
                     format(np.where(p_value<0.05,'rejected', 'accepted'), 
                            np.where(p_value<0.05,'less','greater'))
        print(Ho + txt_Statistic + txt_pvalue + conclusion)
    
        # Visualize th p-value to help understanding
        plotting_test_Ho(ax, perm_replicates, diff_converse_observed, 
                         msg = Ho + txt_Statistic + txt_pvalue + conclusion, 
                         x_label='Converse difference in permutation samples', title=subsegment.upper())
    
    if i+1 < nrows*ncols: 
        for k in range(i+1, nrows*ncols): axes[k].axis('off')
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=.05, right=.95, bottom=.05, top=.9, hspace=1.2, wspace=None);
    plt.show()
        
    
    
print("****************************************************")
topic = "10. Using your segmentation function"; print("** %s" % topic)
print("****************************************************")
# Setting a special configuration
plt.rcParams.update({'axes.labelsize': 6, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 
                     'legend.fontsize': 5, 'font.size': 5})


Ho = 'Ho: The converse rate is not affected by the email personalization.\n'

print('---------------------------------------------language_displayed')
# Use ab_segmentation on language displayed
ab_segmentation('language_displayed', Ho) #Without graphs
#ab_segmentation_plot('language_displayed', Ho) #With graphs

print('---------------------------------------------age_group')
# Use ab_segmentation on age group
#ab_segmentation('age_group', Ho) #Without graphs
ab_segmentation_plot('age_group', Ho) #With graphs

# Returning to stablished global configuration
plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                     'legend.fontsize': 6, 'font.size': 6})



print("****************************************************")
topic = "11. Wrap-up"; print("** %s" % topic)
print("****************************************************")

print('---------------------------------------------Explore')
print('---------------------------------------------Explore')
    
    
print("\n\n****************************************************")
print("** END                                            **")
print("****************************************************")
plt.style.use('default')