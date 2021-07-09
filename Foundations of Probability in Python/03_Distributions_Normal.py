# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:40:32 2020

@author: jacqueline.cortez
Subject: Practicing Statistics Interview Questions in Python
Chapter 3: Important probability distributions
    Until now we've been working with binomial distributions, but there are many probability 
    distributions a random variable can take. In this chapter we'll introduce three more that 
    are related to the binomial distribution: the normal, Poisson, and geometric distributions.
"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "Importing libraries"; print("** %s\n" % topic)

import numpy                         as np                                    #For making operations in lists
import matplotlib.pyplot             as plt                                   #For creating charts
import seaborn                       as sns                                   #For visualizing data

from scipy.stats                     import norm                              #Generate normal data
from scipy.stats                     import poisson                           #To generate poisson distribution.

print("****************************************************")
topic = "Preparing the environment"; print("** %s\n" % topic)

SEED = 13
np.random.seed(SEED) 

print("****************************************************")
topic = "3. Plotting normal distributions"; print("** %s\n" % topic)

# Create the sample using norm.rvs()
mean = 3.15
std  = 1.5
size = 10000

sample = norm.rvs(loc=mean, scale=std, size=size, random_state=SEED)
#ylim = norm.pdf(mean, loc=mean, scale=std)
median = np.median(sample)

# Plot the sample
sns.distplot(sample)
plt.xlabel('Sample'); plt.ylabel('Probability Distribution (PDF)'); # Labeling the axis.
plt.title("Customer spending at Restaurant with Mean:\${:,.2f} and std:\${:,.2f}".format(mean, std), color='red')

plt.axvspan(xmin=mean - 3*std, xmax=mean + 3*std, color = '#D9D9D9', alpha = 0.5, label="Mean ± 3 STD (99.7%)") # Plot shaded area for interval
plt.axvspan(xmin=mean - 2*std, xmax=mean + 2*std, color = '#A6A6A6', alpha = 0.5, label="Mean ± 2 STD (95%)") # Plot shaded area for interval
plt.axvspan(xmin=mean - std,   xmax=mean + std,   color = '#7F7F7F', alpha = 0.5, label="Mean ± 1 STD (68%)") # Plot shaded area for interval

#plt.fill_between([mean - 3*std, mean + 3*std], [0, 0], [ylim, ylim], 
#                 linestyle="--", color='#D9D9D9', alpha=0.5, label="Mean ± 3 STD (99.7%)")
#plt.fill_between([mean - 2*std, mean + 2*std], [0, 0], [ylim, ylim], 
#                 linestyle="--", color='#A6A6A6', alpha=0.5, label="Mean ± 2 STD (97%)")
#plt.fill_between([mean - 1*std, mean + 1*std], [0, 0], [ylim, ylim], 
#                 linestyle="--", color='#7F7F7F', alpha=0.5, label="Mean ± 1 STD (68%)")
plt.axvline(x=mean, color='b', label='Mean', linestyle='-', linewidth=2)
#plt.axvline(x=median, color='r', label='Median', linestyle='--', linewidth=2) # Add vertical lines for the median and mean

plt.legend(loc='best', fontsize='small')
plt.suptitle(topic, color='navy');  # Setting the titles.
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5);
plt.show()


print("****************************************************")
topic = "6. Restaurant spending example"; print("** %s\n" % topic)

print("What is the probability that a customer will spend $3 or less?")
# Probability of spending $3 or less
spending = norm.cdf(3, loc=3.15, scale=1.5)
print("{:,.2%}".format(spending))

print("\nWhat is the probability that a customer will spend more than $5?")
# Probability of spending more than $5
spending = norm.sf(5, loc=3.15, scale=1.5)
print("{:,.2%}".format(spending))

print("\nWhat is the probability that a customer will spend more than $2.15 and $4.15 or less?")
# Probability of spending more than $2.15 and $4.15 or less
spending_4 = norm.cdf(4.15, loc=3.15, scale=1.5)
spending_2 = norm.cdf(2.15, loc=3.15, scale=1.5)
print("{:,.2%}".format(spending_4 - spending_2))

print("\nWhat is the probability that a customer will spend $2.15 or less or more than $4.15?")
# Probability of spending $2.15 or less or more than $4.15
spending_2 = norm.cdf(2.15, loc=3.15, scale=1.5)
spending_over_4 = norm.sf(4.15, loc=3.15, scale=1.5) 
print("{:,.2%}".format(spending_2 + spending_over_4))

print("****************************************************")
topic = "7. Smartphone battery example"; print("** %s\n" % topic)

# Create the sample using norm.rvs()
mean = 5
std  = 1.5
size = 10000

sample = norm.rvs(loc=mean, scale=std, size=size, random_state=SEED)
#ylim = norm.pdf(mean, loc=mean, scale=std)
median = np.median(sample)

# Plot the sample
plt.figure()
sns.distplot(sample)
plt.xlabel('Sample'); plt.ylabel('Probability Distribution (PDF)'); # Labeling the axis.
plt.title("Period of times between charges of the Smartphone battery\nwith Mean:{:,.1f}hrs. and std:{:,.1f}hrs.".format(mean, std), color='red')

plt.axvspan(xmin=mean - 3*std, xmax=mean + 3*std, color = '#D9D9D9', alpha = 0.5, label="Mean ± 3 STD (99.7%)") # Plot shaded area for interval
plt.axvspan(xmin=mean - 2*std, xmax=mean + 2*std, color = '#A6A6A6', alpha = 0.5, label="Mean ± 2 STD (95%)") # Plot shaded area for interval
plt.axvspan(xmin=mean - std,   xmax=mean + std,   color = '#7F7F7F', alpha = 0.5, label="Mean ± 1 STD (68%)") # Plot shaded area for interval

#plt.fill_between([mean - 3*std, mean + 3*std], [0, 0], [ylim, ylim], 
#                 linestyle="--", color='#D9D9D9', alpha=0.5, label="Mean ± 3 STD (99.7%)")
#plt.fill_between([mean - 2*std, mean + 2*std], [0, 0], [ylim, ylim], 
#                 linestyle="--", color='#A6A6A6', alpha=0.5, label="Mean ± 2 STD (97%)")
#plt.fill_between([mean - 1*std, mean + 1*std], [0, 0], [ylim, ylim], 
#                 linestyle="--", color='#7F7F7F', alpha=0.5, label="Mean ± 1 STD (68%)")
plt.axvline(x=mean, color='b', label='Mean', linestyle='-', linewidth=2)
#plt.axvline(x=median, color='r', label='Median', linestyle='--', linewidth=2) # Add vertical lines for the median and mean

plt.legend(loc='best', fontsize='small')
plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None);
plt.show()



print("What is the probability that the battery will last less than 3 hours?")
# Probability that battery will last less than 3 hours
less_than_3h = norm.cdf(3, loc=5, scale=1.5)
print("{:,.2%}".format(less_than_3h))


print("\nWhat is the probability that the battery will last more than 3 hours?")
# Probability that battery will last more than 3 hours
more_than_3h = norm.sf(3, loc=5, scale=1.5)
print("{:,.2%}".format(more_than_3h))


print("\nWhat is the probability that the battery will last between 5 and 7 hours?")
# Probability that battery will last between 5 and 7 hours
P_less_than_7h = norm.cdf(7, loc=5, scale=1.5)
P_less_than_5h = norm.cdf(5, loc=5, scale=1.5)
print("{:,.2%}".format(P_less_than_7h - P_less_than_5h))


print("****************************************************")
topic = "8. Adults' heights example"; print("** %s\n" % topic)

# Create the sample using norm.rvs()
mean_male     = 70
std_male      = 4
mean_female   = 65
std_female    = 3.5
size          = 500000

sample_male   = norm.rvs(loc=mean_male, scale=std_male, size=size, random_state=SEED)
sample_female = norm.rvs(loc=mean_female, scale=std_female, size=size, random_state=SEED)

# Plot the sample
plt.figure()
sns.distplot(sample_female, color='blue', label='Female', hist=False, kde_kws={"linestyle": ":"})
sns.distplot(sample_male, color='brown', label='Male', hist=False, kde_kws={"linestyle": "--"})
plt.xlabel('Sample'); plt.ylabel('Probability Distribution (PDF)'); # Labeling the axis.
plt.title("Adults' height between 18 and 35 years old", color='red')

plt.legend(loc='best', fontsize='small')
plt.suptitle(topic, color='navy');  # Setting the titles.
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=None, hspace=None);
plt.show()


print("Print the range of female heights one standard deviation from the mean.")
# Values one standard deviation from mean height for females
interval = norm.interval(.68, loc=65, scale=3.5)
print("From {:,.2f} to {:,.2f} inches.".format(interval[0],interval[1]))


print("\nPrint the value where the tallest males fall with 0.01 probability.")
# Value where the tallest males fall with 0.01 probability
tallest = norm.ppf((1-.01), loc=70, scale=4)
print("{:,.2f} inches.".format(tallest))


print("\nPrint the probability of being taller than 73 inches for a male and for a female.")
# Probability of being taller than 73 inches for males and females
P_taller_female = norm.sf(73, loc=65, scale=3.5)
P_taller_male = norm.sf(73, loc=70, scale=4)
print("{:,.2%} for females and {:,.2%} for males.".format(P_taller_female, P_taller_male))


print("\nPrint the probability of being shorter than 61 inches for a male and for a female.")
# Probability of being shorter than 61 inches for males and females
P_shorter_female = norm.cdf(61, loc=65, scale=3.5)
P_shorter_male = norm.cdf(61, loc=70, scale=4)
print("{:,.2%} for females and {:,.2%} for males.".format(P_shorter_female, P_shorter_male))


print("****************************************************")
print("** END                                            **")
print("****************************************************")
