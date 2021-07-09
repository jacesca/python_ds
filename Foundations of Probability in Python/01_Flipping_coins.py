# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:36:32 2020

@author: jacqueline.cortez
Subject: Practicing Statistics Interview Questions in Python
Chapter 1: Let's start flipping coins
    A coin flip is the classic example of a random experiment. The possible outcomes are 
    heads or tails. This type of experiment, known as a Bernoulli or binomial trial, allows 
    us to study problems with two possible outcomes, like “yes” or “no” and “vote” or “no vote.” 
    This chapter introduces Bernoulli experiments, binomial distributions to model multiple 
    Bernoulli trials, and probability simulations with the scipy library.
"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "Importing libraries"; print("** %s\n" % topic)

import numpy                         as np                                    #For making operations in lists

from scipy.stats                     import bernoulli                         #Generate bernoulli data
from scipy.stats                     import binom                             #Generate binomial data
from scipy.stats                     import describe                          #To get the arithmetic statistics.


print("****************************************************")
topic = "Preparing the environment"; print("** %s\n" % topic)

SEED = 42
np.random.seed(SEED) 
    

print("****************************************************")
topic = "2. Flipping coins"; print("** %s\n" % topic)

# Simulate one coin flip with 35% chance of getting heads
coin_flip = bernoulli.rvs(p=.35, size=1) #, random_state=SEED
print("Simulating one coin flip with 35% chance of getting heads: {} heads obtained.".format(coin_flip))

# Simulate ten coin flips and get the number of heads
ten_coin_flips = bernoulli.rvs(p=.35, size=10)
coin_flips_sum = sum(ten_coin_flips)
print("Simulating ten coin flips with 35% chance of getting heads: {} heads obtained.".format(coin_flips_sum))

# Simulate five coin flips and get the number of heads
five_coin_flips = bernoulli.rvs(p=.5, size=5)
coin_flips_sum = sum(five_coin_flips)
print("Simulating five coin flips with 50% chance of getting heads: {} heads obtained.".format(coin_flips_sum))
 

    
print("****************************************************")
topic = "3. Using binom to flip even more coins"; print("** %s\n" % topic)

# Simulate 20 trials of 10 coin flips 
draws = binom.rvs(n=10, p=.35, size=20)
print("Simulating 20 trials of 10 coin flips with 35% chance of getting heads:\n",draws)
 
    
print("****************************************************")
topic = "5. Predicting the probability of defects"; print("** %s\n" % topic)

# What is the probability of getting more than 20 heads from a fair coin after 30 coin flips?
np.random.seed(42)
prob_more_20 = binom.sf(k=20, n=30, p=0.5)
print("Probability of getting more than 20 heads from a fair coin after 30 coin flips: {:,.2%}.\n".format(prob_more_20))

# Probability of getting exactly 1 defective component in 50 samples of supplier.
prob_one_defect = binom.pmf(k=1, n=50, p=.02)
print("Probability of getting exactly 1 defective component in 50 samples of supplier with 2% defect rate: {:,.2%}.\n".format(prob_one_defect))

# Probability of not getting any defective components
prob_no_defects = binom.pmf(k=0, n=50, p=.02)
print("Probability of not getting any defective component in 50 samples of supplier with 2% defect rate: {:,.2%}.\n".format(prob_no_defects))

# Probability of getting 2 or less defective components
prob_two_or_less_defects = binom.cdf(k=2, n=50, p=0.02)
print("Probability of getting 2 or less defective component in 50 samples of supplier with 2% defect rate: {:,.2%}.\n".format(prob_two_or_less_defects))

    
print("****************************************************")
topic = "6. Predicting employment status"; print("** %s\n" % topic)

# Calculate the probability of getting exactly 5 yes responses
prob_five_yes = binom.pmf(k=5, n=8, p=.65)
print("Calculate the probability of getting exactly 5 yes responses.\n{:,.2%}.\n".format(prob_five_yes))

# Calculate the probability of getting 3 or less yes responses
prob_three_or_less_no = 1-binom.cdf(k=3, n=8, p=0.65)
print("Calculate the probability of getting 3 or fewer yes responses.\n.{:,.2%}\n".format(prob_three_or_less_no))

# Calculate the probability of getting more than 3 yes responses
prob_more_than_three_yes = binom.sf(k=3, n=8, p=0.65)
print("Calculate the probability of getting more than 3 yes responses.\n.{:,.2%}\n".format(prob_more_than_three_yes))
 
    
print("****************************************************")
topic = "7. Predicting burglary conviction rate"; print("** %s\n" % topic)

# What is the probability of solving 4 burglaries?
four_solved = binom.pmf(k=4, n=9, p=.2) #, random_state=SEED
print("What is the probability of solving 4 burglaries?\n{:,.2%}.\n".format(four_solved))

# What is the probability of solving more than 3 burglaries?
more_than_three_solved = binom.sf(k=3, n=9, p=.2)
print("What is the probability of solving more than 3 burglaries?\n{:,.2%}.\n".format(more_than_three_solved))

# What is the probability of solving 2 or 3 burglaries?
two_or_three_solved = binom.pmf(k=2, n=9, p=.2) + binom.pmf(k=3, n=9, p=.2)
print("What is the probability of solving 2 or 3 burglaries?\n{:,.2%}.\n".format(two_or_three_solved))

# What is the probability of solving 1 or fewer or more than 7 burglaries?
tail_probabilities = binom.cdf(k=1, n=9, p=.2) + binom.sf(k=7, n=9, p=.2)
print("What is the probability of solving 1 or fewer or more than 7 burglaries?\n{:,.2%}.\n".format(tail_probabilities)) 
    

print("****************************************************")
topic = "10. Calculating the sample mean"; print("** %s\n" % topic)

# Sample mean from a generated sample of 100 fair coin flips
sample_of_100_flips = binom.rvs(n=1, p=0.5, size=100)
sample_mean_100_flips = describe(sample_of_100_flips).mean
print("Generate a sample of 100 fair coin flips using .rvs() and calculate the sample mean using describe(): \n{:,.2%}\n".format(sample_mean_100_flips))

# Sample mean from a generated sample of 1,000 fair coin flips
sample_mean_1000_flips = describe(binom.rvs(n=1, p=0.5, size=1000)).mean
print("Generate a sample of 1,000 fair coin flips and calculate the sample mean: \n{:,.2%}\n".format(sample_mean_1000_flips))

# Sample mean from a generated sample of 2,000 fair coin flips
sample_mean_2000_flips = describe(binom.rvs(n=1, p=0.5, size=2000)).mean
print("Generate a sample of 2,000 fair coin flips and calculate the sample mean: \n{:,.2%}\n".format(sample_mean_2000_flips)) 

    
print("****************************************************")
topic = "11. Checking the result"; print("** %s\n" % topic)

n=10; p=0.3; size=2000;
sample = binom.rvs(n=n, p=p, size=size)

# Calculate the sample mean and variance from the sample variable
sample_describe = describe(sample)
print("Calculate the sample mean and variance from the sample variable: \n{}".format(sample_describe ))

# Calculate the sample mean using the values of n and p and sample variance using the value of 1-p
mean = n*p
variance = mean*(1-p)
print("\nCalculate the sample mean using the values of n and p and sample variance using the value of 1-p")
print("Mean = ", mean)
print("Variance = ", variance)

# Calculate the sample mean and variance for 10 coin flips with p=0.3
binom_stats = binom.stats(n=n, p=p)
print("\nCalculate the sample mean and variance for 10 coin flips with p=0.3: ", binom_stats)
 
    
print("****************************************************")
topic = "12. Calculating the mean and variance of a sample"; print("** %s\n" % topic)

averages = []
variances = []

for i in range(0, 1500):
    # 10 trials of 10 coin flips with 25% probability of heads
    sample = binom.rvs(n=10, p=.25, size=10)
    # Mean and variance of the values in the sample variable
    averages.append(describe(sample).mean)
    variances.append(describe(sample).variance)
    
# Calculate the mean of the averages variable
print("Mean {}".format(describe(averages).mean))

# Calculate the mean of the variances variable
print("Variance {}".format(describe(variances).mean))
    
print("****************************************************")
print("** END                                            **")
print("****************************************************")
