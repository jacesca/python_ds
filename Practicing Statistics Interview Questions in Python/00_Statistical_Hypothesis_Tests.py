# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 19:48:06 2020

@author: jacqueline.cortez
Subject: 17 Statistical Hypothesis Tests in Python (Cheat Sheet)
Source: https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/
"""

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "Preparing the environment"; print("** %s\n" % topic)

import numpy                          as np                                   #For making operations in lists
SEED = 123
np.random.seed(SEED)

print("****************************************************")
topic = "1. Normality Tests"; print("** %s\n" % topic)

# This section lists statistical tests that you can use to check if your 
# data has a Gaussian distribution.

#***********************************************************************
#**** Shapiro-Wilk Test
#***********************************************************************
#Tests whether a data sample has a Gaussian distribution.
#
#Assumptions:
#Observations in each sample are independent and identically distributed.
#
#Interpretation:
#H0: the sample has a Gaussian distribution.
#H1: the sample does not have a Gaussian distribution.
#
#Python Code:

from scipy.stats                     import shapiro                           #For Shapiro-Wilk Normality Test. Tests whether a data sample has a Gaussian distribution.
print("SHAPIRO-WILK TEST:")
data = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
stat, p = shapiro(data)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably Gaussian (p > 0.05).')
else:
	print('Probably not Gaussian (p <= 0.05).')

#More information:
#A Gentle Introduction to Normality Tests in Python -->https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
#scipy.stats.shapiro --> https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html
#Shapiro-Wilk test on Wikipedia --> https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test


#***********************************************************************
#**** D’Agostino’s K^2 Test
#***********************************************************************
#Tests whether a data sample has a Gaussian distribution.
#
#Assumptions:
#Observations in each sample are independent and identically distributed.
#
#Interpretation
#H0: the sample has a Gaussian distribution.
#H1: the sample does not have a Gaussian distribution.
#
#Python Code:
    
from scipy.stats                     import normaltest                        #For D'Agostino's K^2 Normality Test. Tests whether a data sample has a Gaussian distribution.
print("\nD'AGOSTINO'S K^2 TEST:")
data = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869, 0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
stat, p = normaltest(data)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably Gaussian (p > 0.05).')
else:
	print('Probably not Gaussian (p <= 0.05).')
"""
Give a warning if the sample has n<20.
C:\Anaconda3\lib\site-packages\scipy\stats\stats.py:1535: 
UserWarning: kurtosistest only valid for n>=20 ... continuing anyway, n=10 "anyway, n=%i" % int(n))
"""
#More information:
#A Gentle Introduction to Normality Tests in Python --> https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
#scipy.stats.normaltest --> https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html
#D’Agostino’s K-squared test on Wikipedia --> https://en.wikipedia.org/wiki/D%27Agostino%27s_K-squared_test


#***********************************************************************
#**** Anderson-Darling Test
#***********************************************************************
#Tests whether a data sample has a Gaussian distribution.
#
#Assumptions:
#Observations in each sample are independent and identically distributed.
#
#Interpretation:
#H0: the sample has a Gaussian distribution.
#H1: the sample does not have a Gaussian distribution.
#
#Python Code

from scipy.stats                     import anderson                          #For Anderson-Darling Normality Test. Tests whether a data sample has a Gaussian distribution.
print("\nANDERSON-DARLING TEST:")
data = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
result = anderson(data)
print('stat=%.3f' % (result.statistic))
for i in range(len(result.critical_values)):
	sl, cv = result.significance_level[i], result.critical_values[i]
	if result.statistic < cv:
		print('Probably Gaussian at the %.1f%% level (statistic < critical value).' % (sl))
	else:
		print('Probably not Gaussian at the %.1f%% level (statistic >= critical value).' % (sl))

#More Information:
#A Gentle Introduction to Normality Tests in Python --> https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
#scipy.stats.anderson --> https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html
#Anderson-Darling test on Wikipedia --> https://en.wikipedia.org/wiki/Anderson%E2%80%93Darling_test

print("\n****************************************************")
topic = "2. Correlation Tests"; print("** %s\n" % topic)
#This section lists statistical tests that you can use to check if two samples are related.

#***********************************************************************
#**** Pearson’s Correlation Coefficient
#***********************************************************************
#Tests whether two samples have a linear relationship.
#
#Assumptions:
#Observations in each sample are independent and identically distributed.
#Observations in each sample are normally distributed.
#Observations in each sample have the same variance.
#
#Interpretation
#H0: the two samples are independent.
#H1: there is a dependency between the samples.
#
#Python Code:

from scipy.stats                     import pearsonr                          #For learning machine. For Pearson's Correlation test. To check if two samples are related.
print("PEARSON'S CORRELATION COEFFICIENT:")
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [0.353, 3.517, 0.125, -7.545, -0.555, -1.536, 3.350, -1.578, -3.537, -1.579]
stat, p = pearsonr(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent (p > 0.05).')
else:
	print('Probably dependent (p <= 0.05).')

#More Information:
#How to Calculate Correlation Between Variables in Python --> https://machinelearningmastery.com/how-to-use-correlation-to-understand-the-relationship-between-variables/
#scipy.stats.pearsonr https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
#Pearson’s correlation coefficient on Wikipedia --> https://en.wikipedia.org/wiki/Pearson_correlation_coefficient


#***********************************************************************
#**** Spearman’s Rank Correlation
#***********************************************************************
#Tests whether two samples have a monotonic relationship.
#
#Assumptions:
#Observations in each sample are independent and identically distributed.
#Observations in each sample can be ranked.
#
#Interpretation:
#H0: the two samples are independent.
#H1: there is a dependency between the samples.
#
#Python Code:

from scipy.stats                     import spearmanr                         #For Spearman's Rank Correlation Test.  To check if two samples are related.
print("\nSPEARMAN'S RANK CORRELATION:")
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [0.353, 3.517, 0.125, -7.545, -0.555, -1.536, 3.350, -1.578, -3.537, -1.579]
stat, p = spearmanr(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent (p > 0.05).')
else:
	print('Probably dependent (p <= 0.05).')

#More Information:
#How to Calculate Nonparametric Rank Correlation in Python --> https://machinelearningmastery.com/how-to-calculate-nonparametric-rank-correlation-in-python/
#scipy.stats.spearmanr --> https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
#Spearman’s rank correlation coefficient on Wikipedia --> https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
    

#***********************************************************************
#**** Kendall’s Rank Correlation
#***********************************************************************
#Tests whether two samples have a monotonic relationship.
#
#Assumptions:
#Observations in each sample are independent and identically distributed.
#Observations in each sample can be ranked.
#
#Interpretation:
#H0: the two samples are independent.
#H1: there is a dependency between the samples.
#
#Python Code

from scipy.stats                     import kendalltau                        #For Kendall's Rank Correlation Test. To check if two samples are related.
print("\nKENDALL'S RANK CORRELATION:")
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [0.353, 3.517, 0.125, -7.545, -0.555, -1.536, 3.350, -1.578, -3.537, -1.579]
stat, p = kendalltau(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent (p > 0.05).')
else:
	print('Probably dependent (p <= 0.05).')

#More Information:
#How to Calculate Nonparametric Rank Correlation in Python --> https://machinelearningmastery.com/how-to-calculate-nonparametric-rank-correlation-in-python/
#scipy.stats.kendalltau --> https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html
#Kendall rank correlation coefficient on Wikipedia --> https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient


#***********************************************************************
#**** Chi-Squared Test
#***********************************************************************
#Tests whether two categorical variables are related or independent.
#
#Assumptions:
#Observations used in the calculation of the contingency table are independent.
#25 or more examples in each cell of the contingency table.
#
#Interpretation:
#H0: the two samples are independent.
#H1: there is a dependency between the samples.
#
#Python Code

from scipy.stats                     import chi2_contingency                  #For Chi-Squared Test. Tests whether two categorical variables are related or independent
print("\nCHI-SQUARED TEST:")
table = [[10, 20, 30],[6,  9,  17]]
stat, p, dof, expected = chi2_contingency(table)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent (p > 0.05).')
else:
	print('Probably dependent (p <= 0.05).')
"""
Give an error if the sample has negatives values.
C:\Anaconda3\lib\site-packages\scipy\stats\contingency.py, line 245, in chi2_contingency
raise ValueError("All values in `observed` must be nonnegative.")
ValueError: All values in `observed` must be nonnegative.
"""

#More Information:
#A Gentle Introduction to the Chi-Squared Test for Machine Learning --> https://machinelearningmastery.com/chi-squared-test-for-machine-learning/
#scipy.stats.chi2_contingency --> https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
#Chi-Squared test on Wikipedia --> https://en.wikipedia.org/wiki/Chi-squared_test

print("\n****************************************************")
topic = "3. Stationary Tests"; print("** %s\n" % topic)
#This section lists statistical tests that you can use to check if a time series is stationary or not.

#***********************************************************************
#**** Augmented Dickey-Fuller Unit Root Test
#***********************************************************************
#Tests whether a time series has a unit root, e.g. has a trend or more generally is autoregressive.
#
#Assumptions:
#Observations in are temporally ordered.
#
#Interpretation:
#H0: a unit root is present (series is non-stationary).
#H1: a unit root is not present (series is stationary).
#
#Python Code:

from statsmodels.tsa.stattools       import adfuller                          #For Augmented Dickey-Fuller unit root test. Tests whether a time series has a unit root, e.g. has a trend or more generally is autoregressive. Tests that you can use to check if a time series is stationary or not.
print("STATIONARY TEST:")
data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
stat, p, lags, obs, crit, t = adfuller(data)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably not Stationary (p > 0.05).')
else:
	print('Probably Stationary (p <= 0.05).')

#More Information:
#How to Check if Time Series Data is Stationary with Python --> https://machinelearningmastery.com/time-series-data-stationary-python/
#statsmodels.tsa.stattools.adfuller API. --> https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html
#Augmented Dickey–Fuller test, Wikipedia. --> https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test


#***********************************************************************
#**** Kwiatkowski-Phillips-Schmidt-Shin
#***********************************************************************
#Tests whether a time series is trend stationary or not.
#
#Assumptions:
#Observations in are temporally ordered.
#
#Interpretation:
#H0: the time series is not trend-stationary.
#H1: the time series is trend-stationary.
#
#Python Code:

from statsmodels.tsa.stattools       import kpss                              #For Kwiatkowski-Phillips-Schmidt-Shin test. Tests whether a time series is trend stationary or not.
print("\nKWIATKOWSKI-PHILLIPS-SCHMIDT-SHIN TEST:")
data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
stat, p, lags, crit = kpss(data, nlags='legacy')
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably not Stationary (p > 0.05).')
else:
	print('Probably Stationary (p <= 0.05).')
"""
Give a warning if we don't use the nlags attribute
C:\Anaconda3\lib\site-packages\statsmodels\tsa\stattools.py:1661: 
FutureWarning: The behavior of using lags=None will change in the next release. 
Currently lags=None is the same as lags='legacy', and so a sample-size lag length is used. 
After the next release, the default will change to be the same as lags='auto' which uses an automatic lag length 
selection method. To silence this warning, either use 'auto' or 'legacy'.

FutureWarning: the 'lags'' keyword is deprecated, use 'nlags' instead.
"""

#More Information:
#statsmodels.tsa.stattools.kpss API. --> https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html#statsmodels.tsa.stattools.kpss
#KPSS test, Wikipedia. --> https://en.wikipedia.org/wiki/KPSS_test

print("\n****************************************************")
topic = "4. Parametric Statistical Hypothesis Tests"; print("** %s\n" % topic)
#This section lists statistical tests that you can use to compare data samples.

#***********************************************************************
#**** Student’s t-test
#***********************************************************************
#Tests whether the means of two independent samples are significantly different.
#
#Assumptions:
#Observations in each sample are independent and identically distributed.
#Observations in each sample are normally distributed.
#Observations in each sample have the same variance.
#
#Interpretation:
#H0: the means of the samples are equal.
#H1: the means of the samples are unequal.
#
#Python Code:

from scipy.stats                     import ttest_ind                         #For Student's t-test. Tests whether the means of two independent samples are significantly different.
print("STUDENT’S T-TEST:")
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
stat, p = ttest_ind(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution (p > 0.05).')
else:
	print('Probably different distributions (p <= 0.05).')

#More Information:
#How to Calculate Parametric Statistical Hypothesis Tests in Python --> https://machinelearningmastery.com/parametric-statistical-significance-tests-in-python/
#scipy.stats.ttest_ind --> https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
#Student’s t-test on Wikipedia --> https://en.wikipedia.org/wiki/Student%27s_t-test
    
    
#***********************************************************************
#**** Paired Student’s t-test
#***********************************************************************
#Tests whether the means of two paired samples are significantly different.
#
#Assumptions:
#Observations in each sample are independent and identically distributed.
#Observations in each sample are normally distributed.
#Observations in each sample have the same variance.
#Observations across each sample are paired.
#
#Interpretation:
#H0: the means of the samples are equal.
#H1: the means of the samples are unequal.
#
#Python Code:

from scipy.stats                     import ttest_rel                         #For Paired Student's t-test. Tests whether the means of two paired samples are significantly different.
print("\nPAIRED STUDENT’S T-TEST:")
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
stat, p = ttest_rel(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution (p > 0.05).')
else:
	print('Probably different distributions (p <= 0.05).')

#More Information:
#How to Calculate Parametric Statistical Hypothesis Tests in Python
#scipy.stats.ttest_rel
#Student’s t-test on Wikipedia


#***********************************************************************
#**** Analysis of Variance Test (ANOVA)
#***********************************************************************
#Tests whether the means of two or more independent samples are significantly different.
#
#Assumptions:
#Observations in each sample are independent and identically distributed (iid).
#Observations in each sample are normally distributed.
#Observations in each sample have the same variance.
#
#Interpretation:
#H0: the means of the samples are equal.
#H1: one or more of the means of the samples are unequal.
#
#Python Code:

from scipy.stats                     import f_oneway                          #For Analysis of Variance Test. Tests whether the means of two or more independent samples are significantly different.
print("\nANALYSIS OF VARIANCE TEST (ANOVA):")
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
data3 = [-0.208, 0.696, 0.928, -1.148, -0.213, 0.229, 0.137, 0.269, -0.870, -1.204]
stat, p = f_oneway(data1, data2, data3)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution (p > 0.05).')
else:
	print('Probably different distributions (p <= 0.05).')

#More Information
#How to Calculate Parametric Statistical Hypothesis Tests in Python --> https://machinelearningmastery.com/parametric-statistical-significance-tests-in-python/
#scipy.stats.f_oneway --> https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html
#Analysis of variance on Wikipedia -- https://en.wikipedia.org/wiki/Analysis_of_variance


#***********************************************************************
#**** Repeated Measures ANOVA Test
#***********************************************************************
#Tests whether the means of two or more paired samples are significantly different.
#
#Assumptions:
#Observations in each sample are independent and identically distributed.
#Observations in each sample are normally distributed.
#Observations in each sample have the same variance.
#Observations across each sample are paired.
#
#Interpretation:
#H0: the means of the samples are equal.
#H1: one or more of the means of the samples are unequal.
#
#Python Code:
#
#Currently not supported in Python.
#
#More Information:
#How to Calculate Parametric Statistical Hypothesis Tests in Python --> https://machinelearningmastery.com/parametric-statistical-significance-tests-in-python/
#Analysis of variance on Wikipedia --> https://en.wikipedia.org/wiki/Analysis_of_variance

print("\n****************************************************")
topic = "5. Nonparametric Statistical Hypothesis Tests"; print("** %s\n" % topic)

#***********************************************************************
#**** Mann-Whitney U Test
#***********************************************************************
#Tests whether the distributions of two independent samples are equal or not.
#
#Assumptions:
#Observations in each sample are independent and identically distributed (iid).
#Observations in each sample can be ranked.
#
#Interpretation:
#H0: the distributions of both samples are equal.
#H1: the distributions of both samples are not equal.
#
#Python Code:

from scipy.stats                     import mannwhitneyu                      #For Mann-Whitney U Test. Tests whether the distributions of two independent samples are equal or not.
print("MANN-WHITNEY U TEST:")
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
stat, p = mannwhitneyu(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution (p > 0.05)')
else:
	print('Probably different distributions (p <= 0.05)')

#More Information:
#How to Calculate Nonparametric Statistical Hypothesis Tests in Python --> https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/
#scipy.stats.mannwhitneyu --> https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
#Mann-Whitney U test on Wikipedia --> https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test


#***********************************************************************
#**** Wilcoxon Signed-Rank Test
#***********************************************************************
#Tests whether the distributions of two paired samples are equal or not.
#
#Assumptions:
#Observations in each sample are independent and identically distributed.
#Observations in each sample can be ranked.
#Observations across each sample are paired.
#
#Interpretation:
#H0: the distributions of both samples are equal.
#H1: the distributions of both samples are not equal.
#
#Python Code:

from scipy.stats                     import wilcoxon                          #For Wilcoxon Signed-Rank Test. Tests whether the distributions of two paired samples are equal or not.
print("\nWILCOXON SIGNED-RANK TEST:")
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
stat, p = wilcoxon(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution (p > 0.05).')
else:
	print('Probably different distributions (p <= 0.05).')

#More Information:
#How to Calculate Nonparametric Statistical Hypothesis Tests in Python --> https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/
#scipy.stats.wilcoxon --> https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
#Wilcoxon signed-rank test on Wikipedia --> https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test


#***********************************************************************
#**** Kruskal-Wallis H Test
#***********************************************************************
#Tests whether the distributions of two or more independent samples are equal or not.
#
#Assumptions:
#Observations in each sample are independent and identically distributed.
#Observations in each sample can be ranked.
#
#Interpretation:
#H0: the distributions of all samples are equal.
#H1: the distributions of one or more samples are not equal.
#
#Python Code:

from scipy.stats                     import kruskal                           #For Kruskal-Wallis H Test. Tests whether the distributions of two or more independent samples are equal or not.
print("\nKRUSKAL-WALLIS H TEST:")
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
stat, p = kruskal(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution (p > 0.05).')
else:
	print('Probably different distributions (p <= 0.05).')

#More Information:
#How to Calculate Nonparametric Statistical Hypothesis Tests in Python --> https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/
#scipy.stats.kruskal --> https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html
#Kruskal-Wallis one-way analysis of variance on Wikipedia --> https://en.wikipedia.org/wiki/Kruskal%E2%80%93Wallis_one-way_analysis_of_variance


#***********************************************************************
#**** Friedman Test
#***********************************************************************
#Tests whether the distributions of two or more paired samples are equal or not.
#
#Assumptions:
#Observations in each sample are independent and identically distributed.
#Observations in each sample can be ranked.
#Observations across each sample are paired.
#
#Interpretation:
#H0: the distributions of all samples are equal.
#H1: the distributions of one or more samples are not equal.
#
#Python Code:

from scipy.stats                     import friedmanchisquare                 #For Friedman Test. Tests whether the distributions of two or more paired samples are equal or not.
print("\nFRIEDMAN TEST:")
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
data3 = [-0.208, 0.696, 0.928, -1.148, -0.213, 0.229, 0.137, 0.269, -0.870, -1.204]
stat, p = friedmanchisquare(data1, data2, data3)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution (p > 0.05).')
else:
	print('Probably different distributions (p <= 0.05).')

#More Information:
#How to Calculate Nonparametric Statistical Hypothesis Tests in Python --> https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/
#scipy.stats.friedmanchisquare --> https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html
#Friedman test on Wikipedia --> https://en.wikipedia.org/wiki/Friedman_test
"""
Further Reading
This section provides more resources on the topic if you are looking to go deeper:

A Gentle Introduction to Normality Tests in Python --> https://machinelearningmastery.com/how-to-use-correlation-to-understand-the-relationship-between-variables/
How to Use Correlation to Understand the Relationship Between Variables --> https://machinelearningmastery.com/how-to-use-correlation-to-understand-the-relationship-between-variables/
How to Use Parametric Statistical Significance Tests in Python --> https://machinelearningmastery.com/parametric-statistical-significance-tests-in-python/
A Gentle Introduction to Statistical Hypothesis Tests --> https://machinelearningmastery.com/statistical-hypothesis-tests/

Summary
In this tutorial, you discovered the key statistical hypothesis tests that you may need to use 
in a machine learning project.

Specifically, you learned:
- The types of tests to use in different circumstances, such as normality checking, relationships between 
  variables, and differences between samples.
- The key assumptions for each test and how to interpret the test result.
- How to implement the test using the Python API.
"""
print("\n****************************************************")
print("** END                                            **")
print("****************************************************")

#import contextily                                                             #To add a background web map to our plot
#import inspect                                                                #Used to get the code inside a function
#import folium                                                                 #To create map street folium.__version__'0.10.0'
#import geopandas         as gpd                                               #For working with geospatial data 
#import matplotlib        as mpl                                               #To format numbers with ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}')) or ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
#import matplotlib.dates  as mdates                                            #For providing sophisticated date plotting capabilities
#import matplotlib.pyplot as plt                                               #For creating charts
#import numpy             as np                                                #For making operations in lists
#import pandas            as pd                                                #For loading tabular data
#import seaborn           as sns                                               #For visualizing data
#import pprint                                                                 #Import pprint to format disctionary output
#import missingno         as msno                                              #Missing data visualization module for Python

#import os                                                                     #To raise an html page in python command
#import tempfile                                                               #To raise an html page in python command
#import webbrowser                                                             #To raise an html page in python command  

#import calendar                                                               #For accesing to a vary of calendar operations
#import datetime                                                               #For accesing datetime functions
#import math                                                                   #For accesing to a complex math operations
#import nltk                                                                   #For working with text data
#import random                                                                 #For generating random numbers
#import re                                                                     #For regular expressions
#import tabula                                                                 #For extracting tables from pdf
#import timeit                                                                 #For Measure execution time of small code snippets
#import time                                                                   #To measure the elapsed wall-clock time between two points
#import scykit-learn                                                           #For performing machine learning  
#import warnings
#import wikipedia

#from collections                     import defaultdict                       #Returns a new dictionary-like object
#from datetime                        import date                              #For obteining today function
#from datetime                        import datetime                          #For obteining today function
#from functools                       import reduce                            #For accessing to a high order functions (functions or operators that return functions)
#from glob                            import glob                              #For using with pathnames matching
#from itertools                       import combinations                      #For iterations
#from itertools                       import cycle                             #Used in the function plot_labeled_decision_regions()
#from math                            import ceil                              #Used in the function plot_labeled_decision_regions()
#from math                            import floor                             #Used in the function plot_labeled_decision_regions()
#from math                            import radian                            #For accessing a specific math operations
#from math                            import sqrt
#from matplotlib                      import colors                            #To create custom cmap
#from matplotlib.ticker               import StrMethodFormatter                #Import the necessary library to delete the scientist notation
#from mpl_toolkits.mplot3d            import Axes3D
#from numpy.random                    import randint                           #numpy.random.randint(low, high=None, size=None, dtype='l')-->Return random integers from low (inclusive) to high (exclusive).  
#from pandas.api.types                import CategoricalDtype                  #For categorical data
#from pandas.plotting                 import parallel_coordinates              #For Parallel Coordinates
#from pandas.plotting                 import register_matplotlib_converters    #For conversion as datetime index in x-axis
#from shapely.geometry                import LineString                        #(Geospatial) To create a Linestring geometry column 
#from shapely.geometry                import Point                             #(Geospatial) To create a point geometry column 
#from shapely.geometry                import Polygon                           #(Geospatial) To create a point geometry column 
#from string                          import Template                          #For working with string, regular expressions


#from bokeh.io                        import curdoc                            #For interacting visualizations
#from bokeh.io                        import output_file                       #For interacting visualizations
#from bokeh.io                        import show                              #For interacting visualizations
#from bokeh.plotting                  import ColumnDataSource                  #For interacting visualizations
#from bokeh.plotting                  import figure                            #For interacting visualizations
#from bokeh.layouts                   import column                            #For interacting visualizations
#from bokeh.layouts                   import gridplot                          #For interacting visualizations
#from bokeh.layouts                   import row                               #For interacting visualizations
#from bokeh.layouts                   import widgetbox                         #For interacting visualizations
#from bokeh.models                    import Button                            #For interacting visualizations
#from bokeh.models                    import CategoricalColorMapper            #For interacting visualizations
#from bokeh.models                    import CheckboxGroup                     #For interacting visualizations
#from bokeh.models                    import ColumnDataSource                  #For interacting visualizations
#from bokeh.models                    import HoverTool                         #For interacting visualizations
#from bokeh.models                    import RadioGroup                        #For interacting visualizations
#from bokeh.models                    import Select                            #For interacting visualizations
#from bokeh.models                    import Slider                            #For interacting visualizations
#from bokeh.models                    import Toggle                            #For interacting visualizations
#from bokeh.models.widgets            import Panel                             #For interacting visualizations
#from bokeh.models.widgets            import Tabs                              #For interacting visualizations
#from bokeh.palettes                  import Spectral6                         #For interacting visualizations


#import keras                                                                  #For DeapLearning
#import keras.backend as k                                                     #For DeapLearning
#from keras.applications.resnet50     import decode_predictions                #For DeapLearning
#from keras.applications.resnet50     import preprocess_input                  #For DeapLearning
#from keras.applications.resnet50     import ResNet50                          #For DeapLearning
#from keras.callbacks                 import EarlyStopping                     #For DeapLearning
#from keras.callbacks                 import ModelCheckpoint                   #For DeapLearning
#from keras.datasets                  import fashion_mnist                     #For DeapLearning
#from keras.datasets                  import mnist                             #For DeapLearning
#from keras.layers                    import BatchNormalization                #For DeapLearning
#from keras.layers                    import Concatenate                       #For DeapLearning
#from keras.layers                    import Conv2D                            #For DeapLearning
#from keras.layers                    import Dense                             #For DeapLearning
#from keras.layers                    import Dropout                           #For DeapLearning
#from keras.layers                    import Embedding                         #For DeapLearning
#from keras.layers                    import Flatten                           #For DeapLearning
#from keras.layers                    import GlobalMaxPooling1D                #For DeapLearning
#from keras.layers                    import Input                             #For DeapLearning
#from keras.layers                    import LSTM                              #For DeapLearning
#from keras.layers                    import MaxPool2D                         #For DeapLearning
#from keras.layers                    import SpatialDropout1D                  #For DeapLearning
#from keras.layers                    import Subtract                          #For DeapLearning
#from keras.models                    import load_model                        #For DeapLearning
#from keras.models                    import Model                             #For DeapLearning
#from keras.models                    import Sequential                        #For DeapLearning
#from keras.optimizers                import Adam                              #For DeapLearning
#from keras.optimizers                import SGD                               #For DeapLearning
#from keras.preprocessing             import image                             #For DeapLearning
#from keras.preprocessing.text        import Tokenizer                         #For DeapLearning
#from keras.preprocessing.sequence    import pad_sequences                     #For DeapLearning
#from keras.utils                     import plot_model                        #For DeapLearning
#from keras.utils                     import to_categorical                    #For DeapLearning
#from keras.wrappers.scikit_learn     import KerasClassifier                   #For DeapLearning


#import networkx          as nx                                                #For Network Analysis in Python
#import nxviz             as nv                                                #For Network Analysis in Python
#from nxviz                           import ArcPlot                           #For Network Analysis in Python
#from nxviz                           import CircosPlot                        #For Network Analysis in Python 
#from nxviz                           import MatrixPlot                        #For Network Analysis in Python 


#import scipy.stats as stats                                                   #For accesign to a vary of statistics functiosn
#from scipy.cluster.hierarchy         import dendrogram                        #For learning machine - unsurpervised
#from scipy.cluster.hierarchy         import fcluster                          #For learning machine - unsurpervised
#from scipy.cluster.hierarchy         import linkage                           #For learning machine - unsurpervised
#from scipy.ndimage                   import gaussian_filter                   #For working with images
#from scipy.ndimage                   import median_filter                     #For working with images
#from scipy.signal                    import convolve2d                        #For learning machine - deep learning
#from scipy.sparse                    import csr_matrix                        #For learning machine 
#from scipy.special                   import expit as sigmoid                  #For learning machine 
#from scipy.stats                     import anderson                          #For Anderson-Darling Normality Test. Tests whether a data sample has a Gaussian distribution.
#from scipy.stats                     import bernoulli                         #Generate bernoulli data
#from scipy.stats                     import chi2_contingency                  #For Chi-Squared Test. Tests whether two categorical variables are related or independent
#from scipy.stats                     import f_oneway                          #For Analysis of Variance Test. Tests whether the means of two or more independent samples are significantly different.
#from scipy.stats                     import friedmanchisquare                 #For Friedman Test. Tests whether the distributions of two or more paired samples are equal or not.
#from scipy.stats                     import kendalltau                        #For Kendall's Rank Correlation Test. To check if two samples are related.
#from scipy.stats                     import kruskal                           #For Kruskal-Wallis H Test. Tests whether the distributions of two or more independent samples are equal or not.
#from scipy.stats                     import mannwhitneyu                      #For Mann-Whitney U Test. Tests whether the distributions of two independent samples are equal or not.
#from scipy.stats                     import normaltest                        #For D'Agostino's K^2 Normality Test. Tests whether a data sample has a Gaussian distribution.
#from scipy.stats                     import pearsonr                          #For learning machine. For Pearson's Correlation test. To check if two samples are related.
#from scipy.stats                     import randint                           #For learning machine 
#from scipy.stats                     import shapiro                           #For Shapiro-Wilk Normality Test. Tests whether a data sample has a Gaussian distribution.
#from scipy.stats                     import spearmanr                         #For Spearman's Rank Correlation Test.  To check if two samples are related.
#from scipy.stats                     import ttest_ind                         #For Student's t-test. Tests whether the means of two independent samples are significantly different.
#from scipy.stats                     import ttest_rel                         #For Paired Student's t-test. Tests whether the means of two paired samples are significantly different.
#from scipy.stats                     import wilcoxon                          #For Wilcoxon Signed-Rank Test. Tests whether the distributions of two paired samples are equal or not.


#from skimage                         import exposure                          #For working with images
#from skimage                         import measure                           #For working with images
#from skimage.filters.thresholding    import threshold_otsu                    #For working with images
#from skimage.filters.thresholding    import threshold_local                   #For working with images 


#from sklearn                         import datasets                          #For learning machine
#from sklearn.cluster                 import KMeans                            #For learning machine - unsurpervised
#from sklearn.decomposition           import NMF                               #For learning machine - unsurpervised
#from sklearn.decomposition           import PCA                               #For learning machine - unsurpervised
#from sklearn.decomposition           import TruncatedSVD                      #For learning machine - unsurpervised
#from sklearn.ensemble                import AdaBoostClassifier                #For learning machine - surpervised
#from sklearn.ensemble                import BaggingClassifier                 #For learning machine - surpervised
#from sklearn.ensemble                import GradientBoostingRegressor         #For learning machine - surpervised
#from sklearn.ensemble                import RandomForestClassifier            #For learning machine
#from sklearn.ensemble                import RandomForestRegressor             #For learning machine - unsurpervised
#from sklearn.ensemble                import VotingClassifier                  #For learning machine - unsurpervised
#from sklearn.feature_selection       import chi2                              #For learning machine
#from sklearn.feature_selection       import SelectKBest                       #For learning machine
#from sklearn.feature_extraction.text import CountVectorizer                   #For learning machine
#from sklearn.feature_extraction.text import HashingVectorizer                 #For learning machine
#from sklearn.feature_extraction.text import TfidfVectorizer                   #For learning machine - unsurpervised
#from sklearn.impute                  import SimpleImputer                     #For learning machine
#from sklearn.linear_model            import ElasticNet                        #For learning machine
#from sklearn.linear_model            import Lasso                             #For learning machine
#from sklearn.linear_model            import LinearRegression                  #For learning machine
#from sklearn.linear_model            import LogisticRegression                #For learning machine
#from sklearn.linear_model            import Ridge                             #For learning machine
#from sklearn.manifold                import TSNE                              #For learning machine - unsurpervised
#from sklearn.metrics                 import accuracy_score                    #For learning machine
#from sklearn.metrics                 import classification_report             #For learning machine
#from sklearn.metrics                 import confusion_matrix                  #For learning machine
#from sklearn.metrics                 import mean_squared_error as MSE         #For learning machine
#from sklearn.metrics                 import roc_auc_score                     #For learning machine
#from sklearn.metrics                 import roc_curve                         #For learning machine
#from sklearn.model_selection         import cross_val_score                   #For learning machine
#from sklearn.model_selection         import GridSearchCV                      #For learning machine
#from sklearn.model_selection         import KFold                             #For learning machine
#from sklearn.model_selection         import RandomizedSearchCV                #For learning machine
#from sklearn.model_selection         import train_test_split                  #For learning machine
#from sklearn.multiclass              import OneVsRestClassifier               #For learning machine
#from sklearn.neighbors               import KNeighborsClassifier as KNN       #For learning machine
#from sklearn.pipeline                import FeatureUnion                      #For learning machine
#from sklearn.pipeline                import make_pipeline                     #For learning machine - unsurpervised
#from sklearn.pipeline                import Pipeline                          #For learning machine
#from sklearn.preprocessing           import FunctionTransformer               #For learning machine
#from sklearn.preprocessing           import Imputer                           #For learning machine
#from sklearn.preprocessing           import MaxAbsScaler                      #For learning machine (transforms the data so that all users have the same influence on the model)
#from sklearn.preprocessing           import Normalizer                        #For learning machine - unsurpervised (for pipeline)
#from sklearn.preprocessing           import normalize                         #For learning machine - unsurpervised
#from sklearn.preprocessing           import scale                             #For learning machine
#from sklearn.preprocessing           import StandardScaler                    #For learning machine
#from sklearn.svm                     import SVC                               #For learning machine
#from sklearn.tree                    import DecisionTreeClassifier            #For learning machine - supervised
#from sklearn.tree                    import DecisionTreeRegressor             #For learning machine - supervised


#import statsmodels                   as sm                                    #For stimations in differents statistical models
#import statsmodels.api               as sm                                    #Make a prediction model
#import statsmodels.formula.api       as smf                                   #Make a prediction model    
#from statsmodels.graphics.tsaplots   import plot_acf                          #For autocorrelation function
#from statsmodels.graphics.tsaplots   import plot_pacf                         #For simulating data and for plotting the PACF. Partial Autocorrelation Function measures the incremental benefit of adding another lag.
#from statsmodels.tsa.arima_model     import ARIMA                             #Similar to use ARMA but on original data (before differencing)
#from statsmodels.tsa.arima_model     import ARMA                              #To estimate parameters from data simulated (AR model)
#from statsmodels.tsa.arima_process   import ArmaProcess                       #For simulating data and for plotting the PACF, For Simulate Autoregressive (AR) Time Series 
#from statsmodels.tsa.stattools       import acf                               #For autocorrelation function
#from statsmodels.tsa.stattools       import adfuller                          #For Augmented Dickey-Fuller unit root test. Test for Random Walk. Tests whether a time series has a unit root, e.g. has a trend or more generally is autoregressive. Tests that you can use to check if a time series is stationary or not.
#from statsmodels.tsa.stattools       import coint                             #Test for cointegration
#from statsmodels.tsa.stattools       import kpss                              #For Kwiatkowski-Phillips-Schmidt-Shin test. Tests whether a time series is trend stationary or not.


#import tensorflow              as tf                                          #For DeapLearning



# Setting the pandas options
#pd.set_option("display.max_columns",20)
#pd.options.display.float_format = '{:,.4f}'.format 
#pd.reset_option("all")
#pd.set_option('display.max_rows', -1)                                         #Shows all rows

#register_matplotlib_converters()                                              #Require to explicitly register matplotlib converters.

#Setting images params
#plt.rcParams = plt.rcParamsDefault
#plt.rcParams['figure.constrained_layout.use'] = True
#plt.rcParams['figure.constrained_layout.h_pad'] = 0.09
#plt.rcParams.update({'figure.max_open_warning': 0})                           #To solve the max images open
#plt.rcParams["axes.labelsize"] = 8                                            #Font
#plt.rc('xtick',labelsize=8)
#plt.rc('ytick',labelsize=6)
#plt.rcParams['figure.max_open_warning'] = 60                                  #params = {'legend.fontsize': 'x-large', 'figure.figsize': (15, 5), 'axes.labelsize': 'x-large', 'axes.titlesize':'x-large', 'xtick.labelsize':'x-large', 'ytick.labelsize':'x-large'}
#plt.rcParams["legend.fontsize"] = 8
#plt.style.use('dark_background')
#plt.style.use('default')
#plt.xticks(fontsize=7); plt.yticks(fontsize=8);
#plt.xticks(rotation=45)
#ax.xaxis.set_major_locator(plt.MaxNLocator(3))                                #Define the number of ticks to show on x axis.
#ax.xaxis.set_major_locator(mdates.AutoDateLocator())                          #Other way to Define the number of ticks to show on x axis.
#ax.xaxis.set_major_formatter(mdates.AutoDateFormatter())                      #Other way to Define the number of ticks to show on x axis.


#plt.xticks(rotation=45)                                                       #rotate x-axis labels by 45 degrees
#plt.yticks(rotation=90)                                                       #rotate y-axis labels by 90 degrees
#plt.savefig("sample.jpg")                                                     #save image of `plt`

#To supress the scientist notation in plt
#from matplotlib.ticker import StrMethodFormatter                              #Import the necessary library to delete the scientist notation
#ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))                  #Delete the scientist notation
#ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))                  #Delete the scientist notation

#from matplotlib.axes._axes import _log as matplotlib_axes_logger              #To avoid warnings
#matplotlib_axes_logger.setLevel('ERROR')
#matplotlib_axes_logger.setLevel(0)                                            #To restore default

#ax.tick_params(labelsize=6)                                                   #axis : {'x', 'y', 'both'}
#ax.tick_params(axis='x', rotation=45)                                         #Set rotation atributte

#Setting the numpy options
#np.set_printoptions(precision=3)                                              #precision set the precision of the output:
#np.set_printoptions(suppress=True)                                            #suppress suppresses the use of scientific notation for small numbers
#np.set_printoptions(threshold=np.inf)                                         #Show all the columns and rows from an array.
#np.set_printoptions(threshold=8)                                              #Return to default value.
#np.random.seed(SEED)

#tf.compat.v1.set_random_seed(SEED)                                            #Instead of tf.set_random_seed, because it is deprecated.

#sns.set(font_scale=0.8)                                                       #Font
#sns.set(rc={'figure.figsize':(11.7,8.27)})                                    #To set the size of the plot
#sns.set(color_codes=True)                                                     #Habilita el uso de los codigos de color
#sns.set()                                                                     #Seaborn defult style
#sns.set_style(this_style)                                                     #['white', 'dark', 'whitegrid', 'darkgrid', 'ticks']:
#sns.despine(left=True)                                                        #Remove the spines (all borders)
#sns.palettes.SEABORN_PALETTES                                                 #Despliega todas las paletas disponibles 
#sns.palplot(sns.color_palette())                                              #Display a palette
#sns.color_palette()                                                           #The current palette
#sns.set(style=”whitegrid”, palette=”pastel”, color_codes=True)
#sns.mpl.rc(“figure”, figsize=(10,6))

#warnings.filterwarnings('ignore', 'Objective did not converge*')              #To avoid the warning, review https://github.com/scikit-learn/scikit-learn/issues/13394
#warnings.filterwarnings('default', 'Objective did not converge*')             #To avoid the warning, review https://github.com/scikit-learn/scikit-learn/issues/13394


#Create categorical type data to use
#cats = CategoricalDtype(categories=['good', 'bad', 'worse'],  ordered=True)
# Change the data type of 'rating' to category
#weather['rating'] = weather.rating.astype(cats)


#print("The area of your rectangle is {}cm\u00b2".format(area))                 #Print the superscript 2

### Show a basic html page
#tmp=tempfile.NamedTemporaryFile()
#path=tmp.name+'.html'
#f=open(path, 'w')
#f.write("<html><body><h1>Test</h1></body></html>")
#f.close()
#webbrowser.open('file://' + path)

