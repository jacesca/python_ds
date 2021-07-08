# -*- coding: utf-8 -*-
"""
Created on Thu May 28 09:09:06 2020
@author: jacesca@gmail.com

Utilities created in DataCamp's statistical thinking courses.
For academic purpouse. The scipy.stats module offers
more efficient calculation.

Source of inspiration: dc_stat_think
    pip install dc_stat_think
    ##C:\Anaconda3\Lib\site-packages\dc_stat_think
"""
import numpy as np

def ecdf(data, formal=False, buff=0.1, min_x=None, max_x=None):
    """Compute ECDF for a one-dimensional array of measurements."""
    n = len(data) # Number of data points: n
    x = np.sort(data) # x-data for the ECDF: x
    y = np.arange(1, n+1) / n # y-data for the ECDF: y
    
    if formal:
        # Set defaults for min and max tails
        if min_x is None: min_x = x[0] - (x[-1] - x[0])*buff
        if max_x is None: max_x = x[-1] + (x[-1] - x[0])*buff
        
        # Set up output arrays
        x_formal = np.empty(2*(len(x) + 1))
        y_formal = np.empty(2*(len(x) + 1))
        
        # y-values for steps
        y_formal[:2] = 0
        y_formal[2::2] = y
        y_formal[3::2] = y
        
        # x- values for steps
        x_formal[0] = min_x
        x_formal[1] = x[0]
        x_formal[2::2] = x
        x_formal[3:-1:2] = x[1:]
        x_formal[-1] = max_x

        x, y = x_formal, y_formal
    
    return x, y

def ecdf_formal(x, data):
    """Compute the values of the formal ECDF generated from `data` at x."""
    return np.searchsorted(np.sort(data), x, side='right') / len(data)

def ks_stat(points, model): 
    """
    Compute the 2-sample Kolmogorov-Smirnov statistic with the  assumption that the ECDF of 
    `model` is an approximation for the CDF of a continuous distribution function.

    Parameters
    ----------
    points: ndarray. One-dimensional array of data.
    model : ndarray. One-dimensional array of data generated to approximate the CDF of a theoretical 
            distribution.
    Returns
    -------
    output: float. Approximate Kolmogorov-Smirnov statistic.
    Notes
    -----
    .. Compares the distances between the concave corners of `points` and the value of the ECDF of
       `model` and also the distances between the convex corners of `points` and the value of the
       ECDF of `model`. This approach is taken because we are  approximating the CDF of a continuous 
       distribution function with the ECDF of `model`.
    .. This is not strictly speaking a 2-sample K-S statistic because of the assumption that the ECDF 
       of `model` is approximating the CDF of a continuous distribution. This can be seen from a 
       pathological example. Imagine we have two data sets,
           data_1 = np.array([0, 0])
           data_2 = np.array([0, 0])
       The distance between the ECDFs of these two data sets should be  zero everywhere. This function 
       will return 1.0, since that is the distance from the "top" of the step in the ECDF of `data_2`
       and the "bottom" of the step in the ECDF of `data_1.
    .. Because this is not a 2-sample K-S statistic, it should not be used as such. The intended use 
       it to take a hacker stats approach to comparing a set of measurements to a theoretical 
       distirbution.
       scipy.stats.kstest() computes the K-S statistic exactly (and also does the K-S hypothesis test 
       exactly in a much more efficient calculation). If you do want to compute a two-sample
       K-S statistic, use scipy.stats.ks_2samp().
    """
    # Sort data in model param
    model = np.sort(model)
    # Compute ECDF from given points: x, y
    x, y = ecdf(points)
    # Compute corresponding values of the target CDF
    cdf = ecdf_formal(x, model) #trayecto

    # Compute distances between concave corners and CDF
    D_top = y - cdf
    # Compute distance between convex corners and CDF
    D_bottom = cdf - y + 1/len(points)
    return np.max(np.concatenate((D_top, D_bottom)))

def draw_ks_reps(n, f, args=(), size=10000, n_reps=10000):
    """
    Draw Kolmogorov-Smirnov replicates.
    Parameters
    ----------
    n     : int. Size of experimental sample.
    func  : function. Function with call signature `func(*args, size=1)` that generates random number drawn from theoretical distribution.
    size  : int, default 10000. Number of random numbers to draw from theoretical distribution to approximate its analytical distribution.
    n_reps: int, default 1. Number of pairs Kolmogorov-Smirnov replicates to draw.
    args  : tuple, default (). Arguments to be passed to `func`.
    Returns
    -------
    output: ndarray. Array of Kolmogorov-Smirnov replicates
    Notes
    -----
    .. The theoretical distribution must be continuous for the K-S statistic to make sense.
    .. This function approximates the theoretical distribution by drawing many samples out of it, in the 
       spirit of hacker stats. scipy.stats.kstest() computes the K-S statistic exactly, and also does the 
       K-S hypothesis test exactly in a much more efficient calculation.
    """
    # Generate samples from target distribution
    x_f = np.sort(f(*args, size=size))
    # Initialize K-S replicates
    reps = np.empty(n_reps)
    # Draw replicates
    for i in range(n_reps):
        # Draw samples for comparison
        x_samp = f(*args, size=n)
        # Compute K-S statistic
        reps[i] = ks_stat(x_samp, x_f)
    return reps
    
def bootstrap_replicate_1d(data, func, replace=True):
    """Generate bootstrap replicate of 1D data."""
    return func(np.random.choice(data, size=len(data), replace=replace))

def draw_bs_reps(data, func, size=1, replace=True):
    """Draw bootstrap replicates for a specific function applied to a 1D data."""
    bs_replicates = np.zeros(size) # Initialize array of replicates: bs_replicates
    for i in range(size): # Generate replicates
        bs_replicates[i] = bootstrap_replicate_1d(data, func, replace)
    return bs_replicates

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

def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""
    diff = data_1.mean()-data_2.mean() # The difference of means of data_1, data_2: diff
    return diff

def gaussian_model(x, mu, sigma):
    """Define gaussian model function"""
    coeff_part = 1/(np.sqrt(2 * np.pi * sigma**2))
    exp_part = np.exp( - (x - mu)**2 / (2 * sigma**2) )
    return coeff_part*exp_part

def draw_bs_pairs_linreg(x, y, size=1, replace=True):
    """
    Perform pairs bootstrap for linear regression.
    Parameters
    ----------
    x             : array_like. x-values of data.
    y             : array_like. y-values of data.
    size          : int, default 1. Number of pairs bootstrap replicates to draw.
    Returns
    -------
    slope_reps    : ndarray. Pairs bootstrap replicates of the slope.
    intercept_reps: ndarray. Pairs bootstrap replicates of the intercept.
    Notes
    -----
    .. Entries where either `x` or `y` has a nan are ignored.
    .. It is possible that a pairs bootstrap sample has the same pair over and over again. In this case, 
       a linear regression cannot be computed. The pairs bootstrap replicate in this instance is NaN.
    """
    inds = np.arange(len(x)) # Set up array of indices to sample from: inds
    bs_slope_reps = np.zeros(size) # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_intercept_reps = np.zeros(size)
    for i in range(size): # Generate replicates
        bs_inds = np.random.choice(inds, size=len(inds), replace=replace)
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1)
    return bs_slope_reps, bs_intercept_reps 

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    corr_mat = np.corrcoef(x, y) # Compute correlation matrix: corr_mat
    return corr_mat[0,1]         # Return entry [0,1]

def b_value(mags, mt, perc=[2.5,97.5], n_reps=None):
    """Compute the b-value and optionally its confidence interval."""
    # Extract magnitudes above completeness threshold: m
    m = mags[mags >= mt]
    # Compute b-value: b
    b = (m.mean() - mt)*np.log(10)
    # Draw bootstrap replicates
    if n_reps is None:
        return b
    else:
        m_bs_reps = draw_bs_reps(m, np.mean, n_reps)
        # Compute b-value from replicates: b_bs_reps
        b_bs_reps = (m_bs_reps - mt) * np.log(10)
        # Compute confidence interval: conf_int
        conf_int = np.percentile(b_bs_reps, perc)
        return b, conf_int
    
def plotting_test_Ho(ax, bs_sample, effect_size, msg, x_label='', title='', 
                     params_title={}, params_text={'fontsize':8}, 
                     params_legend={'loc':'upper right', 'framealpha':.9, 'fontsize':8},
                     params_label={'fontsize':8}, params_ticks={'labelsize':8},
                     greater=True, pos_x_msg=0.01, pos_y_msg=0.97):
    """Plot the histogram of the replicates and its confidence interval."""
    # Compute the 95% confidence interval: conf_int
    conf_int = np.percentile(bs_sample, [2.5, 97.5])
    
    # add a 'best fit' line
    mu = bs_sample.mean()
    sigma = bs_sample.std()
    
    msg = msg + "\n95% confidence interval: [{:,.5f}, {:,.5f}].".format(conf_int[0], conf_int[1])
    
    # Plot the histogram of the replicates
    _, bins, _ = ax.hist(bs_sample, bins=50, rwidth=.9, density=True, color='red', label='replicates sample')
    y = gaussian_model(bins, mu, sigma)
    ax.plot(bins, y, '--', color='black')
    ax.axvline(bs_sample.mean(), color='darkred', linestyle='solid', linewidth=2, label='replicates mean')
    ax.axvspan(conf_int[0], conf_int[1], color='red', alpha=0.1, label='confidence interval')
    
    ax.axvline(effect_size, color='darkblue', lw=2, label='effect size')
    min_x, max_x = ax.get_xlim()
    if greater: ax.axvspan(effect_size, max_x, color='gray', alpha=0.5, label='p-value')
    else: ax.axvspan(min_x, effect_size, color='gray', alpha=0.5, label='p-value')
    
    ax.tick_params(**params_ticks)
    ax.set_xlim(min_x, max_x)
    ax.set_xlabel(x_label, **params_label)
    ax.set_ylabel('PDF', **params_label)
    t = ax.text(pos_x_msg, pos_y_msg, msg, transform=ax.transAxes, color='black', ha='left', va='top', **params_text)
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='darkgray'))
    ax.set_title(title, **params_title)
    ax.legend(**params_legend)
    ax.grid(True)    

def plotting_CI(ax, bs_sample, effect_mean, msg, x_label='', title='', 
                params_title={}, params_text={'fontsize':8}, 
                params_legend={'loc':'upper right', 'framealpha':.9, 'fontsize':8},
                params_label={'fontsize':8}, params_ticks={'labelsize':8},
                pos_x_msg=0.01, pos_y_msg=0.97, msg_ci=True):
    """Plot the histogram of the replicates and its confidence interval."""
    # Compute the 95% confidence interval: conf_int
    conf_int = np.percentile(bs_sample, [2.5, 97.5])
    
    # add a 'best fit' line
    mu = bs_sample.mean()
    sigma = bs_sample.std()
    
    if msg_ci: msg = msg + "\n95% confidence interval: [{:,.5f}, {:,.5f}].".format(conf_int[0], conf_int[1])
    
    # Plot the histogram of the replicates
    _, bins, _ = ax.hist(bs_sample, bins=50, rwidth=.9, density=True, color='lightgreen', label='replicates sample')
    y = gaussian_model(bins, mu, sigma)
    ax.plot(bins, y, '--', color='black')
    ax.axvline(effect_mean, color='darkgreen', linestyle='solid', linewidth=2, label='Mean')
    ax.axvspan(conf_int[0], conf_int[1], color='green', alpha=0.2, label='confidence interval')
    
    ax.tick_params(**params_ticks)
    ax.set_xlabel(x_label, **params_label)
    ax.set_ylabel('Frequency', **params_label)
    t = ax.text(pos_x_msg, pos_y_msg, msg, transform=ax.transAxes, color='black', ha='left', va='top', **params_text)
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='darkgray'))
    ax.set_title(title, **params_title)
    ax.legend(**params_legend)
    ax.grid(True)    