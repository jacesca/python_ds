# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 4: Advanced Applications of Simulation
    In this chapter, students will be introduced to some basic and advanced 
    applications of simulation to solve real-world problems. We'll work through 
    a business planning problem, learn about Monte Carlo Integration, Power 
    Analysis with simulation and conclude with a financial portfolio simulation. 
    After completing this chapter, students will be ready to apply simulation 
    to solve everyday problems.
Source: https://learn.datacamp.com/courses/statistical-simulation-in-python
"""
###############################################################################
## Importing libraries
###############################################################################
import numpy as np

###############################################################################
## Preparing the environment
###############################################################################
# Global variables
SEED = 123
SIZE = 10000

# Global configuration
np.set_printoptions(formatter={'float': '{:,.4f}'.format})
from scipy.stats import ttest_ind

###############################################################################
## Reading the data
###############################################################################



###############################################################################
## Main part of the code
###############################################################################
def Simulation_for_Business_Planning(size=5000, seed=223):
    print("****************************************************")
    topic = "1. Simulation for Business Planning"; print("** %s" % topic)
    print("****************************************************")
    
    print('Suppose that you manage a small corn farm and are interested in ' + \
          'optimizing your costs. In this exercise, we will model the production of corn.')
    print("Let's assume that corn production depends on only two factors: " + \
          'rain, which you do not control, and cost, which you control. ')
    print('Rain is normally distributed with mean 50 and standard deviation 15. ' + \
          "For now, let's fix cost at 5,000. Corn produced in any season is a " + \
          'Poisson random variable while the average corn production is ' + \
          'governed by the equation: «100×(cost)⁰˙¹×(rain)⁰˙²». ')
    print('Simulate one outcome.\n')
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('-------------------------------------Forcasting rain')
    #Forecasting rain
    rain = np.random.normal(loc=50, scale=15, size=size)
    neg_rain = len(rain[rain<0])
    if neg_rain>0:
        print(f"{len(rain[rain<0])} negative elements found, transformed to positive")
        rain = np.abs(rain)
    
    print('------------------------Continueing with the process')
    #Forecasting cost
    cost = 5000
        
    def corn_production(cost, rain, size=1):
        #Forecasting corn production
        return np.random.poisson(lam=100*(cost**.1)*(rain**.2))
    
    production = corn_production(cost, rain, size) 
    print(f'Mean Production estimate in the simulation: {production.mean():,.4f}.\n\n')
    
    
    topic = "2. Modeling Corn Production"; print("** %s" % topic)
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('Simulate one outcome...')
    production = corn_production(cost, rain, 1) 
    print(f'Mean Production estimate in the simulation: {production.mean():,.4f}.\n\n')
    
    
    
def Modeling_Profits(size=SIZE, seed=223):
    print("****************************************************")
    topic = "3. Modeling Profits"; print("** %s" % topic)
    print("****************************************************")
    
    print('Suppose that price is normally distributed with mean 40 and ' + \
          'standard deviation 10.\n')
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Function to calculate profits
    def corn_produced(cost, rain):
        #Forecasting corn production
        return np.random.poisson(lam=100*(cost**.1)*(rain**.2))
    
    def profits(cost):
        rain = np.random.normal(50, 15)
        if rain<0: rain = np.abs(rain)
        price = np.random.normal(40, 10)
        supply = corn_produced(rain, cost)
        demand = -7.9993*price + 999.93
        profitable = np.where(supply <= demand, (supply * price), (demand * price)) - cost
        return profitable
    
    #Forecasting cost
    cost = 5000
    
    result = profits(cost)
    print("Simulated profit = {:,.4f}.\n\n".format(result))
    
    
    
def Optimizing_Costs(size=1000, seed=573):
    print("****************************************************")
    topic = "4. Optimizing Costs"; print("** %s" % topic)
    print("****************************************************")
    
    print('Now we will use the functions you have built to optimize our ' + \
          'cost of production. We are interested in maximizing average profits.')
    print('However, our profits depend on a number of factors, but we only ' + \
          'control cost. Thus, we can simulate the uncertainty in the other ' + \
          'factors and vary cost to see how our profits are impacted.')
    print('Since you manage the small corn farm, you have the ability to ' + \
          'choose your cost - from $100 to $5,000. You want to choose the ' + \
          'cost that gives you the maximum average profit.\n\n')
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Function to calculate profits
    def corn_produced(cost, rain):
        #Forecasting corn production
        return np.random.poisson(lam=100*(cost**.1)*(rain**.2))
    
    def profits(cost):
        rain = np.random.normal(50, 15)
        if rain<0: rain = np.abs(rain)
        price = np.random.normal(40, 10)
        supply = corn_produced(rain, cost)
        demand = -7.9993*price + 999.93
        profitable = np.where(supply <= demand, (supply * price), (demand * price)) - cost
        return profitable
    
    # Initialize results and cost_levels variables
    cost_levels, results = np.arange(100, 5100, 100), {}

    # For each cost level, simulate profits and store mean profit
    for cost in cost_levels:
        profits_result = np.zeros(size)
        for i in range(size):
            profits_result[i] = profits(cost)
        results[cost] = profits_result.mean()
    
    # Get the cost that maximizes average profit
    maximun_profit = max(results, key=results.get)
    print("Average profit is maximized when cost = ${:,.2f}, forcasting a revenue of ${:,.2f} .\n\n".format(maximun_profit, results[maximun_profit]))
    
    
    
def Monte_Carlo_Integration(size=1000000, seed=SEED):
    print("****************************************************")
    topic = "5. Monte Carlo Integration"; print("** %s" % topic)
    print("****************************************************")
    
    print('We are interested in find ∫₁² x²...')
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Define the sim_integrate function
    def sim_integrate(func, xmin, xmax, size=size):
        x = np.random.uniform(xmin, xmax, size)
        y = np.random.uniform(min(min(func(x)),0), max(func(x)), size)
        area = (max(y) - min(y))*(xmax-xmin)
        result = area * np.mean(abs(y) < abs(func(x)))
        return result
    
    # Call the sim_integrate function and print results
    result = sim_integrate(func = lambda x: x**2, xmin = 1, xmax = 2)
    print(f"...Simulated answer = {result:.6f}, Actual Answer = 7/3≈{7/3:.6f}, (Size of simulation: {size:,.0f}).\n\n")
    
    
    topic = "6. Integrating a Simple Function"; print("** %s" % topic)
    print('We are interested in find ∫₀¹ xeˣ...')
    # Initialize seed and parameters
    np.random.seed(seed) 
    # Call the sim_integrate function and print results
    result = sim_integrate(func = lambda x: x*(np.e**x), xmin = 0, xmax = 1)
    print(f"...Simulated answer = {result:.6f}, Actual Answer = 1, (Size of simulation: {size:,.0f}).\n\n")
    
    
    topic = "7. Calculating the value of pi"; print("** %s" % topic)
    print('Imagine a square of side 2 with the origin (0,0) as its center and ' + \
          'the four corners having coordinates (1,1), (1,−1), (−1,1), (−1,−1). ' + \
          'The area of this square is 2×2=4. ')
    print('Now imagine a circle of radius 1 with its center at the origin ' + \
          'fitting perfectly inside this square. The area of the circle will be ' + \
          'π×radius2=π.\n')
    
    print('----------------------------------Using the function')
    print('------------------------------y = raiz(r - x^2), r=1')
    # Initialize seed and parameters
    np.random.seed(seed) 
    # Call the sim_integrate function and print results
    result = sim_integrate(func = lambda x: np.sqrt(1-(x**2))*2, xmin = -1, xmax = 1) #Because np.sqrt gives one side only of the square.
    print(f"...Simulated answer = {result:.6f}, Real π = {np.pi:.6f}, (Size of simulation: {size:,.0f}).\n")
    
    print('-----------------------------According to the lesson')
    print('--------------------------------x^2 + y^2 = r^2, r=1')
    # Initialize seed and parameters
    np.random.seed(seed) 
    # Initialize sims and circle_points
    circle_points = np.zeros(size) 
    for i in range(size):
        # Generate the two coordinates of a point
        point = np.random.uniform(low=-1, high=1, size=2)
        # if the point lies within the unit circle, increment counter
        circle_points[i] = point[0]**2 + point[1]**2 <= 1
    
    # Estimate pi as 4 times the avg number of points in the circle.
    pi_sim = 4*circle_points.mean()
    print(f"...Simulated answer = {pi_sim:.6f}, Real π = {np.pi:.6f}, (Size of simulation: {size:,.0f}).\n\n")
    
    print('---------------------------------Improving the model')
    print('--------------------------------x^2 + y^2 = r^2, r=1')
    # Initialize seed and parameters
    np.random.seed(seed) 
    # Initialize sims and circle_points
    circle_points = np.zeros(size) 
    
    # Generate the two coordinates of a point
    point = np.random.uniform(low=-1, high=1, size=(2, size))
    # if the point lies within the unit circle, increment counter
    circle_points = point[0, :]**2 + point[1, :]**2 <= 1
    
    # Estimate pi as 4 times the avg number of points in the circle.
    pi_sim = 4*circle_points.mean()
    print(f"...Simulated answer = {pi_sim:.6f}, Real π = {np.pi:.6f}, (Size of simulation: {size:,.0f}).\n\n")
    
    
def Power_Analysis_Part_I_y_II(size=1000, seed=SEED):
    print("****************************************************")
    topic = "10. Power Analysis - Part I"; print("** %s" % topic)
    print("****************************************************")
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('Suppose that you are in charge of a news media website and you ' + \
          'are interested in increasing the amount of time users spend on ' + \
          'your website. ')
    print('Currently, the time users spend on your website is normally ' + \
          'distributed with a mean of 1 minute and a variance of 0.5 minutes. ')
    print('Suppose that you are introducing a feature that loads pages faster ' + \
          'and want to know the sample size required to measure a 10% increase ' + \
          'in time spent on the website.')
    print('In this exercise, we will set up the framework to run one simulation, ' + \
          'run a t-test, & calculate the p-value.\n\n')
    
    # Initialize effect_size, control_mean, control_sd
    effect_size, sample_size, control_mean, control_sd = .1, 50, 1, .5
    
    # Simulate control_time_spent and treatment_time_spent, assuming equal variance
    control_time_spent = np.random.normal(loc=control_mean, scale=control_sd, size=sample_size)
    treatment_time_spent = np.random.normal(loc=control_mean*(1+effect_size), scale=control_sd, size=sample_size)
    
    # Run the t-test and get the p_value
    t_stat, p_value = ttest_ind(control_time_spent, treatment_time_spent)
    stat_sig = p_value < .05
    print("P-value: {}, Statistically Significant? {}.\n\n".format(p_value, stat_sig))
    
    
    topic = "11. Power Analysis - Part II"; print("** %s" % topic)
    
    print("Power of an experiment is the experiment's ability to detect a difference " + \
          'between treatment & control if the difference really exists. It is good ' + \
          'statistical hygiene to strive for 80% power.')
    print('For our website, we want to know how many people need to visit each variant, ' + \
          'such that we can detect a 10% increase in time spent with 80% power. \n\n')
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Initialize effect_size, control_mean, control_sd
    effect_size, sample_size, control_mean, control_sd = .1, 50, 1, .5
    
    # Keep incrementing sample size by 10 till we reach required power
    while 1:
        control_time_spent = np.random.normal(loc=control_mean, scale=control_sd, 
                                              size=(sample_size, size))
        treatment_time_spent = np.random.normal(loc=control_mean*(1+effect_size), scale=control_sd, 
                                                size=(sample_size, size))
        t, p = ttest_ind(treatment_time_spent, control_time_spent) #p has a shape=size
        
        # Power is the fraction of times in the simulation when the p-value was less than 0.05
        power = (p < 0.05).mean()
        if power >= .8: break;
        else: sample_size += 10
    print("For 80% power, sample size required = {}.\n\n".format(sample_size))
    
    
    
def Applications_in_Finance(size=1000, seed=SEED):
    print("****************************************************")
    topic = "12. Applications in Finance"; print("** %s" % topic)
    print("****************************************************")
    
    print('In the next few exercises, you will calculate the expected returns ' + \
          'of a stock portfolio & characterize its uncertainty.')
    print('Suppose you have invested $10,000 in your portfolio comprising of ' + \
          "multiple stocks. You want to evaluate the portfolio's performance " + \
          'over 10 years. You can tweak your overall expected rate of return and ' + \
          'volatility (standard deviation of the rate of return). ')
    print('Assume the rate of return follows a normal distribution.\n\n')
    
    
    topic = "13. Portfolio Simulation - Part I"; print("** %s" % topic)
    print('First, write a function that takes the principal (initial investment), ' + \
          'number of years, expected rate of return and volatility as inputs ' + \
          "and returns the portfolio's total value after 10 years.")
    print('After, use this function to calculate the return for 5 years wiht ' + \
          'an average rate of 0.07, volatility of 0.15, and 1000 as initial ' + \
          'investment.\n\n')
    
    # rates is a Normal random variable and has size equal to number of years
    def portfolio_return(yrs, avg_return, sd_of_return, principal, seed=seed):
        rates = np.random.normal(loc=avg_return, scale=sd_of_return, size=yrs)
        # Calculate the return at the end of the period
        end_return = principal
        for x in rates:
            end_return = end_return*(1+x)
        return end_return
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    result = portfolio_return(yrs = 5, avg_return = 0.07, sd_of_return = 0.15, principal = 1000)
    print(f"Portfolio return after 5 years = $ {result:,.2f}.\n\n")
    
    
    
    topic = "14. Portfolio Simulation - Part II"; print("** %s" % topic)
    print('Use the simulation function you built to evaluate 10-year returns. ')    
    print('Your stock-heavy portfolio has an initial investment of $10,000, ' + \
          'an expected return of 7% and a volatility of 30%. ')
    print('You want to get a 95% confidence interval of what your investment will' + \
          ' be worth in 10 years. \n\n')
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Run 1,000 iterations and store the results
    rets = np.zeros(size)
    for i in range(size):
        rets[i] = portfolio_return(yrs = 10, avg_return = 0.07, sd_of_return = 0.3, principal = 10000)
    
    # Calculate the 95% CI
    ci = np.percentile(rets, [2.5, 97.5])
    print(f"95% CI of Returns: Lower = {ci[0]:,.2f}, Upper = {ci[1]:,.2f}")
    
    
    topic = "15. Portfolio Simulation - Part III"; print("** %s" % topic)
    print('Use simulation for decision making.')
    print('You have the choice of rebalancing your portfolio with some bonds such ' + \
          'that the expected return is 4% & volatility is 10%. ')
    print('You have a principal of $10,000. You want to select a strategy based on ' + \
          'how much your portfolio will be worth in 10 years. ')
    print("Let's simulate returns for both the portfolios and choose based on the " + \
          'least amount you can expect with 75% probability (25th percentile).\n\n')
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Run 1,000 iterations and store the results
    rets_stock, rets_bond = np.zeros(size), np.zeros(size)
    for i in range(size):
        rets_stock[i] = portfolio_return(yrs = 10, avg_return = .07, sd_of_return = .3, principal = 10000)
        rets_bond[i] = portfolio_return(yrs = 10, avg_return = .04, sd_of_return = .1, principal = 10000)
    
    # Calculate the 25th percentile of the distributions and the amount you'd lose or gain
    rets_stock_perc = np.percentile(rets_stock, 25)
    rets_bond_perc = np.percentile(rets_bond, 25)
    additional_returns = rets_stock_perc - rets_bond_perc
    
    print(f"Stock heavy portfolio = {rets_stock_perc:,.2f}")
    print(f"Bond heavy portfolio  = {rets_bond_perc:,.2f}")
    print(f"Sticking to stocks gets you an additional return of {additional_returns:,.2f}.\n\n")
    
    
    
def main():
    print("\n\n****************************************************")
    print("** BEGIN                                          **")
    
    Simulation_for_Business_Planning()
    Modeling_Profits()
    Optimizing_Costs()
    
    Monte_Carlo_Integration()
    
    Power_Analysis_Part_I_y_II()
    
    Applications_in_Finance()
    
    print("****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    # Return to default
    np.set_printoptions(formatter={'float': None})