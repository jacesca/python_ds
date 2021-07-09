# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:36:32 2020

@author: jacqueline.cortez
Subject: Practicing Statistics Interview Questions in Python
Chapter 1: Basics of randomness & simulation
    This chapter gives you the tools required to run a simulation. We'll start 
    with a review of random variables and probability distributions. We will 
    then learn how to run a simulation by first looking at a simulation workflow 
    and then recreating it in the context of a game of dice. Finally, we will 
    learn how to use simulations for making decisions.
Source: https://learn.datacamp.com/courses/statistical-simulation-in-python
Help:
    Lesson 6: https://risk-engineering.org/notebook/coins-dice.html
"""

###############################################################################
## Importing libraries
###############################################################################
import numpy as np
import scipy.stats
import sympy.stats
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

###############################################################################
## Preparing the environment
###############################################################################
# Global configuration
plt.rcParams.update({'axes.labelsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 
                     'legend.fontsize': 8, 'font.size': 8})

#Global variables
suptitle_param = dict(color='darkblue', fontsize=12)
title_param    = {'color': 'darkred', 'fontsize': 14, 'weight': 'bold'}
figsize        = (12.1, 5.9)
SEED           = 123
size           = 10000


###############################################################################
## Main part of the code
###############################################################################
def Poisson_random_variable(seed=SEED):
    print("****************************************************")
    topic = "3. Poisson random variable"; print("** %s\n" % topic)
    # Initialize seed and parameters
    np.random.seed(seed) 
    lam, size_1, size_2 = 5, 3, 1000  
    
    # Draw samples & calculate absolute difference between lambda and sample mean
    samples_1 = np.random.poisson(lam, size_1)
    samples_2 = np.random.poisson(lam, size_2)
    answer_1 = abs(samples_1.mean() - lam)
    answer_2 = abs(samples_2.mean() - lam) 
    
    print(f"Theoric lambda: {lam}\n" +\
          f'Lambda in series 1: {samples_1.mean()}\n' +\
          f'Lambda in series 2: {samples_2.mean()}\n' +\
          "|Lambda - sample mean| with {} samples is {} and with {} samples is {}. ".format(size_1, answer_1, size_2, answer_2))

    
    
def Shuffling_a_deck_of_cards(seed=SEED):
    print("****************************************************")
    topic = "4. Shuffling a deck of cards"; print("** %s\n" % topic)
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    deck_of_cards = list(zip(['Heart']*13, range(13))) + \
                list(zip(['Club']*13, range(13))) + \
                list(zip(['Spade']*13, range(13))) + \
                list(zip(['Diamond']*13, range(13)))
    
    # Print deck_of_cards
    print(f"Deck of cards: \n{deck_of_cards}\n\n")
    
    # Shuffle the deck
    np.random.shuffle(deck_of_cards)
    
    # Print out the top three cards
    card_choices_after_shuffle = deck_of_cards[:3]
    print(f"Three first elements after shuffle: \n {card_choices_after_shuffle}")

    
    
def Throwing_a_fair_die(seed=SEED):
    print("****************************************************")
    topic = "6. Throwing a fair die"; print("** %s\n" % topic)
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print("------------------------------------USING NUMPY ONLY")
    print("FIRST TWO STEPS:")
    print("1. Define possible outcomes for random variables.")
    print("2. Assign probabilities.\n")
    # Define die outcomes and probabilities
    die, probabilities, throws = list(range(1,7)), list(np.repeat(1/6,6)), 1
    
    # Use np.random.choice to throw the die once and record the outcome
    outcome = np.random.choice(die, size=throws, p=probabilities)
    print("Outcome of one throw: {}\n\n".format(outcome[0]))
    
    print("-----------------------------------USING SCIPY STATS")
    #Simulating dice throws
    dice = scipy.stats.randint(1, 7)
    print("Record in 10 throws: ", dice.rvs(10, random_state=seed))
    print("Max value record: ", dice.rvs(10, random_state=seed).max())

    print("---------What is the probability of a die rolling 4?")
    print(dice.pmf(4))
    
    print("------What is the probability of rolling 4 or below?")
    print(dice.cdf(4))

    print("------------------What is the probability of rolling")
    print("------------------------between 2 and 4 (inclusive)?")
    print(dice.cdf(4) - dice.cdf(1))

    print("-----------------------------------USING SYMPY STATS")
    print("--------------------------------------Expected value")
    D = sympy.stats.Die("D", 6)
    expected_value = sympy.stats.E(D)
    print(expected_value, ' == ', float(expected_value))
    print(type(expected_value))

    print("----------------The probability of a dice roll of 4:")
    print(sympy.stats.P(sympy.Eq(D, 4)))
	
    print("------------------Probability of rolling 4 or below:")
    print(sympy.stats.P(D <= 4))

    print("-Probability of rolling between 2 and 4 (inclusive):")
    print(sympy.stats.P(sympy.And(D >= 2, D <= 4)))
 
    print("-----------------------------------VISUALIZE THE PMF")
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    
    # Pmf of one throw dice
    rolls = dice.rvs(size)
    values, counts = np.unique(rolls, return_counts=True)
    ax=axes[0]
    ax.stem(values, counts, use_line_collection=True);
    ax.axvline(dice.mean(), ls='--', lw=2, color='black', label='Expected value')
    ax.legend()
    ax.set_xlabel('One throw Dice')
    ax.set_ylabel('PMF')
    ax.set_title('PMF of 1 throw dice',**title_param)
    
    # Pmf of threw throw dice
    rolls = [dice.rvs(3).sum() for i in range(size)]
    values, counts = np.unique(rolls, return_counts=True)
    ax=axes[1]
    ax.stem(values, counts, use_line_collection=True);
    ax.axvline(dice.mean()*3, ls='--', lw=2, color='black', label='Expected value')
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Three throws Dice')
    ax.set_ylabel('PMF')
    ax.set_title('PMF of 3 throw dice',**title_param)
    
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, right=None, bottom=None, top=None, hspace=None, wspace=.5);
    plt.show()
    
    
def Throwing_two_fair_dice(seed=SEED):
    print("****************************************************")
    topic = "7. Throwing two fair dice"; print("** %s\n" % topic)
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print("------------------------------------USING NUMPY ONLY")
    print("THIRD STEP:")
    print("3. Define relationships between random variables.\n")
    # Initialize number of dice, simulate & record outcome
    #die, probabilities, num_dice = [1,2,3,4,5,6], [1/6, 1/6, 1/6, 1/6, 1/6, 1/6], 2
    die, probabilities, num_dice = range(1,7), np.repeat(1/6, 6), 2
    outcomes = np.random.choice(die, size=num_dice, p=probabilities) 

    # Win if the two dice show the same number
    if outcomes[0] == outcomes[1]: 
        answer = 'win' 
    else:
        answer = 'lose'
            
    print("The dice show {} and {}. You {}!".format(outcomes[0], outcomes[1], answer))
    
    
    
def Simulating_the_dice_game(seed=SEED):
    print("****************************************************")
    topic = "8. Simulating the dice game"; print("** %s\n" % topic)
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print("------------------------------------USING NUMPY ONLY")
    print("FOURTH STEP:")
    print("4. Get multiple outcomes by repeated random sampling.\n")
    
    # Initialize model parameters & simulate dice throw
    #die, probabilities, num_dice = [1,2,3,4,5,6], [1/6, 1/6, 1/6, 1/6, 1/6, 1/6], 2
    die, probabilities, num_dice = range(1,7), np.repeat(1/6, 6), 2
    size = 100
    wins = np.zeros(size)
    
    # Using numpy only
    for i in range(size):
        outcomes = np.random.choice(die, size=num_dice, p=probabilities)
        wins[i] = np.where((outcomes == np.max(outcomes)).all(), 1, 0)
    """
    # Using scipy.stats
    dice = scipy.stats.randint(1, 7)
    for i in range(size):
        outcomes = dice.rvs(num_dice)
        wins[i] = np.where((outcomes == np.max(outcomes)).all(), 1, 0)
    """
    print("In {:.0f} games, you win {:.0f} times".format(size, wins.sum()))
    
    print("-----------------------------------VISUALIZE THE PMF")
    fig, ax = plt.subplots()
    
    # Pmf of two equal throw dice
    values, counts = np.unique(wins, return_counts=True)
    
    ax.stem(values, counts, use_line_collection=True);
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Two equal throws Dice')
    ax.set_ylabel('PMF')
    ax.set_title('PMF of 2 equal throw dice',**title_param)
    ax.grid(True)
    fig.suptitle(topic, **suptitle_param)
    #plt.subplots_adjust(left=None, right=None, bottom=None, top=None, hspace=None, wspace=.5);
    plt.show()
    
    
    
def Simulating_one_lottery_drawing(seed=SEED):
    print("****************************************************")
    topic = "10. Simulating one lottery drawing"; print("** %s\n" % topic)
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print("------------------------------------FIRST TWO STEPS:")
    print("1. Define possible outcomes for random variables.")
    print("2. Assign probabilities.\n")
    # Pre-defined constant variables
    lottery_ticket_cost, num_tickets, grand_prize = 10, 1000, 1000000
    
    # Probability of winning
    chance_of_winning = 1/num_tickets
    probability = [1-chance_of_winning, chance_of_winning]
    
    print("----------------------------------------THIRD STEP:")
    print("3. Define relationships between random variables.\n")
    # Simulate a single drawing of the lottery
    gains = [-lottery_ticket_cost, grand_prize-lottery_ticket_cost]
    
    outcome = np.random.choice(a=gains, size=1, p=probability, replace=True)
    
    print("Outcome of one drawing of the lottery is {}".format(outcome))
    
    
def Should_we_buy(seed=SEED):
    print("****************************************************")
    topic = "11. Should we buy?"; print("** %s\n" % topic)
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print("------------------------------------FIRST TWO STEPS:")
    print("1. Define possible outcomes for random variables.")
    print("2. Assign probabilities.\n")
    # Initialize size and simulate outcome
    lottery_ticket_cost, num_tickets, grand_prize = 10, 1000, 1000000
    chance_of_winning = 1/num_tickets
    size = 2000
    probs = [1 - chance_of_winning, chance_of_winning]
    
    print("----------------------------------------THIRD STEP:")
    print("3. Define relationships between random variables.\n")
    payoffs = [-lottery_ticket_cost, grand_prize - lottery_ticket_cost]
    
    print("-------------------------------------------FOURTH STEP:")
    print("4. Get multiple outcomes by repeated random sampling.\n")
    outcomes = np.random.choice(a=payoffs, size=size, p=probs, replace=True)
    
    # Mean of outcomes.
    answer = outcomes.mean()
    print("Average payoff from {} simulations = $ {:,.0f}".format(size, answer))
    
    values, counts = np.unique(outcomes, return_counts=True)
    print(np.unique(outcomes, return_counts=True))
    print("Wins : {} times.".format(counts[np.where(values==payoffs[1])][0]))
    print("Losts: {} times.".format(counts[np.where(values==payoffs[0])][0]))
    
    
    
def Calculating_a_break_even_lottery_price(seed=SEED):
    print("****************************************************")
    topic = "12. Calculating a break-even lottery price"; print("** %s\n" % topic)
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print("------------------------------------FIRST TWO STEPS:")
    print("1. Define possible outcomes for random variables.")
    print("2. Assign probabilities.\n")
    grand_prize, num_tickets = 1000000, 1000
    chance_of_winning = 1/num_tickets
    
    print("----------------------------------------THIRD STEP:")
    print("3. Define relationships between random variables.\n")
    # Initialize simulations and cost of ticket
    sims, lottery_ticket_cost = 3000, 0
    
    print("-------------------------------------------FOURTH STEP:")
    print("4. Get multiple outcomes by repeated random sampling.\n")
    # Use a while loop to increment `lottery_ticket_cost` till average value of outcomes falls below zero
    while 1:
        outcomes = np.random.choice([-lottery_ticket_cost, grand_prize-lottery_ticket_cost],
                                    size=sims, p=[1-chance_of_winning, chance_of_winning], replace=True)
        mean_payoff = outcomes.mean()
        print(f"With a ticket cost of $ {lottery_ticket_cost:,.2f} we have a payoff of $ {mean_payoff:,.2f}")
        if mean_payoff < 0:
            break
        else:
            lottery_ticket_cost += 1
    
    print("--------------------------------------------FIFTH STEP:")
    print("5. Analyze sample outcomes.")
    
    answer = lottery_ticket_cost - 1
    print("The highest price at which it makes sense to buy the ticket is {}".format(answer))

    
     
    
def main():
    print("****************************************************")
    print("** BEGIN                                          **")
    
    Poisson_random_variable()
    Shuffling_a_deck_of_cards()
    
    Throwing_a_fair_die()
    Throwing_two_fair_dice(seed=223)
    Simulating_the_dice_game(seed=223)
    
    Simulating_one_lottery_drawing()
    Should_we_buy()
    Calculating_a_break_even_lottery_price(333)
            
    print("****************************************************")
    print("** END                                            **")
    print("****************************************************")

if __name__ == '__main__':
    main()
    plt.style.use('default')