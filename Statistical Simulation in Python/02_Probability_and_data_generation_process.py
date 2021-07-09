# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 2: Probability & data generation process
    This chapter provides a basic introduction to probability concepts and a 
    hands-on understanding of the data generating process. We'll look at a number 
    of examples of modeling the data generating process and will conclude with 
    modeling an eCommerce advertising simulation.
Source: https://learn.datacamp.com/courses/statistical-simulation-in-python
"""
###############################################################################
## Importing libraries
###############################################################################
import numpy as np

###############################################################################
## Preparing the environment
###############################################################################
SEED = 123
SIZE = 10000

###############################################################################
## Reading the data
###############################################################################


###############################################################################
## Main part of the code
###############################################################################
def Two_of_a_kind(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "3 Two of a kind"; print("** %s" % topic)
    print("****************************************************")
    
    print('Estimating the probability of getting at least two ' + \
          'of a kind. Two of a kind is when you get two cards ' + \
          'of different suites but having the same numeric ' + \
          'value.\n')
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    deck_of_cards = list(zip(['Heart']*13, range(13))) + \
                    list(zip(['Club']*13, range(13))) + \
                    list(zip(['Spade']*13, range(13))) + \
                    list(zip(['Diamond']*13, range(13)))
                
    # Shuffle deck & count card occurrences in the hand
    num_card_equal = 2
    wins = np.zeros(size)
    for i in range(size):
        np.random.shuffle(deck_of_cards)
        hand, cards_in_hand = deck_of_cards[0:5], {}
        # Use .get() method on cards_in_hand
        for card in hand: cards_in_hand[card[1]] = cards_in_hand.get(card[1], 0) + 1
        # Condition for getting at least 2 of a kind
        wins[i] = np.where( max(cards_in_hand.values()) >= num_card_equal, 1, 0) 

    print("Probability of seeing at least two of a kind = ", wins.mean())
    
    
    
def Game_of_thirteen(size=SIZE, seed=111):
    print("****************************************************")
    topic = "4 Game of thirteen"; print("** %s" % topic)
    print("****************************************************")
    
    print('You have a deck of 13 cards, each numbered from 1 ' + \
          'through 13. Shuffle this deck and draw cards one by ' + \
          'one. A coincidence is when the number on the card ' + \
          'matches the order in which the card is drawn. You ' + \
          'win the game if you get through all the cards without ' + \
          "any coincidences. Let's calculate the probability of " + \
          'winning at this game using simulation.\n')
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Pre-set constant variables
    deck_of_cards, wins = np.arange(1,14), np.zeros(size)
    for i in range(size):
        cards_in_hand = np.random.permutation(deck_of_cards)
        # Condition for getting at least 2 of a kind
        wins[i] = np.not_equal(deck_of_cards, cards_in_hand).all() 

    print("Probability of not matching the order of throws = ", wins.mean())
    
    
    
def The_conditional_urn(size=5000, seed=SEED):
    print("****************************************************")
    topic = "6 The conditional urn"; print("** %s" % topic)
    print("****************************************************")
    
    print('We have an urn that contains 7 white and 6 black balls. ' + \
          'Four balls are drawn at random. We would like to know ' + \
          'the probability that the first and third balls are white, ' + \
          'while the second and the fourth balls are black.\n')
    # Initialize seed and parameters
    np.random.seed(seed) 
    urn = np.append(np.repeat('W', 7), np.repeat('B', 6))
    match = ['W', 'B', 'W', 'B']
    wins = np.zeros(size)
    for i in range(size):
        wins[i] = (np.random.choice(urn, size=4, replace=False) == match).all()
    
    print("Probability of having {} = {} ".format(match, wins.mean()))
    
    
    
def Birthday_problem(size=2000, seed=111):
    print("****************************************************")
    topic = "7 Birthday problem"; print("** %s" % topic)
    print("****************************************************")
    
    print('How many people do you need in a room to ensure at least ' +\
          'a 50% chance that two of them share the same birthday?\n')
    
    print('-----------------------------------------First aprox')
    # Initialize seed and parameters
    np.random.seed(seed) 
    people, min_search, days_of_year = 2, 0.5, np.arange(1, 366)
    while (people > 0):
        wins = np.zeros(size)
        for i in range(size):
            birthday = np.random.choice(days_of_year, size=people, replace=True)
            wins[i] = (len(birthday) != len(set(birthday)))
        if (wins.mean() >= min_search): break
        people += 1
        
    print(f"With {people} people, there's a {min_search:.0%} chance that two share a birthday.")
    
    
    print('-------------------------------------Using functions')
    # Initialize seed and parameters
    np.random.seed(seed) 
    def pmf_same_birthday(people, size=size):
        days_of_year, wins = np.arange(1, 366), np.zeros(size)
        for i in range(size):
            birthday = np.random.choice(days_of_year, size=people, replace=True)
            wins[i] = (len(birthday) != len(set(birthday)))
        return (wins.mean())
    
    people, min_search  = 2, 0.5
    while (people > 0):
        if pmf_same_birthday(people) >= min_search: break
        people += 1
        
    print(f"With {people} people, there's a {min_search:.0%} chance that two share a birthday.")
    
    
    print('---------------------------------According to lesson')
    # Initialize seed and parameters
    np.random.seed(seed) 
    # Draw a sample of birthdays & check if each birthday is unique
    days = np.arange(1,366)
    people = 2

    def birthday_sim(people):
        sims, unique_birthdays = 2000, 0 
        for _ in range(sims):
            draw = np.random.choice(days, size=people, replace=True)
            if len(draw) == len(set(draw)): 
                unique_birthdays += 1
        out = 1 - unique_birthdays / sims
        return out
    
    # Break out of the loop if probability greater than 0.5
    while (people > 0):
        prop_bds = birthday_sim(people)
        if prop_bds > .5: 
            break
        people += 1
        
    print("With {} people, there's a 50% chance that two share a birthday.".format(people))

    
    
def Full_house(size=50000, seed=SEED):
    print("****************************************************")
    topic = "8 Full house"; print("** %s" % topic)
    print("****************************************************")
    
    print('Calculate the probability of getting a full house. A full ' + \
          'house is when you get two cards of different suits that ' + \
          'share the same numeric value and three other cards that ' + \
          'have the same numeric value.\n')
    
    deck = list(zip(['Heart']*13, range(13))) + \
           list(zip(['Club']*13, range(13))) + \
           list(zip(['Spade']*13, range(13))) + \
           list(zip(['Diamond']*13, range(13)))
                
    print('---------------------Using the example in exercise 3')
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Copy deck & init count full house occurrences in the hand
    deck_of_cards, wins = deck.copy(), np.zeros(size)
    for i in range(size):
        np.random.shuffle(deck_of_cards)
        hand, cards_in_hand = deck_of_cards[0:5], {}
        # Use .get() method on cards_in_hand
        for card in hand: cards_in_hand[card[1]] = cards_in_hand.get(card[1], 0) + 1
        # Condition for getting a full house
        wins[i] = (max(cards_in_hand.values()) == 3) & (min(cards_in_hand.values()) == 2)

    print("Probability of seeing a full house =", wins.mean())
    
    
    print('----------------------------------Improving the code')
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Copy deck & init count full house occurrences in the hand
    deck_of_cards, wins = deck.copy(), np.zeros(size)
    for i in range(size):
        np.random.shuffle(deck_of_cards)
        # Condition for getting a full house
        hand, values = np.unique(np.reshape(deck_of_cards[0:5], (5,2))[:,1], return_counts=True)
        wins[i] = (max(values)==3) & (min(values)==2)

    print("Probability of seeing a full house =", wins.mean())
    
    print('---------------------------------According to lesson')
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    #Shuffle deck & count card occurrences in the hand
    n_sims, full_house, deck_of_cards = size, 0, deck.copy() 
    for i in range(n_sims):
        np.random.shuffle(deck_of_cards)
        hand, cards_in_hand = deck_of_cards[0:5], {}
        for card in hand:
            # Use .get() method to count occurrences of each card
            cards_in_hand[card[1]] = cards_in_hand.get(card[1], 0) + 1
            
        # Condition for getting full house
        condition = (max(cards_in_hand.values()) ==3) & (min(cards_in_hand.values())==2)
        if  condition == True: 
            full_house += 1
    
    print("Probability of seeing a full house = {}".format(full_house/n_sims))
    
    
def Driving_test(size=1000, seed=222):
    print("****************************************************")
    topic = "10 Driving test"; print("** %s" % topic)
    print("****************************************************")
    
    print('Suppose that you are about to take a driving test tomorrow. ' +\
          'Based on your own practice and based on data you have gathered, ' +\
          'you know that the probability of you passing the test is 90% ' +\
          'when it is sunny and only 30% when it is raining. Your local ' +\
          'weather station forecasts that there iss a 40% chance of rain ' +\
          'tomorrow. Based on this information, you want to know what is ' +\
          'the probability of you passing the driving test tomorrow.\n')
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Pre-set constant variables
    wins_pass, p_rain, p_pass = np.zeros(size), 0.40, {'sun':0.9, 'rain':0.3}
    
    def test_outcome(p_rain):  
        # Simulate whether it will rain or not
        weather = np.random.choice(['rain', 'sun'], p=[p_rain, 1-p_rain])
        # Simulate and return whether you will pass or fail
        return np.random.choice([True, False], p=[p_pass[weather], 1-p_pass[weather]])
        
    for i in range(size):
        wins_pass[i] = test_outcome(p_rain)
    
    # Calculate fraction of outcomes where you pass
    pass_outcomes_frac = wins_pass.mean()
    print("Probability of Passing the driving test = {}".format(pass_outcomes_frac))
    
    
    
def National_elections(size=1000, seed=224):
    print("****************************************************")
    topic = "11 National elections"; print("** %s" % topic)
    print("****************************************************")
    
    print('Consider national elections in a country with two political ' +\
          'parties - Red and Blue. This country has 50 states and the ' +\
          'party that wins the most states wins the elections. You have ' +\
          'the probability p of Red winning in each individual state and ' +\
          'want to know the probability of Red winning nationally. Suppose ' +\
          'the election outcome in each state follows a binomial distribution ' +\
          'with probability p such that 0 indicates a loss for Red and 1 ' +\
          'indicates a win. What is the probability of Red winning less than ' +\
          '45% of the states?\n')
    
    probs = [0.52076814, 0.67846401, 0.82731745, 0.64722761, 0.03665174,
             0.17835411, 0.75296372, 0.22206157, 0.72778372, 0.28461556,
             0.72545221, 0.106571  , 0.09291364, 0.77535718, 0.51440142,
             0.89604586, 0.39376099, 0.24910244, 0.92518253, 0.08165597,
             0.4212476 , 0.74123879, 0.2479099 , 0.46125805, 0.19584491,
             0.24440482, 0.349916  , 0.80224624, 0.80186664, 0.82968251,
             0.91178779, 0.51739059, 0.67338858, 0.15675863, 0.37772308,
             0.77134621, 0.71727114, 0.92700912, 0.28386132, 0.25502498,
             0.30081506, 0.19724585, 0.29129564, 0.56623386, 0.97681039,
             0.96263926, 0.0548948 , 0.14092758, 0.54739446, 0.54555576]
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    wins_state = np.zeros(size)

    for i in range(size):
        # Simulate elections in the 50 states
        election = np.random.binomial(1, probs)
        # Get average of Red wins and add to `outcomes`
        wins_state[i] = election.mean()
        
    # Calculate probability of Red winning in less than 45% of the states
    prob_red_wins = np.mean([win < .45 for win in wins_state])
    
    print("Probability of Red winning in less than 45% of the states = {}".format(prob_red_wins))    
    
    
    
def Fitness_goals(size=1000, seed=222):
    print("****************************************************")
    topic = "12 Fitness goals"; print("** %s" % topic)
    print("****************************************************")
    
    print('On days when you go to the gym, you average around 15k steps, ' +\
          'and around 5k steps otherwise. You go to the gym 40% of the time. ' +\
          "Let's model the step counts in a day as a Poisson random variable " +\
          'with a mean λ dependent on whether or not you go to the gym. For ' +\
          'simplicity, let’s say you have an 80% chance of losing 1lb and a 20% ' +\
          'chance of gaining 1lb when you get more than 10k steps. The ' +\
          'probabilities are reversed when you get less than 8k steps. ' +\
          'Otherwise, there is an even chance of gaining or losing 1lb. ' +\
          'Given all this information, find the probability of losing weight ' +\
          'in a month.\n')
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    outcomes, days = np.zeros(size), 30
    for i in range(size):
        pounds = np.zeros(days)
        for k in range(days):
            lam = np.random.choice([5000, 15000], p=[.6, .4])
            steps = np.random.poisson(lam)
            probs = np.where(steps>10000, [.2, .8], 
                    np.where(steps<8000, [.8, .2], 
                    [.5,.5]))
            pounds[k] = np.random.choice([1, -1], p=probs)
        outcomes[i] = pounds.sum()
    
    prob_to_loss_weight = np.mean(outcomes<0)
    
    print("Probability of Weight Loss = {}".format(prob_to_loss_weight))
    
    
    
def eCommerce_Ad_Simulation(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "13 eCommerce Ad Simulation"; print("** %s" % topic)
    print("****************************************************")
    
    topic = "14 Sign up Flow"; print("** %s" % topic)
    print('Model the DGP of an eCommerce ad flow starting with sign-ups.')
    print('(1) On any day, we get many ad impressions, which can be modeled ' +\
          'as Poisson random variables (RV). You are told that λ is normally ' +\
          'distributed with a mean of 100k visitors and standard deviation ' +\
          '2000.')
    print('(2) During the signup journey, the customer sees an ad, decides ' +\
          'whether or not to click, and then whether or not to signup. ' +\
          'Thus both clicks and signups are binary, modeled using binomial RVs. ')
    print('What about probability p of success?')
    print('(3) Our current low-cost option gives us a click-through rate of 1% and ' +\
          'a sign-up rate of 20%. A higher cost option could increase the ' +\
          'clickthrough and signup rate by up to 20%, but we are unsure of the ' +\
          'level of improvement, so we model it as a uniform RV.\n')
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    lam_mu, lam_sigma = 100000, 2000
    low_ct_rate, low_su_rate = .01, .20
    ct_rate = {'low': low_ct_rate, 'high': np.random.uniform(low=low_ct_rate, high=low_ct_rate*1.2)}
    su_rate = {'low': low_su_rate, 'high': np.random.uniform(low=low_su_rate, high=low_su_rate*1.2)}
    
    def get_signups(cost, size, 
                    lam_mu=lam_mu, lam_sigma=lam_sigma, 
                    ct_rate=ct_rate, su_rate=su_rate):
        ad_impressions = np.random.poisson(lam = np.random.normal(loc=lam_mu, scale=lam_sigma, size=size))
        clicks = np.random.binomial(n=ad_impressions, p=ct_rate[cost])
        signups = np.random.binomial(n=clicks, p=su_rate[cost])
        return signups
        
    print("Simulated Signups = {}\n\n".format(get_signups('high', 1)))
    
    
    
    topic = "15 Purchase Flow"; print("** %s" % topic)
    print('Model the revenue generation process. ')
    print('(1) Once the customer has signed up, they decide whether or not ' +\
          'to purchase - a natural candidate for a binomial RV. ' +\
          "Let's assume that 10% of signups result in a purchase.")
    print('(2) Although customers can make many purchases. The purchase value ' +\
          'could be modeled by the exponential RV with an averaged around $1000. ')
    print('(3) The revenue, then, is the sum of all purchase values.\n')
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    def get_revenue(signups):
        purchases = np.random.binomial(n=signups, p=.1)
        total_purchases = np.zeros(len(purchases))
        for i in range(len(purchases)):
            purchase_values = np.random.exponential(scale=1000, size=purchases[i])
            total_purchases[i] = purchase_values.sum()
        return total_purchases
    
    signups = get_signups(cost='low', size=1)
    revenue = get_revenue(signups)
    print("Simulated Revenue = ${:,.2f}\n\n".format(revenue.sum()))
    #print("Simulated Revenue = ${:,.2f}\n\n".format(get_revenue(get_signups('low', 1)).sum()))
    
    
    topic = "16 Probability of losing money"; print("** %s" % topic)
    print('This company has the option of spending $3000, to redesign the ad.')
    print('This could potentially get them higher clickthrough and signup rates, ' +\
          'but this is not guaranteed. ')
    print('We would like to know whether or not to spend this extra $3000 ' +\
          'by calculating the probability of losing money. ')
    print('In other words, the probability that the revenue from the high-cost ' +\
          'option minus the revenue from the low-cost option is lesser than the ' +\
          'cost.\n')
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Initialize cost_diff
    cost_diff = 3000
    
    # Get revenue when the cost is 'low' and when the cost is 'high'
    rev_high = get_revenue(get_signups('high', size))
    rev_low = get_revenue(get_signups('low', size))
    
    # calculate fraction of times rev_high - rev_low is less than cost_diff
    frac_less_than_cost = np.mean((rev_high - rev_low) < cost_diff)
    print("Probability of losing money = {}".format(frac_less_than_cost))
    # calculate fraction of times rev_high - rev_low is less than cost_diff
    
    
        
def main():
    print("\n\n****************************************************")
    print("** BEGIN                                          **")
    
    Two_of_a_kind()
    Game_of_thirteen()
    
    The_conditional_urn()
    Birthday_problem()
    Full_house()
    
    Driving_test()
    National_elections()
    Fitness_goals()
    
    eCommerce_Ad_Simulation()
    
    print("\n\n****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()