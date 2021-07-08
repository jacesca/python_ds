# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 1: Chatbots 101
    In this chapter, you'll learn how to build your first chatbot. After gaining 
    a bit of historical context, you'll set up a basic structure for receiving text 
    and responding to users, and then learn how to add the basic elements of personality. 
    You'll then build rule-based systems for parsing text.
Source: https://learn.datacamp.com/courses/building-chatbots-in-python
"""
###############################################################################
## Importing libraries
###############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import time
import random
import re

###############################################################################
## Preparing the environment
###############################################################################
#Global variables
suptitle_param = dict(color='darkblue', fontsize=11)
title_param    = {'color': 'darkred', 'fontsize': 14, 'weight': 'bold'}
plot_param     = {'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                  'legend.fontsize': 8, 'font.size': 8}
figsize        = (12.1, 5.9)
SEED           = 42
SIZE           = 10000

# Global configuration
sns.set()
pd.set_option("display.max_columns",24)
plt.rcParams.update(**plot_param)
np.random.seed(SEED)

###############################################################################
## Reading the data
###############################################################################



###############################################################################
## Main part of the code
###############################################################################
def Introduction_to_Conversational_Software(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "1. Introduction to Conversational Software"; print("** %s" % topic)
    print("****************************************************")
    topic = "2. EchoBot I"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('-------------------------Define the respond function')
    bot_template = "BOT : {0}"
    user_template = "USER : {0}"
    
    # Define a function that responds to a user's message: respond
    def respond(message):
        # Concatenate the user's message to the end of a standard bot respone
        bot_message = "i can hear you! You said: " + message
        # Delay the response
        time.sleep(0.5)
        # Return the result
        return bot_message
    
    print('-----------------------------------------Interacting')
    # Test function
    print(respond("Hello!"))
    
    
    print("****************************************************")
    topic = "3. EchoBot II"; print("** %s" % topic)
    print("****************************************************")
    
    print('--------------------Define the send_message function')
    # Define a function that sends a message to the bot: send_message
    def send_message(message):
        # Print user_template including the user_message
        print(user_template.format(message))
        # Get the bot's response to the message
        response = respond(message)
        # Delay the response
        time.sleep(0.5)
        # Print the bot template including the bot's response.
        print(bot_template.format(response))
        
    print('-----------------------------------------Interacting')
    # Send a message to the bot
    send_message("Hello!")
    
    
    
def Chitchat(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "4. Creating a personality"; print("** %s" % topic)
    print("****************************************************")
    topic = "5. Chitchat"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('-------------------------Define the respond function')
    # Define variables
    name = "Greg"
    weather = "cloudy"
    
    # Define a dictionary with the predefined responses
    responses = {"what's your name?": "my name is {0}".format(name),
                 "what's today's weather?": "the weather is {0}".format(weather),
                 "default": "i'm not sure what are you taking about...",}
    
    # Return the matching response if there is one, default otherwise
    def respond(message):
        # Check if the message is in the responses
        if message in responses:
            # Return the matching message
            bot_message = responses[message]
        else:
            # Return the "default" message
            bot_message = responses["default"]
        return bot_message
    
    print('--------------------Define the send_message function')
    bot_template = "BOT : {0}"
    user_template = "USER : {0}"
    
    # Define a function that sends a message to the bot: send_message
    def send_message(message):
        # Print user_template including the user_message
        print(user_template.format(message))
        # Get the bot's response to the message
        response = respond(message)
        # Delay the response
        time.sleep(0.5)
        # Print the bot template including the bot's response.
        print(bot_template.format(response))
        
    print('-----------------------------------------Interacting')
    send_message("what's your name?")
    send_message("what's your favorite color?")
    send_message("what's today's weather?")
    
    
    
def Adding_variety(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "6. Adding variety"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('-------------------------Define the respond function')
    # Define variables
    name = "Greg"
    weather = "cloudy"
    
    # Define a dictionary containing a list of responses for each message
    responses = {"what's your name?"      : ["my name is {0}".format(name),
                                             "they call me {0}".format(name),
                                             "i go by {0}".format(name),],
                 "what's today's weather?": ["the weather is {0}".format(weather),
                                             "it's {0} today".format(weather),],
                 "default"                : ["i'm not sure about what i would say..."],}
    
    # Use random.choice() to choose a matching response
    def respond(message):
        if message in responses:
            bot_message = random.choice(responses[message])
        else:
            bot_message = random.choice(responses["default"])
        return bot_message
    
    print('--------------------Define the send_message function')
    bot_template = "BOT : {0}"
    user_template = "USER : {0}"
    
    # Define a function that sends a message to the bot: send_message
    def send_message(message):
        # Print user_template including the user_message
        print(user_template.format(message))
        # Get the bot's response to the message
        response = respond(message)
        # Delay the response
        time.sleep(0.5)
        # Print the bot template including the bot's response.
        print(bot_template.format(response))
        
    print('-----------------------------------------Interacting')
    send_message("what's your name?")
    send_message("what's your name?")
    send_message("what's your name?")
    
    
    
def ELIZA_I_asking_questions(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "7. ELIZA I: asking questions"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('-------------------------Define the respond function')
    responses = {'question' : ["i don't know :(", 
                               "i don't hav any idea", 
                               "i'm not sure", 
                               'you tell me!'],
                 'statement': ['tell me more!',
                               'why do you think that?',
                               'how long have you felt this way?',
                               'i find that extremely interesting',
                               'can you back that up?',
                               'oh wow!',
                               ':)']}
    def respond(message):
        # Check for a question mark
        if message.endswith('?'):
            # Return a random question
            return random.choice(responses["question"])
        # Return a random statement
        return random.choice(responses["statement"])
    
    
    print('--------------------Define the send_message function')
    bot_template = "BOT : {0}"
    user_template = "USER : {0}"
    
    # Define a function that sends a message to the bot: send_message
    def send_message(message):
        # Print user_template including the user_message
        print(user_template.format(message))
        # Get the bot's response to the message
        response = respond(message)
        # Delay the response
        time.sleep(0.5)
        # Print the bot template including the bot's response.
        print(bot_template.format(response))
        
    print('-----------------------------------------Interacting')
    # Send messages ending in a question mark
    send_message("what's today's weather?")
    send_message("what happened yesterday?")
    
    # Send messages which don't end with a question mark
    send_message("I love building chatbots")
    send_message("I'm enjoying this class")
    
    
    
def ELIZA_II_Extracting_key_phrases(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "8. Text Munging with regular expressions"; print("** %s" % topic)
    print("****************************************************")
    topic = "9. ELIZA II: Extracting key phrases"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('-------------------Define the swap_pronouns function')
    def swap_pronouns(phrase):
        if 'I' in phrase:
            return re.sub('i', 'you', phrase)
        if 'my' in phrase:
            return re.sub('my', 'your', phrase)
        if 'You' in phrase:
            return re.sub('you', 'i', phrase)
        if 'your' in phrase:
            return re.sub('your', 'my', phrase)
        else:
            return phrase

    print('----------------------Define the match_rule function')
    rules = {r'I want (.*)'         : ['what would it mean if you got {0}?',
                                       'why do you want {0}',
                                       "what's stopping you from getting {0}"],
             r'do you remember (.*)': ['did you think I would forget {0}',
                                       "why haven't you been able to forget {0}",
                                       'what about {0}',
                                       'yes... why?'],
             r'do you think (.*)'   : ['if {0} Absolutely.', 
                                       'no chance'],
             r'if (.*)'             : ["do you really think it's likely that {0}",
                                       'do you wish that {0}',
                                       'what do you think about {0}',
                                       'really, if {0}']}
    # Define match_rule()
    def match_rule(rules, message):
        response, phrase = "default", None
        
        # Iterate over the rules dictionary
        for pattern, responses in rules.items():
            # Create a match object
            match = re.search(pattern, message)
            if match is not None:
                # Choose a random response
                response = random.choice(responses)
                if '{0}' in response:
                    phrase = swap_pronouns(match.group(1))
        # Return the response and phrase
        return response.format(phrase)

    print('-------------------------Define the respond function')
    def respond(message):
        # Return a respond
        return match_rule(rules, message)
    
    
    print('--------------------Define the send_message function')
    bot_template = "BOT : {0}"
    user_template = "USER : {0}"
    
    # Define a function that sends a message to the bot: send_message
    def send_message(message):
        # Print user_template including the user_message
        print(user_template.format(message))
        # Get the bot's response to the message
        response = respond(message)
        # Delay the response
        time.sleep(0.5)
        # Print the bot template including the bot's response.
        print(bot_template.format(response))
        
    print('-----------------------------------------Interacting')
    # Send messages ending in a question mark
    send_message("do you remember your last birthday?")
    
    
    
def ELIZA_IV_Putting_it_all_together(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "10. ELIZA III: Pronouns"; print("** %s" % topic)
    print("****************************************************")
    topic = "11. ELIZA IV: Putting it all together"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('-------------------Define the swap_pronouns function')
    # Define replace_pronouns()
    def replace_pronouns(message):
        message = message.lower()
        if 'me' in message:
            # Replace 'me' with 'you'
            return re.sub('me', 'you', message)
        if 'my' in message:
            # Replace 'my' with 'your'
            return re.sub('my', 'your', message)
        if 'your' in message:
            # Replace 'your' with 'my'
            return re.sub('your', 'my', message)
        if 'you' in message:
            # Replace 'you' with 'me'
            return re.sub('you', 'me', message)
        
        return message


    print('----------------------Define the match_rule function')
    rules = {r'I want (.*)'         : ['what would it mean if you got {0}',
                                       'why do you want {0}',
                                       "what's stopping you from getting {0}"],
             r'do you remember (.*)': ['did you think I would forget {0}',
                                       "why haven't you been able to forget {0}",
                                       'what about {0}',
                                       'yes... why?'],
             r'do you think (.*)'   : ['if {0} Absolutely.', 
                                       'no chance'],
             r'if (.*)'             : ["do you really think it's likely that {0}",
                                       'do you wish that {0}',
                                       'what do you think about {0}',
                                       'really, if {0}']}
    
    # Define match_rule()
    def match_rule(rules, message):
        response, phrase = "default", None
        
        # Iterate over the rules dictionary
        for pattern, responses in rules.items():
            # Create a match object
            match = re.search(pattern, message)
            if match is not None:
                # Choose a random response
                response = random.choice(responses)
                if '{0}' in response:
                    phrase = match.group(1)
        # Return the response and phrase
        return response, phrase

    print('-------------------------Define the respond function')
    # Define respond()
    def respond(message):
        # Call match_rule
        response, phrase = match_rule(rules, message)
        if '{0}' in response:
            # Replace the pronouns in the phrase
            phrase = replace_pronouns(phrase)
            # Include the phrase in the response
            response = response.format(phrase)
        
        return response

    print('--------------------Define the send_message function')
    bot_template = "BOT : {0}"
    user_template = "USER : {0}"
    
    # Define a function that sends a message to the bot: send_message
    def send_message(message):
        # Print user_template including the user_message
        print(user_template.format(message))
        # Get the bot's response to the message
        response = respond(message)
        # Delay the response
        time.sleep(0.5)
        # Print the bot template including the bot's response.
        print(bot_template.format(response))
        
    print('-----------------------------------------Interacting')
    # Send the messages
    send_message("do you remember your last birthday?")
    send_message("do you think humans should be worried about AI?")
    send_message("I want a robot friend")
    send_message("what if you could be anything you wanted?")
    
    
        
def main():
    print("****************************************************")
    print("** BEGIN                                          **")
    
    Introduction_to_Conversational_Software()
    
    Chitchat()
    Adding_variety()
    ELIZA_I_asking_questions()
    
    ELIZA_II_Extracting_key_phrases()
    ELIZA_IV_Putting_it_all_together()

    
    print("****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    pd.reset_option("display.max_columns")
    plt.style.use('default')
    np.set_printoptions(formatter={'float': None})