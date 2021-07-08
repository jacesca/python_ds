# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 4: Dialogue
    Everything you've built so far has statelessly mapped intents to actions 
    and responses. It's amazing how far you can get with that! But to build 
    more sophisticated bots you will always want to add some statefulness. 
    That's what you'll do here, as you build a chatbot that helps users order 
    coffee.
Source: https://learn.datacamp.com/courses/building-chatbots-in-python
"""
###############################################################################
## Importing libraries
###############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sqlite3
import string
import re
import random
#from pprint import pprint

from rasa_nlu.converters import load_data #rasa_nlu.__version__==0.11.3 #To generate the json file: https://rodrigopivi.github.io/Chatito/ #Documentation: https://legacy-docs.rasa.com/docs/nlu/0.11.4/config/
from rasa_nlu.config import RasaNLUConfig #rasa_nlu.__version__==0.11.3 #Documentation: https://legacy-docs.rasa.com/docs/nlu/0.11.4/config/
from rasa_nlu.model import Trainer #rasa_nlu.__version__==0.11.3 #Documentation: https://legacy-docs.rasa.com/docs/nlu/0.11.4/config/

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
## Setting the interpreter
###############################################################################
# Create args dictionary
pipeline = ["nlp_spacy", "tokenizer_spacy", "ner_crf", "ner_synonyms"]
args = {'pipeline'       : pipeline, 
        'language'       : 'en',
        "entity_synonyms": "data/entity_synonyms.json"}

# Create a configuration and trainer
config = RasaNLUConfig(cmdline_args=args)
trainer = Trainer(config)

# Load the training data
training_data = load_data("data/training_data_chapter3.json")

# Create an interpreter by training the model
interpreter = trainer.train(training_data)

###############################################################################
## Main part of the code
###############################################################################
def Form_filling(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "2. Form filling"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------------------------Global variables')
    # Define the INIT state
    INIT = 0
    
    # Define the CHOOSE_COFFEE state
    CHOOSE_COFFEE = 1
    
    # Define the ORDERED state
    ORDERED = 2
    
    # Define the policy rules
    policy = {(INIT, "order"): (CHOOSE_COFFEE, "ok, Colombian or Kenyan?"),
              (INIT, "none"): (INIT, "I'm sorry - I'm not sure how to help you"),
              (CHOOSE_COFFEE, "specify_coffee"): (ORDERED, "perfect, the beans are on their way!"),
              (CHOOSE_COFFEE, "none"): (CHOOSE_COFFEE, "I'm sorry - would you like Colombian or Kenyan?"),
              }
    
    print('------------------------------------Global functions')
    def send_message(policy, state, message):
        print("USER : {}".format(message))
        new_state, response = respond(policy, state, message)
        print("BOT  : {}".format(response))
        return new_state
    
    def interpret(message):
        msg = message.lower()
        if 'order' in msg:
            return 'order'
        if 'kenyan' in msg or 'colombian' in msg:
            return 'specify_coffee'
        return 'none'
    
    def respond(policy, state, message):
        (new_state, response) = policy[(state, interpret(message))]
        return new_state, response
    
    print('---------------------------Ready to the conversation')
    # Create the list of messages
    messages = ["I'd like to become a professional dancer",
                "well then I'd like to order some coffee",
                "my favourite animal is a zebra",
                "kenyan"
                ]
    
    # Call send_message() for each message
    state = INIT
    for message in messages:    
        state = send_message(policy, state, message)    
    
    
def Asking_contextual_questions(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "3. Asking contextual questions"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------------------------Global variables')
    # Define the states
    INIT=0 
    CHOOSE_COFFEE=1
    ORDERED=2
    
    # Define the policy rules dictionary
    policy_rules = {(INIT, "ask_explanation"): (INIT, "I'm a bot to help you order coffee beans"),
                    (INIT, "order"): (CHOOSE_COFFEE, "ok, Colombian or Kenyan?"),
                    (CHOOSE_COFFEE, "specify_coffee"): (ORDERED, "perfect, the beans are on their way!"),
                    (CHOOSE_COFFEE, "ask_explanation"): (CHOOSE_COFFEE, "We have two kinds of coffee beans - the Kenyan ones make a slightly sweeter coffee, and cost $6. The Brazilian beans make a nutty coffee and cost $5.")    
                    }
    
    print('------------------------------------Global functions')
    def send_message(state, message):
        print("USER : {}".format(message))
        new_state, response = respond(state, message)
        print("BOT  : {}".format(response))
        return new_state
    
    def interpret(message):
        msg = message.lower()
        if 'order' in msg:
            return 'order'
        if 'kenyan' in msg or 'colombian' in msg:
            return 'specify_coffee'
        if 'what' in msg:
            return 'ask_explanation'
        return 'none'

    def respond(state, message):
        (new_state, response) = policy_rules[(state, interpret(message))]
        return new_state, response
    
    # Define send_messages()
    def send_messages(messages):
        state = INIT
        for msg in messages:
            state = send_message(state, msg)
    
    print('---------------------------Ready to the conversation')
    conversation = ["what can you do for me?",
                    "well then I'd like to order some coffee",
                    "what do you mean by that?",
                    "kenyan"
                    ]
    
    # Send the messages
    send_messages(conversation)
    
    
def Dealing_with_rejection(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "4. Dealing with rejection"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------------------------Global variables')
    responses = ["I'm sorry :( I couldn't find anything like that",
                 '{} is a great hotel!',
                 '{} or {} would work!',
                 '{} is one option, but I know others too :)']
    
    print('------------------------------------Global functions')
    def interpret(message):
        data = interpreter.parse(message)
        if 'no' in message:
            data["intent"]["name"] = "deny"
        return data

    def find_hotels(params, excluded):
        query = 'SELECT * FROM hotels'
        if len(params) > 0:
            filters = ["{}=?".format(k) for k in params] + \
                      ["name!='{}'".format(k) for k in excluded] 
            query += " WHERE " + " and ".join(filters)
        t = tuple(params.values())
        #print(f"      query       = {query}")
        # open connection to DB
        conn = sqlite3.connect('hotels.db')
        # create a cursor
        c = conn.cursor()
        c.execute(query, t)
        return c.fetchall()
    
    # Define respond()
    def respond(message, params, prev_suggestions, excluded):
        # Interpret the message
        parse_data = interpret(message)
        #print("      parse_data  ="); pprint(parse_data)
        # Extract the intent
        intent = parse_data['intent']['name']
        # Extract the entities
        entities = parse_data['entities']
        # Add the suggestion to the excluded list if intent is "deny"
        if intent == "deny":
            excluded.extend(prev_suggestions)
        # Fill the dictionary with entities	
        for ent in entities:
            params[ent["entity"]] = str(ent["value"])
        # Find matching hotels
        results = [
            r 
            for r in find_hotels(params, excluded) 
            if r[0] not in excluded
        ]
        # Extract the suggestions
        names = [r[0] for r in results]
        n = min(len(results), 3)
        suggestions = names[:2] if n==2 else names[:1]
        return responses[n].format(*names), params, suggestions, excluded
    
    print('---------------------------Ready to the conversation')
    # Initialize the empty dictionary and lists
    params, suggestions, excluded = {}, [], []
    
    conversation = ["I want a mid range hotel", "no that doesn't work for me"]
    
    # Send the messages
    for message in conversation:
        print("USER: {}".format(message))
        response, params, suggestions, excluded = respond(message, params, suggestions, excluded)
        print(f"      suggestions = {suggestions}")
        print(f"      excluded    = {excluded}")
        print(f"      params      = {params}")
        print("BOT : {}".format(response))
    
    
def Asking_questions_and_queuing_answers(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "5. Asking questions & queuing answers"; print("** %s" % topic)
    print("****************************************************")
    topic = "6. Pending actions I"; print("** %s" % topic)
    print("****************************************************")
    topic = "7. Pending actions II"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------------------------Global functions')
    def interpret(message):
        msg = message.lower()
        if 'order' in msg:
            return 'order'
        elif 'yes' in msg:
            return 'affirm'
        elif 'no' in msg:
            return 'deny'
        return 'none'
    
    # Define policy()
    def policy(intent):
        # Return "do_pending" if the intent is "affirm"
        if intent == "affirm":
            return "do_pending", None
        # Return "Ok" if the intent is "deny"
        if intent == "deny":
            return "Ok", None
        if intent == "order":
            return "Unfortunately, the Kenyan coffee is currently out of stock, would you like to order the Brazilian beans?", "Alright, I've ordered that for you!"
    
    # Define send_message()
    def send_message(pending, message):
        print("USER : {}".format(message))
        action, pending_action = policy(interpret(message))
        if action == "do_pending" and pending is not None:
            print("BOT  : {}".format(pending))
        else:
            print("BOT  : {}".format(action))
        return pending_action
        
    # Define send_messages()
    def send_messages(messages):
        pending = None
        for msg in messages:
            pending = send_message(pending, msg)
    
    print('---------------------------Ready to the conversation')
    conversation = ["I'd like to order some coffee",
                    "ok yes please"]
    
    # Send the messages
    send_messages(conversation)
    
    
    
def Pending_state_transitions(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "8. Pending state transitions"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------------------------Global variables')
    # Define the states
    INIT=0
    AUTHED=1
    CHOOSE_COFFEE=2
    ORDERED=3
    
    # Define the policy rules
    policy_rules = {
        (INIT, "order")                  : (INIT, "you'll have to log in first, what's your phone number?", AUTHED),
        (INIT, "number")                 : (AUTHED, "perfect, welcome back!", None),
        (AUTHED, "order")                : (CHOOSE_COFFEE, "would you like Colombian or Kenyan?", None),    
        (CHOOSE_COFFEE, "specify_coffee"): (ORDERED, "perfect, the beans are on their way!", None)
    }
    
    print('------------------------------------Global functions')
    def interpret(message):
        msg = message.lower()
        if 'order' in msg:
            return 'order'
        if 'kenyan' in msg or 'colombian' in msg:
            return 'specify_coffee'
        if any([d in msg for d in string.digits]):
            return 'number'    
        return 'none'

    def send_message(state, pending, message):
        print("USER : {}".format(message))
        new_state, response, pending_state = policy_rules[(state, interpret(message))]
        
        print("BOT  : {}".format(response))
        
        if pending is not None:
            new_state, response, pending_state = policy_rules[pending]
            print("BOT : {}".format(response))  
            pending = None
        
        if pending_state is not None:
            pending = (pending_state, interpret(message))
        
        return new_state, pending
    
    # Define send_messages()
    def send_messages(messages):
        state = INIT
        pending = None
        for msg in messages:
            state, pending = send_message(state, pending, msg)
    
    print('---------------------------Ready to the conversation')
    conversation = ["I'd like to order some coffee",
                    "555-1234",
                    "kenyan"]
    # Send the messages
    send_messages(conversation)
    
    
def Putting_it_all_together_I(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "9. Putting it all together I"; print("** %s" % topic)
    print("****************************************************")
    topic = "10. Putting it all together II"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------------------------Global variables')
    # Define the states
    INIT=0
    AUTHED=1
    CHOOSE_COFFEE=2
    ORDERED=3
    
    eliza_rules = {'I want (.*)'         : ['What would it mean if you got {0}',
                                            'Why do you want {0}',
                                            "What's stopping you from getting {0}"],
                   'do you remember (.*)': ['Did you think I would forget {0}',
                                            "Why haven't you been able to forget {0}",
                                            'What about {0}',
                                            'Yes .. and?'],
                   'do you think (.*)'   : ['if {0}? Absolutely.', 
                                            'No chance'],
                   'if (.*)'	         : ["Do you really think it's likely that {0}",
                                            'Do you wish that {0}',
                                            'What do you think about {0}',
                                            'Really--if {0}']}
    # Define the policy rules
    policy_rules = {(INIT, 'number')                 : (AUTHED, 'perfect, welcome back!', None),
                    (INIT, 'order')                  : (INIT, "you'll have to log in first, what's your phone number?", 1),
                    (AUTHED, 'order')                : (CHOOSE_COFFEE, 'would you like Colombian or Kenyan?', None),
                    (CHOOSE_COFFEE, 'specify_coffee'): (ORDERED, 'perfect, the beans are on their way!', None)}

    print('------------------------------------Global functions')
    def match_rule(rules, message):
        for pattern, responses in rules.items():
            match = re.search(pattern, message)
            if match is not None:
                response = random.choice(responses)
                var = match.group(1) if '{0}' in response else None
                return response, var
        return "default", None
    
    def replace_pronouns(message):
    
        message = message.lower()
        if 'me' in message:
            return re.sub('me', 'you', message)
        if 'i' in message:
            return re.sub('i', 'you', message)
        elif 'my' in message:
            return re.sub('my', 'your', message)
        elif 'your' in message:
            return re.sub('your', 'my', message)
        elif 'you' in message:
            return re.sub('you', 'me', message)
    
        return message
    
    def interpret(message):
        msg = message.lower()
        if 'order' in msg:
            return 'order'
        if 'kenyan' in msg or 'colombian' in msg:
            return 'specify_coffee'
        if any([d in msg for d in string.digits]):
            return 'number'    
        return 'none'

    
    # Define chitchat_response()
    def chitchat_response(message):
        # Call match_rule()
        response, phrase = match_rule(eliza_rules, message)
        # Return none if response is "default"
        if response == "default":
            return None
        if '{0}' in response:
            # Replace the pronouns of phrase
            phrase = replace_pronouns(phrase)
            # Calculate the response
            response = response.format(phrase)
        return response

    # Define send_message()
    def send_message(state, pending, message):
        print("USER : {}".format(message))
        response = chitchat_response(message)
        if response is not None:
            print("BOT : {}".format(response))
            return state, None
        
        # Calculate the new_state, response, and pending_state
        new_state, response, pending_state = policy_rules[(state, interpret(message))]
        print("BOT : {}".format(response))
        if pending is not None:
            new_state, response, pending_state = policy_rules[pending]
            print("BOT : {}".format(response))        
        if pending_state is not None:
            pending = (pending_state, interpret(message))
        return new_state, pending
    
    # Define send_messages()
    def send_messages(messages):
        state = INIT
        pending = None
        for msg in messages:
            state, pending = send_message(state, pending, msg)
    
    print('---------------------------Ready to the conversation')
    conversation = ["I'd like to order some coffee",
                    "555-12345",
                    "do you remember when I ordered 1000 kilos by accident?",
                    "kenyan"]
    
    # Send the messages
    send_messages(conversation)  
    
def Generating_text_with_neural_networks(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "12. Generating text with neural networks"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------------------------Global variables')
    generated = {0.2: "i'm gonna punch lenny in the back of the been a to the on the man to the mother and the father to simpson the father to with the marge in the for the like the fame to the been to the for my bart the don't was in the like the for the father the father a was the father been a say the been to me the do it and the father been to go. i want to the boy i can the from a man to be the for the been a like the father to make my bart of the father",
                 0.5: "i'm gonna punch lenny in the back of the kin't she change and i'm all better it and the was the fad a drivera it? what i want to did hey, he would you would in your bus who know is the like and this don't are for your this all for your manset the for it a man is on the see the will they want to know i'm are for one start of that and i got the better this is. it whoce and i don't are on the mater stop in the from a for the be your mileat",
                 1.0: "i'm gonna punch lenny in the back of the to to macks how screath. firl done we wouldn't wil that kill. of this torshmobote since, i know i ord did, can give crika of sintenn prescoam.whover my me after may? there's right. that up. there's ruining isay.oh.solls.nan'h those off point chuncing car your anal medion.hey, are exallies a off while bea dolk of sure, hello, no in her, we'll rundems... i'm eventy taving me to too the letberngonce",
                 1.2: "i'm gonna punch lenny in the back of the burear prespe-nakes, 'lisa to isn't that godios.and when be the bowniday' would lochs meine, mind crikvin' suhle ovotaci!..... hey, a poielyfd othe flancer, this in are rightplouten of of we doll hurrs, truelturone? rake inswaydan justy!we scrikent.ow.. by back hous, smadge, the lighel irely.yes, homer. wel'e esasmoy ryelalrs all wronencay...... nank. i wenth makedyk. come on help cerzind, now, n"}
    
    print('------------------------------------Global functions')
    def sample_text(seed, temperature):
        return generated[temperature]    
    
    print('---------------------------Ready to the conversation')
    # Feed the seed text into the neural network
    seed = "i'm gonna punch lenny in the back of the"

    # Iterate over the different temperature values
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print("\nGenerating text with riskiness : {}\n".format(temperature))
        # Call the sample_text function
        print(sample_text(seed, temperature))   
       
    
        
def main():
    print("****************************************************")
    print("** BEGIN                                          **")
    
    Form_filling()
    Asking_contextual_questions()
    Dealing_with_rejection()
    Asking_questions_and_queuing_answers()
    Pending_state_transitions()
    Putting_it_all_together_I()
    Generating_text_with_neural_networks()
    
    
    print("****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    pd.reset_option("display.max_columns")
    plt.style.use('default')
    np.set_printoptions(formatter={'float': None})