# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 3: Building a virtual assistant
    In this chapter, you'll build a personal assistant to help you plan a trip. 
    It will be able to respond to questions like "are there any cheap hotels in 
    the north of town?" by looking inside a hotel’s database for matching 
    results.
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
from pprint import pprint 

#from rasa_nlu.model import Metadata #to load the metadata of your model
#from rasa_nlu.model import Interpreter
#from rasa_nlu.components import ComponentBuilder #If multiple models are created, it is reasonable to share components between the different models. E.g. the 'nlp_spacy' component, which is used by every pipeline that wants to have access to the spacy word vectors, can be cached to avoid storing the large word vectors more than once in main memory. To use the caching, a ComponentBuilder should be passed when loading and training models.
from rasa_nlu.converters import load_data #rasa_nlu.__version__==0.11.3 #To generate the json file: https://rodrigopivi.github.io/Chatito/ #Documentation: https://legacy-docs.rasa.com/docs/nlu/0.11.4/config/
from rasa_nlu.config import RasaNLUConfig #rasa_nlu.__version__==0.11.3 #Documentation: https://legacy-docs.rasa.com/docs/nlu/0.11.4/config/
from rasa_nlu.model import Trainer #rasa_nlu.__version__==0.11.3 #Documentation: https://legacy-docs.rasa.com/docs/nlu/0.11.4/config/

#from pandas.core.common import flatten

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
#pipeline = ["nlp_spacy", "tokenizer_spacy", "ner_crf"]
#args = {'pipeline': pipeline, 'language': 'en'}
pipeline = ["nlp_spacy", "tokenizer_spacy", "ner_crf", "ner_synonyms"]
args = {'pipeline'       : pipeline, 
        'language'       : 'en',
        "entity_synonyms": "data/entity_synonyms.json"}

# Create a configuration and trainer
config = RasaNLUConfig(cmdline_args=args)
trainer = Trainer(config)

# Load the training data
#training_data = load_data("data/training_data.json")
training_data = load_data("data/training_data_chapter3.json")

# Create an interpreter by training the model
interpreter = trainer.train(training_data)

"""
model_directory = 'data/'

# will cache components between pipelines (where possible)
#builder = ComponentBuilder(use_cache=True)

# where `model_directory points to the folder the model is persisted in
interpreter = Interpreter.load(model_directory)#, config)
#interpreter = Interpreter.load(model_directory, config, builder)
"""
    
    
###############################################################################
## Main part of the code
###############################################################################
def Virtual_Assistants_and_accessing_data(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "1. Virtual Assistants and accessing data"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('----------------------------------SQLite with Python')
    conn = sqlite3.connect('hotels.db')
    c = conn.cursor()
    c.execute("""
              SELECT * 
              FROM hotels 
              """)
    # Print the columns name
    print(list(np.asarray(c.description)[:,0]))
    # Print the data
    pprint(c.fetchall())
    
    print('----------------------------SQL injection - BAD IDEA')
    area = 'south'
    price = 'hi'
    
    query = """
            SELECT name 
            FROM hotels 
            WHERE area='{}'
                and price='{}'
           """.format(area, price)
    c.execute(query)
    pprint(c.fetchall())
    
    print('--------------------SQL injection - AVOIDED (BETTER)')
    t = (area, price)
    query = """
            SELECT name 
            FROM hotels 
            WHERE area=?
                and price=?
           """
    c.execute(query, t)
    pprint(c.fetchall())
    
    #Close connection
    conn.close()
    
def SQL_basics(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "2. SQL basics"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('----------------Fetching only the "Hotel California"')
    conn = sqlite3.connect('hotels.db')
    c = conn.cursor()
    
    query = "SELECT name from hotels where price = 'mid' AND area = 'north'"
    
    pprint(c.execute(query).fetchall())
    
    #Close connection
    conn.close()
        
def SQL_statements_in_Python(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "3. SQL statements in Python"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------------Pass parameters the safe way')
    # Open connection to DB
    conn = sqlite3.connect('hotels.db')
    
    # Create a cursor
    c = conn.cursor()
    
    # Define area and price
    area, price = "south", "hi"
    t = (area, price)
    
    # Execute the query
    c.execute('SELECT * FROM hotels WHERE area=? AND price=?', t)
    
    # Print the results
    pprint(c.fetchall())
        
    #Close connection
    conn.close()
    
    
def Exploring_a_DB_with_natural_language(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "4. Exploring a DB with natural language"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('---------------------------------------------message')
    message = "a cheap hotel in the north"
    print(message)
    
    print('--------------------------------Parameters from text')
    #Parse some messages
    data = interpreter.parse(message)
    
    params = {}
    param_dictionary = {'location':'area'}
    for ent in data["entities"]:
        if ent["entity"] in param_dictionary.keys():
            params[param_dictionary[ent["entity"]]] = ent["value"]
        else:
            params[ent["entity"]] = ent["value"]
    print(params)
    
    print('---------------------------------------------filters')
    query = "select name FROM hotels"
    filters = ["{}=?".format(k) for k in params.keys()]
    print(filters)
    
    print('------------------------------------------conditions')
    conditions = " and ".join(filters)
    print(conditions)
    
    print('-----------------------------------------final query')
    final_q = " WHERE ".join([query, conditions])
    print(final_q)
    
    print('-------------------------------------------responses')
    responses = ["I'm sorry :( I couldn't find anything like that",
                 "what about {}?",
                 "{} is one option, but I know others too :)"
                ]
    
    # Open connection to DB
    conn = sqlite3.connect('hotels.db')
    
    # Create a cursor
    c = conn.cursor()
    
    # Define query filter params
    t = tuple(params.values())
    
    # Execute the query
    c.execute(final_q, t)
    
    # Get the results
    results = c.fetchall()
    #results = c.fetchone()
    
    #Close connection
    conn.close()
    
    index = min(len(results), len(responses)-1)
    names = [row[0] for row in results]
    #print(responses[index].format(list(flatten(results)))) #All results
    print(responses[index].format(*names)) #Just the first
    
    
    
def Creating_queries_from_parameters(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "5. Creating queries from parameters"; print("** %s" % topic)
    print("****************************************************")
    topic = "6. Using your custom function to find hotels"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Define find_hotels()
    def find_hotels(params):
        # Create the base query
        query = 'SELECT * FROM hotels'
        # Add filter clauses for each of the parameters
        if len(params) > 0:
            filters = ["{}=?".format(k) for k in params]
            query += " WHERE " + " AND ".join(filters)
        # Create the tuple of values
        t = tuple(params.values())
        # Open connection to DB
        conn = sqlite3.connect("hotels.db")
        # Create a cursor
        c = conn.cursor()
        # Execute the query
        c.execute(query, t)
        # Return the results
        q_result = c.fetchall()
        #Close connection
        conn.close()
        return q_result
    
    # Define respond()
    def respond(responses, message):
        # Extract the entities
        entities = interpreter.parse(message)["entities"]
        print("=====>entities = ",entities)
        # Initialize an empty params dictionary
        params = {}
        # Fill the dictionary with entities
        for ent in entities:
            params[ent["entity"]] = str(ent["value"])
        print("=====>params =",params)
        # Find hotels that match the dictionary
        results = find_hotels(params)
        # Get the names of the hotels and index of the response
        names = [r[0] for r in results]
        n = min(len(results),3)
        # Select the nth element of the responses array
        return responses[n].format(*names)
    
    
    print('--------------------------------Applying the function')
    # Create the dictionary of column names and values
    params = {'area': 'south', 
              'price': 'lo'}
    
    # Find the hotels that match the parameters
    print(find_hotels(params))
    
    
    print("****************************************************")
    topic = "7. Creating SQL from natural language"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('---------------------------Defining global varaibles')
    responses = ["I'm sorry :( I couldn't find anything like that",
                 '{} is a great hotel!',
                 '{} or {} would work!',
                 '{} is one option, but I know others too :)']
    
    print('-------------------------Test the respond() function')
    message = 'I want an expensive hotel in the south of town'
    print(message)
    print(respond(responses, message))
    
    
def Incremental_slot_filling_and_negation(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "8. Incremental slot filling and negation"; print("** %s" % topic)
    print("****************************************************")
    topic = "9. Refining your search"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------------------------Global variables')
    responses = ["I'm sorry :( I couldn't find anything like that",
                 '{} is a great hotel!',
                 '{} or {} would work!',
                 '{} is one option, but I know others too :)']

    print('------------------------------------Global functions')
    # Define find_hotels()
    def find_hotels(params):
        # Create the base query
        query = 'SELECT * FROM hotels'
        # Add filter clauses for each of the parameters
        if len(params) > 0:
            filters = ["{}=?".format(k) for k in params]
            query += " WHERE " + " AND ".join(filters)
        # Create the tuple of values
        t = tuple(params.values())
        # Open connection to DB
        conn = sqlite3.connect("hotels.db")
        # Create a cursor
        c = conn.cursor()
        # Execute the query
        c.execute(query, t)
        # Return the results
        q_result = c.fetchall()
        #Close connection
        conn.close()
        return q_result
    
    # Define a respond function, taking the message and existing params as input
    def respond(message, params):
        # Extract the entities
        entities = interpreter.parse(message)["entities"]
        # Fill the dictionary with entities
        for ent in entities:
            params[ent["entity"]] = str(ent["value"])
        
        # Find the hotels
        results = find_hotels(params)
        names = [r[0] for r in results]
        n = min(len(results), 3)
        # Return the appropriate response
        return responses[n].format(*names), params
    
    print('---------------------------Ready to the conversation')
    # Initialize params dictionary
    params = {}

    # Pass the messages to the bot
    messages = ["I want an expensive hotel", "in the north of town"]
    for message in messages:
        print("USER: {}".format(message))
        response, params = respond(message, params)
        print("BOT: {}".format(response))
        
    
def Basic_negation(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "10. Basic negation"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('--------------------------------------------Messages')
    tests = [("no I don't want to be in the south", {'south': False}),
             ('no it should be in the south', {'south': True}),
             ('no in the south not the north', {'north': False, 'south': True}),
             ('not north', {'north': False})]
    pprint(tests)
    
    print('------------------------------------Global functions')
    # Define negated_ents()
    def negated_ents(phrase):
        # Extract the entities using keyword matching
        ents = [e for e in ["south", "north"] if e in phrase]
        # Find the index of the final character of each entity
        ends = sorted([phrase.index(e) + len(e) for e in ents])
        # Initialise a list to store sentence chunks
        chunks = []
        # Take slices of the sentence up to and including each entitiy
        start = 0
        for end in ends:
            chunks.append(phrase[start:end])
            start = end
        result = {}
        # Iterate over the chunks and look for entities
        for chunk in chunks:
            for ent in ents:
                if ent in chunk:
                    # If the entity contains a negation, assign the key to be False
                    if "not" in chunk or "n't" in chunk:
                        result[ent] = False
                    else:
                        result[ent] = True
        return result  
    
    print('------------------------------------Finding negation')
    # Check that the entities are correctly assigned as True or False
    for test in tests:
        result = negated_ents(test[0])
        print(f'{test[0]}\n-->{result}-->{result == test[1]}')
    
    
def Filtering_with_excluded_slots(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "11. Filtering with excluded slots"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------------------------Global variables')
    responses = ["I'm sorry :( I couldn't find anything like that",
                 '{} is a great hotel!',
                 '{} or {} would work!',
                 '{} is one option, but I know others too :)']

    print('------------------------------------Global functions')
    def find_hotels(params, neg_params):
        query = 'SELECT * FROM hotels'
        if len(params)+len(neg_params) > 0:
            filters = ["{}=?".format(k) for k in params] + \
                      ["{}!=?".format(k) for k in neg_params] 
            query += " WHERE " + " and ".join(filters)
        t = tuple(params.values()) + tuple(neg_params.values())
        
        # open connection to DB
        conn = sqlite3.connect('hotels.db')
        # create a cursor
        c = conn.cursor()
        c.execute(query, t)
        # Return the results
        q_result = c.fetchall()
        #Close connection
        conn.close()
        return q_result
    
    # Define negated_ents()
    def negated_ents(phrase, ent_vals, ent_original_vals):
        # Find the index of the final character of each entity
        ends = sorted([phrase.index(e)+len(e) for e in ent_original_vals])
        # Initialise a list to store sentence chunks
        chunks = []
        # Take slices of the sentence up to and including each entitiy
        start = 0
        for end in ends:
            chunks.append(phrase[start:end])
            start = end
        result = {}
        # Iterate over the chunks and look for entities
        for chunk in chunks:
            for ent in ent_original_vals:
                if ent in chunk:
                    # If the entity contains a negation, assign the key to be False
                    result[ent_vals[ent_original_vals.index(ent)]] = ("not" in chunk or "n\'t" in chunk)
        return result  

    # Define the respond function
    def respond(message, params, neg_params):
        # Extract the entities
        entities = interpreter.parse(message)["entities"]
        ent_original_vals = [message[e['start']:e['end']] for e in entities]
        ent_vals = [e["value"] for e in entities]
        print("      ent_vals=",ent_vals)
        print("      ent_original_vals=",ent_original_vals)
        # Look for negated entities
        negated = negated_ents(message, ent_vals, ent_original_vals)
        print("      negated=",negated)
        #print("      entities=",entities)
        for ent in entities:
            if ent["value"] in negated and negated[ent["value"]]:
                neg_params[ent["entity"]] = str(ent["value"])
            else:
                params[ent["entity"]] = str(ent["value"])
        print("      params=",params)
        print("      neg_params=",neg_params)
        # Find the hotels
        results = find_hotels(params, neg_params)
        names = [r[0] for r in results]
        n = min(len(results),3)
        # Return the correct response
        return responses[n].format(*names), params, neg_params
        
    print('---------------------------Ready to the conversation')
    # Initialize params and neg_params
    params = {}
    neg_params = {}
    
    # Pass the messages to the bot
    for message in ["I want a cheap hotel", "but not in the north of town"]:
        print("USER: {}".format(message))
        response, params, neg_params = respond(message, params, neg_params)
        print("BOT : {}".format(response))
    
        
def main():
    print("****************************************************")
    print("** BEGIN                                          **")
    
    Virtual_Assistants_and_accessing_data()
    SQL_basics()
    SQL_statements_in_Python()
    Exploring_a_DB_with_natural_language()
    Creating_queries_from_parameters()
    Incremental_slot_filling_and_negation()
    Basic_negation()
    Filtering_with_excluded_slots()
    
    print("****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    pd.reset_option("display.max_columns")
    plt.style.use('default')
    np.set_printoptions(formatter={'float': None})