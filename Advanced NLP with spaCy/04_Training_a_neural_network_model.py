# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 4: Training a neural network model
    In this chapter, you'll learn how to update spaCy's statistical models to 
    customize them for your use case – for example, to predict a new entity 
    type in online comments. You'll write your own training loop from scratch, 
    and understand the basics of how training works, along with tips and tricks 
    that can make your custom NLP projects more successful.
Source: https://learn.datacamp.com/courses/advanced-nlp-with-spacy

More documentation:
    For training:
        https://spacy.io/usage/training#step-by-step-ner-new
        https://towardsdatascience.com/extend-named-entity-recogniser-ner-to-label-new-entities-with-spacy-339ee5979044
    annotation tool :
        http://brat.nlplab.org/
        https://prodi.gy/
"""
###############################################################################
## Importing libraries
###############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import spacy
from spacy.lang.en import English
from spacy.matcher import Matcher #https://spacy.io/usage/rule-based-matching#adding-patterns-attributes

import random


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
np.set_printoptions(threshold=8) #Return to default value.

###############################################################################
## Reading the data
###############################################################################



###############################################################################
## Main part of the code
###############################################################################
def Creating_training_data(size=SIZE, seed=SEED):
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Create the nlp object
    nlp = English()
    
    print("****************************************************")
    topic = "3. Creating training data"; print("** %s" % topic)
    print("****************************************************")
    
    print('----------------------------------Gathering the data')
    # Read the dataset
    TEXTS = ['How to preorder the iPhone X',
             'iPhone X is coming',
             'Should I pay $1,000 for the iPhone X?',
             'The iPhone 8 reviews are here',
             'Your iPhone goes up to 11 today',
             'I need a new phone! Any tips?']
    
    print('---------------------------------Setting the pattern')
    # Initialize the matcher with the shared vocab
    matcher = Matcher(nlp.vocab)
    
    # Two tokens whose lowercase forms match 'iphone' and 'x'
    pattern1 = [{'LOWER': 'iphone'}, {'LOWER': 'x'}]
    
    # Token whose lowercase form matches 'iphone' and an optional digit
    pattern2 = [{'LOWER': 'iphone'}, {'IS_DIGIT': True, 'OP': '+'}]
    
    # Add patterns to the matcher
    matcher.add('GADGET', None, pattern1, pattern2)
    
    print('---------------------------Setting the training data')
    TRAINING_DATA = []
    
    # Create a Doc object for each text in TEXTS
    for doc in nlp.pipe(TEXTS):
        # Match on the doc and create a list of matched spans
        spans = [doc[start:end] for match_id, start, end in matcher(doc)]
        
        # Get (start character, end character, label) tuples of matches
        entities = [(span.start_char, span.end_char, 'GADGET') for span in spans]
    
        # Format the matches as a (doc.text, entities) tuple
        training_example = (doc.text, {'entities': entities})
        
        # Append the example to the training data
        TRAINING_DATA.append(training_example)
    
    print(*TRAINING_DATA, sep='\n')    
    
    return TRAINING_DATA
    

def Training_with_an_empty_model(TRAINING_DATA, size=SIZE, seed=SEED):
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print("****************************************************")
    topic = "*** TRAINING AN EMPTY MODEL                       **"; print("** %s" % topic)
    print("****************************************************")
    topic = "6. Setting up the pipeline"; print("** %s" % topic)
    print("****************************************************")
    topic = "7. Building a training loop"; print("** %s" % topic)
    print("****************************************************")
    topic = "8. Exploring the model"; print("** %s" % topic)
    print("****************************************************")
    
    print('--------------------------Initializing a blank model')
    # Create a blank 'en' model
    NLP_NEW_LABEL = spacy.blank('en')
    
    print('--------------------------------Setting the pipeline')
    # Create a new entity recognizer and add it to the pipeline
    ner = NLP_NEW_LABEL.create_pipe('ner')
    NLP_NEW_LABEL.add_pipe(ner)
    
    # All the existing entities recognised by the model
    prev_ents = ner.move_names 
    print('[Existing Entities] = ', prev_ents)
    
    print('-----------------------------------Adding new labels')
    # Add the label 'GADGET' to the entity recognizer
    ner.add_label('GADGET')
    new_ents = ner.move_names
    print('[New Entities] = ', list(set(new_ents) - set(prev_ents)))
    
    
    print('------------------------------Beginning the training')
    #print(*TRAINING_DATA, sep='\n')
    
    # Start the training
    optimizer = NLP_NEW_LABEL.begin_training()
    
    # Loop for 10 iterations
    for itn in range(10):
        # Shuffle the training data
        random.shuffle(TRAINING_DATA)
        losses = {} #To show the progress...
    
        # Batch the examples and iterate over them
        for batch in spacy.util.minibatch(TRAINING_DATA, size=2):
            #texts = [text for text, entities in batch]
            #annotations = [entities for text, entities in batch]
            texts = list(zip(*TRAINING_DATA))[0]
            annotations = list(zip(*TRAINING_DATA))[1]
            
            # Update the model
            NLP_NEW_LABEL.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
            print(losses)
    
    print('--------------------------------------Test the model')
    TEST_DATA = ['Apple is slowing down the iPhone 8 and iPhone X - how to stop it',
                 "I finally understand what the iPhone X 'notch' is for",
                 'Everything you need to know about the Samsung Galaxy S9',
                 'Looking to compare iPad models? Here’s how the 2018 lineup stacks up',
                 'The iPhone 8 and iPhone 8 Plus are smartphones designed, developed, and marketed by Apple',
                 'what is the cheapest ipad, especially ipad pro???',
                 'Samsung Galaxy is a series of mobile computing devices designed, manufactured and marketed by Samsung Electronics']
    
    # Process each text in TEST_DATA
    for i, doc in enumerate(NLP_NEW_LABEL.pipe(TEST_DATA), start=1):
        # Print the document text and entitites
        print(f'{i} - {doc.text}')
        print(*[(ent.text, ent.label_) for ent in doc.ents], sep='\n')

    
def Training_and_updating_a_model(TRAINING_DATA, size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "*** TRAINING AND UPDTING A MODEL                  **"; print("** %s" % topic)
    print("****************************************************")
    
    print('-----------------------------------Loading the model')
    # Create a blank 'en' model
    NLP_TO_UPDATE = spacy.load('en_core_web_sm')
    
    print('--------------------------------Setting the pipeline')
    # get names of other pipes to disable them during training
    pipe_exceptions = ['ner']
    other_pipes = [pipe for pipe in NLP_TO_UPDATE.pipe_names if pipe not in pipe_exceptions]
    
    ner = NLP_TO_UPDATE.get_pipe("ner")
    
    # All the existing entities recognised by the model
    prev_ents = ner.move_names 
    print('[Existing Entities] = ', prev_ents)
    
    print('-----------------------------------Adding new labels')
    # Add the label 'GADGET' to the entity recognizer
    ner.add_label('GADGET')
    new_ents = ner.move_names
    print('[New Entities] = ', list(set(new_ents) - set(prev_ents)))
    
    print('------------------------------Beginning the training')
    with NLP_TO_UPDATE.disable_pipes(*other_pipes):
        #print(*TRAINING_DATA, sep='\n')
        
        # Start the training
        optimizer = NLP_TO_UPDATE.begin_training()
        
        # Loop for 10 iterations
        for itn in range(10):
            # Shuffle the training data
            random.shuffle(TRAINING_DATA)
            losses = {} #To show the progress...
            
            # Batch the examples and iterate over them
            for batch in spacy.util.minibatch(TRAINING_DATA, size=2):
                texts = list(zip(*TRAINING_DATA))[0]
                annotations = list(zip(*TRAINING_DATA))[1]
            
                # Update the model
                NLP_TO_UPDATE.update(texts, annotations, sgd=optimizer, losses=losses)
                print(losses)
    
    print('--------------------------------------Test the model')
    TEST_DATA = ['Apple is slowing down the iPhone 8 and iPhone X - how to stop it',
                 "I finally understand what the iPhone X 'notch' is for",
                 'Everything you need to know about the Samsung Galaxy S9',
                 'Looking to compare iPad models? Here’s how the 2018 lineup stacks up',
                 'The iPhone 8 and iPhone 8 Plus are smartphones designed, developed, and marketed by Apple',
                 'what is the cheapest ipad, especially ipad pro???',
                 'Samsung Galaxy is a series of mobile computing devices designed, manufactured and marketed by Samsung Electronics']
    
    # Process each text in TEST_DATA
    for i, doc in enumerate(NLP_TO_UPDATE.pipe(TEST_DATA), start=1):
        # Print the document text and entitites
        print(f'{i} - {doc.text}')
        print(*[(ent.text, ent.label_) for ent in doc.ents], sep='\n')
    
    print('------------Comparing with the default English model')
    nlp = spacy.load('en_core_web_sm')
    TEST_DATA = ['Apple is slowing down the iPhone 8 and iPhone X - how to stop it',
                 "I finally understand what the iPhone X 'notch' is for",
                 'Everything you need to know about the Samsung Galaxy S9',
                 'Looking to compare iPad models? Here’s how the 2018 lineup stacks up',
                 'The iPhone 8 and iPhone 8 Plus are smartphones designed, developed, and marketed by Apple',
                 'what is the cheapest ipad, especially ipad pro???',
                 'Samsung Galaxy is a series of mobile computing devices designed, manufactured and marketed by Samsung Electronics']
    
    # Process each text in TEST_DATA
    for i, doc in enumerate(nlp.pipe(TEST_DATA), start=1):
        # Print the document text and entitites
        print(f'{i} - {doc.text}')
        print(*[(ent.text, ent.label_) for ent in doc.ents], sep='\n')
    
    
    
def main():
    print("****************************************************")
    print("** BEGIN                                          **")
    
    training_data = Creating_training_data()
    Training_with_an_empty_model(training_data)
    Training_and_updating_a_model(training_data)
    
    print("****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    pd.reset_option("display.max_columns")
    plt.style.use('default')
    np.set_printoptions(formatter={'float': None})