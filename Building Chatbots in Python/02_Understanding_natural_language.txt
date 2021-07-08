# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 2: Understanding natural language
    Here, you'll use machine learning to turn natural language into structured 
    data using spaCy, scikit-learn, and rasa NLU. You'll start with a refresher 
    on the theoretical foundations and then move onto building models using the 
    ATIS dataset, which contains thousands of sentences from real people 
    interacting with a flight booking system.
Source: https://learn.datacamp.com/courses/building-chatbots-in-python
"""
###############################################################################
## Importing libraries
###############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re
import time
import random

import spacy
#import en_core_web_lg #English word vector to work with Spacy. Documentation: https://spacy.io/usage/models#languages
import es_core_news_lg #Spanish word vector to work with Spacy. Documentation: https://spacy.io/usage/models#languages

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity #To evaluate the NLP model
from sklearn.metrics import accuracy_score
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import OrdinalEncoder

#from joblib import dump
#from joblib import load

##rasa_nlu.__version__==0.11.3
from rasa_nlu.converters import load_data #rasa_nlu.__version__==0.11.3 #To generate the json file: https://rodrigopivi.github.io/Chatito/ #Documentation: https://legacy-docs.rasa.com/docs/nlu/0.11.4/config/
from rasa_nlu.config import RasaNLUConfig #rasa_nlu.__version__==0.11.3 #Documentation: https://legacy-docs.rasa.com/docs/nlu/0.11.4/config/
from rasa_nlu.model import Trainer #rasa_nlu.__version__==0.11.3 #Documentation: https://legacy-docs.rasa.com/docs/nlu/0.11.4/config/

from pprint import pprint

###############################################################################
## Preparing the environment
###############################################################################
#Global variables
suptitle_param = dict(color='darkblue', fontsize=11)
title_param    = {'color': 'darkred', 'fontsize': 14, 'weight': 'bold'}
plot_param     = {'axes.labelsize': 6, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 
                  'legend.fontsize': 8, 'font.size': 6}
figsize        = (12.1, 5.9)
SEED           = 42
SIZE           = 10000

# Global configuration
sns.set()
pd.set_option("display.max_columns",24)
plt.rcParams.update(**plot_param)
np.set_printoptions(formatter={'float': '{:,.2f}'.format})
np.random.seed(SEED)

# Spacy configuration
#en_nlp = en_core_web_lg.load()
en_nlp = spacy.load('en_core_web_lg')
es_nlp = es_core_news_lg.load()

###############################################################################
## Reading the data
###############################################################################



###############################################################################
## Main part of the code
###############################################################################
def Understanding_Intents_and_Entities(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "1. Understanding Intents and Entities"; print("** %s" % topic)
    print("****************************************************")
    topic = "2. Intent classification with regex I"; print("** %s" % topic)
    print("****************************************************")
    topic = "3. Intent classification with regex II"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('---------------------------Defining global variables')
    keywords = {'greet'   : ['hello', 'hi', 'hey'], 
                'goodbye' : ['bye'  , 'farewell' ], 
                'thankyou': ['thank', 'thx'      ]}
    
    responses = {'default' : [':)', 'what can i say?', 'mmm...'],
                 'goodbye' : ['bye', 'goodbye for now', 'see you sun'],
                 'greet'   : ['hello you! :)', 'hi', 'nice to see you!', ':)'],
                 'thankyou': ['you are very welcome', 'your welcome', ':)']}

    bot_template = "BOT : {0}"
    user_template = "USER : {0}"
    
    print('------------------------------Dictionary of patterns')
    # Define a dictionary of patterns
    patterns = {}
    
    # Iterate over the keywords dictionary
    for intent, keys in keywords.items():
        # Create regular expressions and compile them into pattern objects
        patterns[intent] = re.compile('|'.join(keys))
    
    print('--------------------Define the match_intent function')
    # Define a function to find the intent of a message
    def match_intent(message):
        matched_intent = None
        for intent, pattern in patterns.items():
            # Check if the pattern occurs in the message 
            if re.search(pattern, message):
                matched_intent = intent
        return matched_intent
    
    print('-------------------------Define the respond function')
    # Define a respond function
    def respond(message):
        # Call the match_intent function
        intent = match_intent(message)
        # Fall back to the default response
        key = "default"
        if intent in responses:
            key = intent
        return random.choice(responses[key])
    
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
    # Send the messages
    send_message("hello!")
    send_message("bye byeee")
    send_message("thanks very much!")
    
    
    
def Entity_extraction_with_regex(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "4. Entity extraction with regex"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('---------------------------Defining global variables')
    bot_template = "BOT : {0}"
    user_template = "USER : {0}"
    
    print('-----------------------Define the find_name function')
    # Define find_name()
    def find_name(message):
        name = None
        # Create a pattern for checking if the keywords occur
        name_keyword = re.compile('name|call')
        # Create a pattern for finding capitalized words
        name_pattern = re.compile(r'\s[A-Z]{1}[a-z]*\b')
        if name_keyword.search(message):
            # Get the matching words in the string
            name_words = name_pattern.findall(message)
            if len(name_words) > 0:
                # Return the name if the keywords are present
                name = ' '.join(np.char.strip(name_words))
        return name
    
    print('-------------------------Define the respond function')
    # Define respond()
    def respond(message):
        # Find the name
        name = find_name(message)
        if name is None:
            return "Hi there!"
        else:
            return "Hello, {0}!".format(name)
    
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
    # Send messages
    send_message("My name is David Copperfield")
    send_message("Call me Ishmael")
    send_message("People call me Cassandra")
    send_message('Hi again!')
    
    
    
def Word_vectors(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "5. Word vectors"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('-----------A demostration of Spacy Module in English')
    lang = 'English'
    # Word vectors in spaCy
    print(f'In spacy the "{lang}" lang has a', en_nlp.vocab.vectors_length, 'dimesional word vectors.')
    
    sentence = 'hello can you help me?' 
    print(f'\nAccessing the three first word vectors of the sentence: \n"{sentence}"...\n')
    doc = en_nlp(sentence)
    for token in doc:
        print("{} : {}".format(token, token.vector[:3]))
        
    
    string1 = 'cat'
    string2 = 'can'
    string3 = 'dog'
    
    doc = en_nlp(string1)
    print(f'\nSimilarity between "{string1}" and {string2}: ', doc.similarity(en_nlp(string2)))
    print(f'Similarity between "{string1}" and {string3}: ', doc.similarity(en_nlp(string3)), '\n')
    
    print('-----------A demostration of Spacy Module in Spanish')
    lang = 'Spanish'
    # Word vectors in spaCy
    print(f'In spacy the "{lang}" lang has a', es_nlp.vocab.vectors_length, 'dimesional word vectors.')
    
    sentence = 'Hola, ¿puedes ayudarme?' 
    print(f'\nAccessing the three first word vectors of the sentence: \n"{sentence}"...\n')
    doc = es_nlp(sentence)
    for token in doc:
        print("{} : {}".format(token, token.vector[:3]))
        
    
    string1 = 'perro'
    string2 = 'pedro'
    string3 = 'gato'
    
    doc = es_nlp(string1)
    print(f'\nSimilarity between "{string1}" and {string2}: ', doc.similarity(es_nlp(string2)))
    print(f'Similarity between "{string1}" and {string3}: ', doc.similarity(es_nlp(string3)), '\n')
    
    
    
    
def word_vectors_with_spaCy(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "6. word vectors with spaCy"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------------------------Reading the data')
    df = pd.read_csv('atis_intents.csv', header=None, names=['intent','message'], encoding='utf_8')
    sentences = df.message.tolist()
    
    print('--------------------------------------Applying Spacy')
    # Calculate the length of sentences
    n_sentences = len(sentences)
    
    # Calculate the dimensionality of nlp
    embedding_dim = en_nlp.vocab.vectors_length
    
    """
    # Initialize the array with zeros: X
    X = np.zeros((n_sentences, embedding_dim))
    
    # Iterate over the sentences
    for idx, sentence in enumerate(sentences):
        # Pass each each sentence to the nlp object to create a document
        doc = en_nlp(sentence)
        # Save the document's .vector attribute to the corresponding row in X
        X[idx, :] = doc.vector
    print(X)
    print(f'Size: {n_sentences} X {embedding_dim}.\n')
    """
    
    X = np.array([en_nlp(sentence).vector for sentence in sentences])
    
    print(X)
    print(f'Size: {n_sentences} X {embedding_dim}.\n')
    
    
    
def Intents_and_classification(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "7. Intents and classification"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------------------------Reading the data')
    df = pd.read_csv('atis_intents.csv', header=None, names=['intent','message'], encoding='utf_8')
    print(df.head())
    
    print('Labels found:')
    print(df.intent.value_counts())
    
    #df = df.append(df[df.intent.isin(['atis_ground_service#atis_ground_fare', 'atis_aircraft#atis_flight#atis_flight_no', 'atis_cheapest','atis_airfare#atis_flight_time', 'atis_airfare#atis_flight_time'])])
    df = df.append(df[df.intent.isin(df.intent.value_counts()[df.intent.value_counts()==1].index)])
    print('Labels found after rectification:')
    print(df.intent.value_counts())
    
    print('-------------------Preparint the train and test data')
    sentences = df.message
    labels = df.intent
    
    # Create training and test sets
    sentences_train, sentences_test, labels_train, labels_test = train_test_split(sentences, labels, test_size=.25, stratify=labels, random_state=seed)
    
    print('---------------------------Nearest neighbor approach')
    X_train = np.array([en_nlp(sentence).vector for sentence in sentences_train])
    
    test_message = "i would like to find a flight from charlotte to las vegas that makes a stop in st. louis"
    test_X = en_nlp(test_message).vector
    scores = [cosine_similarity([vector], [test_X])[0][0] for vector in X_train]
    best_approach = labels_train[np.argmax(scores)]
    
    print(f'For sentence: \n"{test_message}" \nBest approach is: "{best_approach}"')
    
    X_test = np.array([en_nlp(sentence).vector for sentence in sentences_test])
    
    # For speed we stop this long time spend functional code
    """
    y_pred = [[labels_train[np.argmax(cosine_similarity([vector], [sent])[0][0])] for vector in X_train][0] for sent in X_test]
    model_score = accuracy_score(labels_test, y_pred)
    print(f'Accuracy of the model: {model_score:,.2%}')
    print(f'Distinct predicted label: {np.unique(y_pred)}\n')
    """
    
    print('--------------------Support vector machines approach')
    clf = SVC()
    clf.fit(X_train, labels_train)
    
    """
    print('------------------------------------Saving the model')
    # To save and reuse the model
    # https://scikit-learn.org/stable/modules/model_persistence.html#model-persistence
    dump(clf, 'intent_classifier_for_flights.joblib')
    
    print('--------------------------------------Load the model')
    clf = load('intent_classifier_for_flights.joblib')
    """
    print('--------------------------------Evaluating the model')
    y_pred = clf.predict(X_test)
    model_score = accuracy_score(labels_test, y_pred)
    
    print(f'Accuracy of the model: {model_score:,.2%}')
    print(f'Distinct predicted label: {np.unique(y_pred)}\n')
    
    
    
    print("****************************************************")
    topic = "8. Intent classification with sklearn"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('--------------------------------------OrdinalEncoder')
    model_labels = sorted(np.unique(labels))
    encoder = OrdinalEncoder(categories = [model_labels])
    y_train = encoder.fit_transform(labels_train.values.reshape(-1,1)).flatten()
    y_test = encoder.fit_transform(labels_test.values.reshape(-1,1)).flatten()
    
    print('--------------------Support vector machines approach')
    clf = SVC(C=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    model_score = accuracy_score(y_test, y_pred)
    
    print(f'Accuracy of the model: {model_score:,.2%}')
    print(f'Distinct predicted label: {np.unique(y_pred)}\n')
    
    """
    # Calculate the confusion matrix: normalized cm
    labels_pred = encoder.inverse_transform(y_pred.reshape(-1,1)).flatten()
    cm = confusion_matrix(labels_test, labels_pred, labels=model_labels, normalize='true')
    print(f'Confusion Matrix: \n{cm}\n')
    """
    
    # Plotting Confusion matrix
    fig, ax = plt.subplots()
    plot_confusion_matrix(clf, X_test, y_test, display_labels=model_labels, 
                          cmap=plt.cm.Blues, normalize='true', values_format='.1%', 
                          ax=ax)
    ax.tick_params(axis='x', rotation=90)
    ax.set_title(f'Support vector machines approach\nAccuracy score: {model_score:.1%}', **title_param)
    ax.grid(False)
    plt.subplots_adjust(left=None, bottom=.3, right=None, top=.85, wspace=None, hspace=None)
    fig.suptitle(topic, **suptitle_param)
    plt.show()
    
    
def Entity_extraction(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "9. Entity extraction"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------Pre-built Named Entity Recognition')
    sentence = "My friend Mary has worked at Google since 2009" 
    doc = en_nlp(sentence)
    
    print(f'My sentence: \n{sentence}\n\nEntities found:\n')
    for ent in doc.ents:
        print(ent.text, ent.label_)
        
    print('-----------------------------------------------Roles')
    sentence1 = 'I want a flight from TelAviv to Bucharest'
    sentence2 = 'Show me flights to Bucharest from TelAviv '
    
    pattern_1 = re.compile(r'.* from (.*) to (.*)') 
    pattern_2 = re.compile(r'.* to (.*) from (.*)')
    
    print(f'In sentence: \n{sentence1}')
    print(f'With {pattern_1} pattern, we found: ', re.findall(pattern_1, sentence1))
    print(f'With {pattern_2} pattern, we found: ', re.findall(pattern_2, sentence1))
    
    print(f'\nIn sentence: \n{sentence2}')
    print(f'With {pattern_1} pattern, we found: ', re.findall(pattern_1, sentence2))
    print(f'With {pattern_2} pattern, we found: ', re.findall(pattern_2, sentence2))
    
    
    print('----------------------------------Dependency parsing')
    doc1 = en_nlp(sentence1)
    TelAviv, Bucharest = doc1[5], doc1[7]
    
    print(f'In sentence: \n"{sentence1}"')
    print(f'Dependency found for {TelAviv}: ', list(TelAviv.ancestors))
    print(f'Dependency found for {Bucharest}: ', list(Bucharest.ancestors))
    
    
    doc2 = en_nlp(sentence2)
    TelAviv, Bucharest = doc2[6], doc2[4]
    
    print(f'In sentence: \n"{sentence2}"')
    print(f'Dependency found for "{TelAviv}": ', list(TelAviv.ancestors))
    print(f'Dependency found for "{Bucharest}": ', list(Bucharest.ancestors))
    
    print('----------------------------------Shopping example')
    sentence = "let's see that jacket in red and some blue jeans"
    doc = en_nlp(sentence)
    
    items = [doc[4], doc[10]] # [jacket, jeans]
    colors = [doc[6], doc[9]] # [red, blue]
    
    print(f'In sentence: \n"{sentence}"\n')
    for color in colors:
        print(f'Dependency found for "{color}": ', list(color.ancestors))
    
    print()
    for color in colors:
        for token in color.ancestors:
            if token in items:
                print('Color "{}" belongs to item "{}"'.format(color, token))
                break;
    
def Using_spaCys_entity_recognizer(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "10. Using spaCy's entity recognizer"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------------------------Finding entities')
    # Define included_entities
    include_entities = ['DATE', 'ORG', 'PERSON']
    
    # Define extract_all_entities()
    def extract_all_entities(message, include_entities=include_entities):
        # Create a dict to hold the entities
        ents = dict.fromkeys(include_entities, [])
        
        # Create a spacy document
        doc = en_nlp(message)
        for ent in doc.ents:
            if ent.label_ in include_entities:
                # Save all interesting entities
                ents.update({ent.label_: ents[ent.label_]+[ent.text]})
        return ents
    
    # Define extract_entities()
    def extract_entities(message, include_entities=include_entities):
        # Create a dict to hold the entities
        ents = dict.fromkeys(include_entities)
        
        # Create a spacy document
        doc = en_nlp(message)
        for ent in doc.ents:
            if ent.label_ in include_entities:
                # Save all interesting entities
                ents[ent.label_] = ent.text
        return ents
    
    sentences = ['Friends called Mary who have worked at Google since 2010',
                 'We worked at Dell from 2000 to 2010',
                 'People who graduated from MIT in 1999']
    for sent in sentences:
        print(f'In sentence: \n{sent} \nWe found:')
        print('All entities: ', extract_all_entities(sent))
        print('Just one of each type: ', extract_entities(sent))
    
    
    
def Assigning_roles_using_spaCys_parser(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "11. Assigning roles using spaCy's parser"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('--------------------------Assigning roles in English')
    colors = ['black', 'red', 'blue']
    items  = ['shoes', 'handback', 'jacket', 'jeans']
    
    def entity_type(word):
        _type = None
        if word.text in colors:
            _type = "color"
        elif word.text in items:
            _type = "item"
        return _type
    
    # Iterate over parents in parse tree until an item entity is found
    def find_parent_item(word):
        # Iterate over the word's ancestors
        for parent in word.ancestors:
            # Check for an "item" entity
            if entity_type(parent) == "item":
                return parent.text
        return None
    
    # For all color entities, find their parent item
    def assign_colors(doc):
        # Iterate over the document
        for word in doc:
            # Check for "color" entities
            if entity_type(word) == "color":
                # Find the parent
                item =  find_parent_item(word)
                print("Item: {0} has color {1}".format(item, word))
    
    # Create the document
    sentence = "let's see that jacket in red and some blue jeans"
    doc = en_nlp(sentence)
    print(f'In sentence: \n"{sentence}" \nWe found:')
        
    # Assign the colors
    assign_colors(doc) 
    
    
    print('---------------------------------Trying with Spanish')
    colors = ['negro', 'rojo', 'azul']
    items  = ['zapato', 'cartera', 'chaqueta', 'pantalón']
    def es_entity_type(word):
        _type = None
        if word.lemma_ in colors:
            _type = "color"
        elif word.lemma_ in items:
            _type = "item"
        return _type
    
    # Iterate over parents in parse tree until an item entity is found
    def es_find_parent_item(word):
        # Iterate over the word's ancestors
        for parent in word.ancestors:
            # Check for an "item" entity
            if es_entity_type(parent) == "item":
                return parent.text
        return None
    
    # For all color entities, find their parent item
    def es_assign_colors(doc):
        # Iterate over the document
        for word in doc:
            # Check for "color" entities
            if es_entity_type(word) == "color":
                # Find the parent
                item =  es_find_parent_item(word)
                print("Item: {0} color {1}".format(item, word))
    
    # Create the document
    sentence = "Veamos esa chaqueta en roja y unos pantalones negro"
    doc = es_nlp(sentence)
    print(f'En la oración: \n"{sentence}" \nEncontramos:')
        
    # Assign the colors
    es_assign_colors(doc) 
    
    
def Rasa_NLU(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "13. Rasa NLU"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------------------Create args dictionary')
    args = {'pipeline':'spacy_sklearn'}
    
    print('------------------Create a configuration and trainer')
    config = RasaNLUConfig(cmdline_args=args)
    trainer = Trainer(config)
    
    print('------------------------------Load the training data')
    training_data = load_data("data/training_data.json")
    
    print('---------Create an interpreter   by training the model')
    interpreter = trainer.train(training_data)
    
    print('--------------------------------Test the interpreter')
    pprint(interpreter.parse("I'm looking for a Mexican restaurant in the North of town"))
        
        
    
def Data_efficient_entity_recognition(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "14. Data-efficient entity recognition"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------------------Create args dictionary')
    pipeline = ["nlp_spacy", "tokenizer_spacy", "ner_crf"]
    args = {'pipeline': pipeline, 'language': 'en'}
    
    print('------------------Create a configuration and trainer')
    config = RasaNLUConfig(cmdline_args=args)
    trainer = Trainer(config)
    
    print('------------------------------Load the training data')
    training_data = load_data("data/training_data.json")
    
    print('---------Create an interpreter   by training the model')
    interpreter = trainer.train(training_data)
    
    print('---------------------------------Parse some messages')
    pprint(interpreter.parse("show me Chinese food in the centre of town"))
    pprint(interpreter.parse("I want an Indian restaurant in the west"))
    pprint(interpreter.parse("are there any good pizza places in the center?"))
    
    
            
def main():
    print("****************************************************")
    print("** BEGIN                                          **")
        
    Understanding_Intents_and_Entities()
    Entity_extraction_with_regex()
    Word_vectors()
    # For speed we stop the call of the function.
    #word_vectors_with_spaCy()
    
    Intents_and_classification()
    Entity_extraction()
    Using_spaCys_entity_recognizer()
    Assigning_roles_using_spaCys_parser()
    
    Rasa_NLU()
    Data_efficient_entity_recognition()

    
    print("****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    pd.reset_option("display.max_columns")
    plt.style.use('default')
    np.set_printoptions(formatter={'float': None})