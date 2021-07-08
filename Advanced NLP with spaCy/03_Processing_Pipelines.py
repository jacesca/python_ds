# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 3: Processing Pipelines
    This chapter will show you to everything you need to know about spaCy's 
    processing pipeline. You'll learn what goes on under the hood when you 
    process a text, how to write your own components and add them to the 
    pipeline, and how to use custom attributes to add your own meta data to 
    the documents, spans and tokens.
Source: https://learn.datacamp.com/courses/advanced-nlp-with-spacy
"""
###############################################################################
## Importing libraries
###############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import time

from pprint import pprint

import spacy
from spacy.lang.en import English
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from spacy.tokens import Token
from spacy.tokens import Doc

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
## NLP configuration
###############################################################################
#model = 'en_core_web_lg'
#nlp = spacy.load(model)
#print(nlp.path)

###############################################################################
## Main part of the code
###############################################################################
def Processing_pipelines(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "1. Processing pipelines"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    #print('----------------------------"en_core_web_lg" dataset')
    #print('Pipeline component names:', nlp.pipe_names)
    #print('Pipeline component details')
    #pprint(nlp.pipeline)
    
    print('-------------------------------------nlp = English()')
    nlp_sm = English()
    print('Pipeline component names:', nlp_sm.pipe_names)
    print('Pipeline component details')
    pprint(nlp_sm.pipeline)
    
    
    
def What_happens_when_you_call_nlp(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "2. What happens when you call nlp?"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------Preparing NLP object and functions')
    ###########################################################################
    ## According to the documentation
    ###########################################################################
    ##lang = "en"
    ##pipeline = ["tagger", "parser", "ner"]
    ##model = 'en_core_web_sm'
    ##data_path = 'C:/Anaconda3/envs/datascience/lib/site-packages/en_core_web_sm/en_core_web_sm-2.3.1'
    ##
    ##cls = spacy.util.get_lang_class(lang)           # 1. Get Language instance, e.g. English()
    ##orig_nlp = cls()                                # 2. Initialize it
    ##for name in pipeline:
    ##    component = orig_nlp.create_pipe(name)      # 3. Create the pipeline components
    ##    orig_nlp.add_pipe(component)                # 4. Add the component to the pipeline
    ##orig_nlp.from_disk(data_path)                   # 5. Load in the binary data
    ###########################################################################
    
    pipeline = ["tagger", "parser", "ner"]
    model = 'en_core_web_sm'
    data_path = 'C:/Anaconda3/envs/datascience/lib/site-packages/en_core_web_sm/en_core_web_sm-2.3.1'
    orig_nlp = English()
    
    for name in pipeline:
        component = orig_nlp.create_pipe(name)      # 3. Create the pipeline components
        orig_nlp.add_pipe(component)                # 4. Add the component to the pipeline
    orig_nlp.from_disk(data_path)                   # 5. Load in the binary data
    
    print('-----------------------------------------Trazing NLP')
    def nlp(text):
        print("Using model '{}' (language '{}' and pipeline {})".format(model, orig_nlp.lang, orig_nlp.pipe_names))
        print("Tokenizing text: {}".format(text))
        doc = orig_nlp.make_doc(text)
        for name, proc in orig_nlp.pipeline:
           print("Calling pipeline component '{}' on Doc".format(name))
           doc = proc(doc)
        print('Returning processed Doc')
        return doc
    
    text = "This is a sentence"
    print(f'Text: "{text}"\n')
    doc = nlp(text)
    print('\nDoc                   :', [token.text for token in doc])
    
    print('tagger --> tags       :', [token.tag_ for token in doc])
    print('parser --> dep        :', [token.dep_ for token in doc])
    print('           head       :', [token.head.text for token in doc])
    print('           sents      :', [sent for sent in doc.sents])
    print('           noun_chunks:', [nchunk for nchunk in doc.noun_chunks])
    print('ner    --> ents       :', [(ent.text, ent.label_) for ent in doc.ents])
    print('           ent_iob    :', [token.ent_iob for token in doc])
    print('           ent_type   :', [token.ent_type for token in doc])
    
    
    
def Inspecting_the_pipeline(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "3. Inspecting the pipeline"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Create the nlp object
    nlp_sm = spacy.load('en_core_web_sm')
    
    print('----------------------------"en_core_web_sm" dataset')
    print('Pipeline component names:', nlp_sm.pipe_names)
    print('Pipeline component details')
    pprint(nlp_sm.pipeline)
    
    
    
def Custom_pipeline_components(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "4. Custom pipeline components"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Create the nlp object
    nlp_sm = spacy.load('en_core_web_sm')
    
    
    print('---------------------Example: a simple component (1)')
    def custom_component(doc):
        # Print the doc's length
        print('Doc length:', len(doc))
        # Return the doc object
        return doc
    
    # Add the component first in the pipeline
    nlp_sm.add_pipe(custom_component, first=True)

    # Print the pipeline component names
    print('Pipeline:', nlp_sm.pipe_names)
    
    print('---------------------------------------------Explore')
    # Process a text
    doc = nlp_sm("Hello world!")
    print('Doc processed:', doc.text)
    
    
    
def Simple_components(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "6. Simple components"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Create the nlp object
    nlp_sm = spacy.load('en_core_web_sm')
    
    print('-------------------------Define the custom component')
    # Define the custom component
    def length_component(doc):
        # Get the doc's length
        doc_length = len(doc)
        print("This document is {} tokens long.".format(doc_length))
        # Return the doc
        return doc

    print('------------------------------Add it to the pipeline')
    # Add the component first in the pipeline and print the pipe names
    nlp_sm.add_pipe(length_component, first=True)
    print(nlp_sm.pipe_names)
    
    print('---------------------------------------------Test it')
    # Process a text
    doc = nlp_sm("This is a sentence.")
    print('Doc processed:', doc.text)
    
    
    
def Complex_components(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "7. Complex components"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Create the nlp object
    nlp_sm = spacy.load('en_core_web_sm')
    
    # Initialize the PhraseMatcher
    matcher = PhraseMatcher(nlp_sm.vocab)
    
    # Create pattern Doc objects and add them to the matcher
    ANIMALS = ['Golden Retriever', 'cat', 'turtle', 'Rattus norvegicus']
    patterns = list(nlp_sm.pipe(ANIMALS))
    matcher.add('ANIMAL', None, *patterns)
    
    print('-------------------------Define the custom component')
    # Define the custom component
    def animal_component(doc):
        # Apply the matcher to the doc
        matches = matcher(doc)
        # Create a Span for each match and assign the label 'ANIMAL'
        spans = [Span(doc, start, end, label='ANIMAL')
                 for match_id, start, end in matches]
        # Overwrite the doc.ents with the matched spans
        doc.ents = spans
        return doc
    
    print('------------------------------Add it to the pipeline')
    # Add the component to the pipeline after the 'ner' component 
    nlp_sm.add_pipe(animal_component, after='ner')
    print(nlp_sm.pipe_names)
    
    print('---------------------------------------------Test it')
    # Process the text and print the text and label for the doc.ents
    doc = nlp_sm("I have a cat and a Golden Retriever")
    print('Doc processed:', doc.text)
    print('Animal entities found:')
    print([(ent.text, ent.label_) for ent in doc.ents])

    
    
def Extension_attributes(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "8. Extension attributes"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Create the nlp object
    nlp_sm = spacy.load('en_core_web_sm')
    
    print('--------------------------------Attribute extensions')
    # Set extension on the Token with default value
    Token.set_extension('is_color', default=False, force=True)
    doc = nlp_sm("The sky is blue.")
    
    df = pd.DataFrame({'token': [token for token in doc],
                       '.is_color (before)': [token._.is_color for token in doc]})
    
    # Overwrite extension attribute value
    doc[3]._.is_color = True
    df['.is_color (after)'] = [token._.is_color for token in doc]
    
    # Print the result
    print('Doc:', doc.text)
    print(df)
    
    print('-----------------------------Property extensions (1)')
    # Define getter function
    def get_is_color(token):
        colors = ['red', 'yellow', 'blue']
        return token.text in colors
    # Set extension on the Token with getter
    Token.set_extension('is_color', getter=get_is_color, force=True)
    
    doc = nlp_sm("The sky is blue and the tree is red.")
    
    df = pd.DataFrame({'token': [token for token in doc],
                       '.is_color': [token._.is_color for token in doc]})
    # Print the result
    print('Doc:', doc.text)
    print(df)
    
    print('-----------------------------Property extensions (2)')
    # Define getter function
    def get_has_color(span):
        colors = ['red', 'yellow', 'blue']
        return any(token.text in colors for token in span)

    # Set extension on the Span with getter
    Span.set_extension('has_color', getter=get_has_color, force=True)
    doc = nlp_sm("The sky is blue.")
    
    spans = [(1,4),(0,2)]
    df = pd.DataFrame({'Span': [doc[start:end].text for start, end in spans],
                       'Has color?':[doc[start:end]._. has_color for start, end in spans]})
    # Print the result
    print('Doc:', doc.text)
    print(df)
    
    print('-----------------------------------Method extensions')
    # Define method with arguments
    def has_token(doc, token_text):
        return token_text in [token.text for token in doc]
    
    # Set extension on the Doc with method
    Doc.set_extension('has_token', method=has_token, force=True)
    doc = nlp_sm("The sky is blue.")
    
    clues = ['blue', 'cloud']
    df = pd.DataFrame({'Token': [clue for clue in clues],
                       'Present in doc?':[doc._. has_token(clue) for clue in clues]})
    # Print the result
    print('Doc:', doc.text)
    print(df)
    
        
    
def Setting_extension_attributes_1(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "9. Setting extension attributes (1)"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Create the nlp object
    nlp_sm = spacy.load('en_core_web_sm')
    
    print('------------------------------------------is_country')
    # Register the Token extension attribute 'is_country' with the default value False
    Token.set_extension('is_country', default=False, force=True)

    # Process the text and set the is_country attribute to True for the token "Spain"
    doc = nlp_sm("I live in Spain.")
    doc[3]._.is_country = True
    
    # Print the token text and the is_country attribute for all tokens
    df = pd.DataFrame({'token': [token for token in doc],
                       'is_country': [token._.is_country for token in doc]})
    print('Doc:', doc.text)
    print(df)
    
    
    print('--------------------------------------------reversed')
    # Define the getter function that takes a token and returns its reversed text
    def get_reversed(token):
        return token.text[::-1]
      
    # Register the Token property extension 'reversed' with the getter get_reversed
    Token.set_extension('reversed', getter=get_reversed, force=True)
    
    # Process the text and print the reversed attribute for each token
    doc = nlp_sm("All generalizations are false, including this one.")
    
    df = pd.DataFrame({'token': [token for token in doc],
                       'reversed': [token._.reversed for token in doc]})
    print('Doc:', doc.text)
    print(df)
    
    
def Setting_extension_attributes_2(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "10. Setting extension attributes (2)"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Create the nlp object
    nlp_sm = spacy.load('en_core_web_sm')
    
    print('------------------------------------------has_number')
    # Define the getter function
    def get_has_number(doc):
        # Return if any of the tokens in the doc return True for token.like_num
        return any(token.like_num for token in doc)
    
    # Register the Doc property extension 'has_number' with the getter get_has_number
    Doc.set_extension('has_number', getter=get_has_number, force=True)
    
    # Process the text and check the custom has_number attribute 
    doc = nlp_sm("The museum closed for five years in 2012.")
    
    print('Doc:', doc.text)
    print('has_number:', doc._.has_number)
    
    
    print('---------------------------------------------to_html')
    # Define the method
    def to_html(span, tag):
        # Wrap the span text in a HTML tag and return it
        return '<{tag}>{text}</{tag}>'.format(tag=tag, text=span.text)
    
    # Register the Span property extension 'to_html' with the method to_html
    Span.set_extension('to_html', method=to_html, force=True)
    
    # Process the text and call the to_html method on the span with the tag name 'strong'
    doc = nlp_sm("Hello world, this is a sentence.")
    span = doc[0:2]
    
    print('Doc:', doc.text)
    print('span:', span.text)
    print('span.to_html:', span._.to_html('strong'))
    
    
def Entities_and_extensions(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "11. Entities and extensions"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Create the nlp object
    nlp_sm = spacy.load('en_core_web_sm')
    
    print('---------------------------------------wikipedia_url')
    def get_wikipedia_url(span):
        # Get a Wikipedia URL if the span has one of the labels
        if span.label_ in ('PERSON', 'ORG', 'GPE', 'LOCATION'):
            entity_text = span.text.replace(' ', '_')
            return "https://en.wikipedia.org/w/index.php?search=" + entity_text
    
    # Set the Span extension wikipedia_url using get getter get_wikipedia_url
    Span.set_extension('wikipedia_url', getter=get_wikipedia_url, force=True)
    
    doc = nlp_sm("In over fifty years from his very first recordings right through to his last album, David Bowie was at the vanguard of contemporary culture.")
    df = pd.DataFrame({'ent': [ent.text for ent in doc.ents],
                       'label': [ent.label_ for ent in doc.ents],
                       'wikipedia_url': [ent._.wikipedia_url for ent in doc.ents]})
    print('Doc:', doc.text)
    print(df)
    
    
    
def Components_with_extensions(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "12. Components with extensions"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Create the nlp object
    nlp_en = English()
    print('Pipeline component names:', nlp_en.pipe_names)
    
    
    print('--------------------------------Creating the matcher')
    # Read files
    filename = 'countries_and_capitals.csv'
    COUNTRIES_AND_CAPITALS = pd.read_csv(filename)
    
    COUNTRIES = COUNTRIES_AND_CAPITALS.Country.tolist()
    #capitals = {country: capital for _, (country, capital) in COUNTRIES_AND_CAPITALS.iterrows()}
    capitals = dict(COUNTRIES_AND_CAPITALS.values.tolist())
    
    # Initialize the PhraseMatcher
    matcher = PhraseMatcher(nlp_en.vocab)
    
    # Create pattern Doc objects and add them to the matcher
    # This is the faster version of: [nlp(country) for country in COUNTRIES]
    patterns = list(nlp_en.pipe(COUNTRIES))
    matcher.add('COUNTRY', None, *patterns)
    
    
    print('------------------------------Prepating the pipeline')
    def countries_component(doc):
        # Create an entity Span with the label 'GPE' for all matches
        doc.ents = [Span(doc, start, end, label='GPE') for match_id, start, end in matcher(doc)]
        return doc
    
    # Add the component to the pipeline
    nlp_en.add_pipe(countries_component, last=True)
    print('Pipeline component names:', nlp_en.pipe_names)
    
    print('---------------------Setting Doc Property extensions')
    # Getter that looks up the span text in the dictionary of country capitals
    get_capital = lambda span: capitals.get(span.text)
    
    # Register the Span extension attribute 'capital' with the getter get_capital 
    Span.set_extension('capital', getter=get_capital, force=True)
    
    
    print('----------------------------------------Applying all')
    # Process the text and print the entity text, label and capital attributes
    doc = nlp_en("Czech Republic may help Slovakia protect its airspace")
    
    df = pd.DataFrame({'ent': [ent.text for ent in doc.ents],
                       'label': [ent.label_ for ent in doc.ents],
                       'capital': [ent._.capital for ent in doc.ents]})
    print('Doc:', doc.text)
    print(df)
    
        
        
def Scaling_and_performance(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "13. Scaling and performance"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Create the nlp object
    nlp_sm = spacy.load('en_core_web_sm')
    
    print('--------------------Processing large volumes of text')
    # Reading the data
    wikipedia_filenames = glob.glob('Wikipedia articles\*.txt')
    LOTS_OF_TEXTS = []
    for filename in wikipedia_filenames:
        with open(filename, 'r', encoding='UTF-8') as f: LOTS_OF_TEXTS.append(f.read())
    
    print("INEFFICIENT MODE...")
    # Log the start time
    start_time = time.time()
    docs = [nlp_sm(text) for text in LOTS_OF_TEXTS]
    # Log the end time
    end_time = time.time()
    print('Time taken: ', end_time - start_time,' sec.')
    
    
    print("EFFICIENT MODE...")
    # Log the start time
    start_time = time.time()
    docs = list(nlp_sm.pipe(LOTS_OF_TEXTS))
    # Log the end time
    end_time = time.time()
    print('Time taken: ', end_time - start_time,' sec.')
    
    
    print('------------------------------Passing in context (1)')
    data = [(text, {'id': i, 'page_number': i*100}) for i, text in enumerate(LOTS_OF_TEXTS, start=1)]
    
    for doc, context in nlp_sm.pipe(data, as_tuples=True):
        print('Doc No.{} has {} tokens and begins in page {}.'.format(context['id'], len(doc), context['page_number']))
    
    print('------------------------------Passing in context (2)')
    Doc.set_extension('id', default=None, force=True)
    Doc.set_extension('page_number', default=None, force=True)
    
    for doc, context in nlp_sm.pipe(data, as_tuples=True):
        doc._.id = context['id']
        doc._.page_number = context['page_number']
        print('Doc No.{} has {} tokens and begins in page {}.'.format(doc._.id, len(doc), doc._.page_number))
    
    print('------------------------Using only the tokenizer (2)')
    print("INEFFICIENT MODE...")
    # Log the start time
    start_time = time.time()
    doc = nlp_sm("Hello world")
    # Log the end time
    end_time = time.time()
    print('Time taken: ', end_time - start_time,' sec.')
    
    
    print("EFFICIENT MODE...")
    # Log the start time
    start_time = time.time()
    doc = nlp_sm.make_doc("Hello world!")
    # Log the end time
    end_time = time.time()
    print('Time taken: ', end_time - start_time,' sec.')
    
    
    print('-----------------------Disabling pipeline components')
    # Disable tagger and parser
    with nlp_sm.disable_pipes('tagger', 'parser'):
        # Process the text and print the entities
        docs = [nlp_sm(text) for text in LOTS_OF_TEXTS]
        for i, doc in enumerate(docs, start=1):
            print('Doc No.{} has {} identified ents'.format(i, len(doc.ents)))
    
    
    
def Processing_streams(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "14. Processing streams"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Create the nlp object
    nlp_sm = spacy.load('en_core_web_sm')
    
    print('-------------------------------Just processing texts')
    TEXTS = ['McDonalds is my favorite restaurant.',
             'Here I thought @McDonalds only had precooked burgers but it seems they only have not cooked ones?? I have no time to get sick..',
             'People really still eat McDonalds :(',
             'The McDonalds in Spain has chicken wings. My heart is so happy ',
             '@McDonalds Please bring back the most delicious fast food sandwich of all times!!....The Arch Deluxe :P',
             'please hurry and open. I WANT A #McRib SANDWICH SO BAD! :D',
             'This morning i made a terrible decision by gettin mcdonalds and now my stomach is payin for it']
    
    print("INEFFICIENT MODE...")
    # Log the start time
    start_time = time.time()
    # Process the texts and print the adjectives
    for text in TEXTS:
        doc = nlp_sm(text)
        print([token.text for token in doc if token.pos_ == 'ADJ'])
    # Log the end time
    end_time = time.time()
    print('Time taken: ', end_time - start_time,' sec.')
    
    
    print("\nEFFICIENT MODE...")
    # Log the start time
    start_time = time.time()
    # Process the texts and print the adjectives
    for doc in nlp_sm.pipe(TEXTS):
        print([token.text for token in doc if token.pos_ == 'ADJ'])
    # Log the end time
    end_time = time.time()
    print('Time taken: ', end_time - start_time,' sec.')
    
    
    print('------------------------------------Finding entities')
    print("INEFFICIENT MODE...")
    # Log the start time
    start_time = time.time()
    # Process the texts and print the entities
    docs = [nlp_sm(text) for text in TEXTS]
    entities = [doc.ents for doc in docs]
    print(*entities)
    # Log the end time
    end_time = time.time()
    print('Time taken: ', end_time - start_time,' sec.')
    
    
    print("\nEFFICIENT MODE...")
    # Log the start time
    start_time = time.time()
    # Process the texts and print the entities
    docs = list(nlp_sm.pipe(TEXTS))
    entities = [doc.ents for doc in docs]
    print(*entities)
    # Log the end time
    end_time = time.time()
    print('Time taken: ', end_time - start_time,' sec.')
    
    
    print('------------------------------------Creatig patterns')
    people = ['David Bowie', 'Angela Merkel', 'Lady Gaga']
    
    print("INEFFICIENT MODE...")
    # Log the start time
    start_time = time.time()
    # Create a list of patterns for the PhraseMatcher
    patterns = [nlp_sm(person) for person in people]
    # Log the end time
    end_time = time.time()
    print('Time taken: ', end_time - start_time,' sec.')
    
    
    print("\nEFFICIENT MODE...")
    # Log the start time
    start_time = time.time()
    # Create a list of patterns for the PhraseMatcher
    patterns = list(nlp_sm.pipe(people))
    # Log the end time
    end_time = time.time()
    print('Time taken: ', end_time - start_time,' sec.')
    
    
    
def Processing_data_with_context(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "15. Processing data with context"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Create the nlp object
    nlp_sm = spacy.load('en_core_web_sm')
    
    print('-------Reading data and setting attribute extensions')
    DATA = [
            ('One morning, when Gregor Samsa woke from troubled dreams, he found himself transformed in his bed into a horrible vermin.',
             {'author': 'Franz Kafka', 'book': 'Metamorphosis'}),
            ("I know not all that may be coming, but be it what it will, I'll go to it laughing.",
             {'author': 'Herman Melville', 'book': 'Moby-Dick or, The Whale'}),
            ('It was the best of times, it was the worst of times.',
             {'author': 'Charles Dickens', 'book': 'A Tale of Two Cities'}),
            ('The only people for me are the mad ones, the ones who are mad to live, mad to talk, mad to be saved, desirous of everything at the same time, the ones who never yawn or say a commonplace thing, but burn, burn, burn like fabulous yellow roman candles exploding like spiders across the stars.',
             {'author': 'Jack Kerouac', 'book': 'On the Road'}),
            ('It was a bright cold day in April, and the clocks were striking thirteen.',
             {'author': 'George Orwell', 'book': '1984'}),
            ('Nowadays people know the price of everything and the value of nothing.',
             {'author': 'Oscar Wilde', 'book': 'The Picture Of Dorian Gray'})
           ]
    
    # Register the Doc extension 'author' (default None)
    Doc.set_extension('author', default=None, force=True)
    
    # Register the Doc extension 'book' (default None)
    Doc.set_extension('book', default=None, force=True)
    
    print('---------------------------Generating the doc object')
    for doc, context in nlp_sm.pipe(DATA, as_tuples=True):
        # Set the doc._.book and doc._.author attributes from the context
        doc._.book = context['book']
        doc._.author = context['author']
        
        # Print the text and custom attribute data
        print(doc.text, '\n', "— '{}' by {}".format(doc._.book, doc._.author), '\n')
    
    
    
def Selective_processing(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "16. Selective processing"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Create the nlp object
    nlp_sm = spacy.load('en_core_web_sm')
    
    # Reading the data
    text = "Chick-fil-A is an American fast food restaurant chain headquartered in the city of College Park, Georgia, specializing in chicken sandwiches."
    
    print('----------------------------------------nlp.make_doc')
    print("Applying nlp directly...")
    # Log the start time
    start_time = time.time()
    # Only tokenize the text
    doc = nlp_sm(text)
    print([token.text for token in doc])
    # Log the end time
    end_time = time.time()
    print('Time taken: ', end_time - start_time,' sec.')
    
    print("\nApplying nlp.make_doc...")
    # Log the start time
    start_time = time.time()
    # Only tokenize the text
    doc = nlp_sm.make_doc(text)
    print([token.text for token in doc])
    # Log the end time
    end_time = time.time()
    print('Time taken: ', end_time - start_time,' sec.')
    
    
    print('-----------------------------------nlp.disable_pipes')
    # Log the start time
    start_time = time.time()
    # Disable the tagger and parser
    with nlp_sm.disable_pipes('tagger', 'parser'):
        # Process the text
        doc = nlp_sm(text)
        # Print the entities in the doc
        print([(ent.text, ent.label_) for ent in doc.ents])
    end_time = time.time()
    print('Time taken: ', end_time - start_time,' sec.')
    
    
        
def main():
    print("****************************************************")
    print("** BEGIN                                          **")
    
    Processing_pipelines()
    What_happens_when_you_call_nlp()
    Inspecting_the_pipeline()
    Custom_pipeline_components()
    Simple_components()
    Complex_components()
    Extension_attributes()
    Setting_extension_attributes_1()
    Setting_extension_attributes_2()
    Entities_and_extensions()
    Components_with_extensions()
    Scaling_and_performance()
    Processing_streams()
    Processing_data_with_context()
    Selective_processing()

    
    print("****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    pd.reset_option("display.max_columns")
    plt.style.use('default')
    np.set_printoptions(formatter={'float': None})