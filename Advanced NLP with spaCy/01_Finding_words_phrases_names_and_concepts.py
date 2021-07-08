# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 1: Finding words, phrases, names and concepts
    This chapter will introduce you to the basics of text processing with spaCy. 
    You'll learn about the data structures, how to work with statistical models, 
    and how to use them to predict linguistic features in your text.
Source: https://learn.datacamp.com/courses/advanced-nlp-with-spacy
"""
###############################################################################
## Importing libraries
###############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import webbrowser
import os
import tempfile

from pprint import pprint

import spacy #https://spacy.io/api/annotation#pos-tagging
from spacy import displacy
from spacy.lang.en import English
from spacy.lang.es import Spanish
from spacy.lang.de import German
from spacy.matcher import Matcher #https://spacy.io/usage/rule-based-matching#adding-patterns-attributes

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
def Introduction_to_spaCy(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "1. Introduction to spaCy"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('--------------------------------------The nlp object')
    nlp_en = English()
    nlp_es = Spanish()
    
    print(nlp_en)
    print(nlp_es)
    
    
    print('--------------------------------------The doc object')
    # Created by processing a string of text with the nlp object
    doc_en = nlp_en("Hello world!")
    doc_es = nlp_es("¡Hola mundo!")
    
    # Iterate over tokens in a Doc
    print(doc_en)
    for i, token in enumerate(doc_en, start=0):
        print(f"{i} - {token.text}")
   
    print(doc_es)
    for i, token in enumerate(doc_es, start=0):
        print(f"{i} - {token.text}")
    
    print('------------------------------------The token object')
    # Index into the Doc to get a single Token
    token_en = doc_en[1]
    token_es = doc_es[1]
    
    # Get the token text via the .text attribute
    print("Position 1:", token_en.text)
    print("Position 1:", token_es.text)

    print('-------------------------------------The span object')
    # A slice from the Doc is a Span object
    span_en = doc_en[1:4]
    span_es = doc_es[1:4]
    
    # Get the span text via the .text attribute
    print("From token 1:4 >>", span_en.text)
    print("From token 1:4 >>", span_es.text)
    
    print('----------------------------------Lexical attributes')
    sentence = "It costs $5, five dollars!"
    print(f'For sentence:"{sentence}", we have:')
    doc_en = nlp_en(sentence)
    df_en = pd.DataFrame({'Index'   : [token.i for token in doc_en],
                          'Text'    : [token.text for token in doc_en],
                          'is_stop' : [token.is_stop for token in doc_en],
                          'is_alpha': [token.is_alpha for token in doc_en],
                          'is_punct': [token.is_punct for token in doc_en],
                          'like_num': [token.like_num for token in doc_en]})
    df_en.set_index('Index', inplace=True)
    print(df_en,'\n')
    
    sentence = "¡Su costo es de $5, cinco dólares!"
    print(f'\nFor sentence:"{sentence}", we have:')
    doc_es = nlp_es(sentence)
    df_es = pd.DataFrame({'Index'   : [token.i for token in doc_es],
                          'Text'    : [token.text for token in doc_es],
                          'is_stop' : [token.is_stop for token in doc_es],
                          'is_alpha': [token.is_alpha for token in doc_es],
                          'is_punct': [token.is_punct for token in doc_es],
                          'like_num': [token.like_num for token in doc_es]})
    df_es.set_index('Index', inplace=True)
    print(df_es,'\n')
    
    
    
def Getting_Started(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "2. Getting Started"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('---------------------------------------------English')
    # Create the nlp object
    nlp = English()
    
    # Process a text
    doc = nlp("This is a sentence.")
    
    # Print the document text
    print(doc.text)
    
    
    print('----------------------------------------------German')
    # Create the nlp object
    nlp = German()
    
    # Process a text (this is German for: "Kind regards!")
    doc = nlp("Liebe Grüße!")
    
    # Print the document text
    print(doc.text)
    
    
    print('---------------------------------------------Spanish')
    # Create the nlp object
    nlp = Spanish()
    
    # Process a text (this is Spanish for: "How are you?")
    doc = nlp("¿Cómo estás?")
    
    # Print the document text
    print(doc.text)
    
    
    
def Documents_spans_and_tokens(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "3. Documents, spans and tokens"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('--------------------------------------The nlp object')
    nlp = English()
    print(nlp)
    
    print('--------------------------------------The doc object')
    # Process the text
    doc = nlp("I like tree kangaroos and narwhals.")
    print(doc)
    
    print('------------------------------------The token object')
    # Select the first token
    first_token = doc[0]
    
    # Print the first token's text
    print(first_token.text)
    
    
    print('-------------------------------------The span object')
    # A slice of the Doc for "tree kangaroos"
    tree_kangaroos = doc[2:4]
    print(tree_kangaroos.text)
    
    # A slice of the Doc for "tree kangaroos and narwhals" (without the ".")
    tree_kangaroos_and_narwhals = doc[2:6]
    print(tree_kangaroos_and_narwhals.text)
    
    
    
def Lexical_attributes(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "4. Lexical attributes"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    nlp = English()
    
    my_text = """
    In 1990, more than 60% of people in East Asia were in extreme poverty. 
    Now less than 4% are.
    """
    
    # Process the text
    doc = nlp(my_text)
    print(doc)
    print("\nFinding the porcentage...")
    print('--------------------------------------------Long way')
    # Iterate over the tokens in the doc
    for token in doc:
        # Check if the token resembles a number
        if token.like_num:
            # Get the next token in the document
            next_token = doc[token.i+1]
            # Check if the next token's text equals '%'
            if next_token.text == '%':
                print('Percentage found:', token.text)
                
    print('-------------------------------------------Short way')
    porcentage_found = [doc[token.i:token.i+2] for token in doc if token.like_num and doc[token.i+1].text=='%']
    print('Percentage found:', porcentage_found)
    
    
def Statistical_models(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "5. Statistical models"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('--------------------------------------Model Packages')
    # Load the small English model
    nlp = spacy.load('en_core_web_sm')
    print(nlp.pipe_names)
    
    print('----------------------Predicting Part-of-speech Tags')
    # Process a text
    doc = nlp("She ate the pizza")
    print(doc)
    
    # Iterate over the tokens
    df = pd.DataFrame({'Text'           : [token.text for token in doc],
                       'Part-Of-Speech' : [token.pos_ for token in doc]})
    print(df)
    
    print('-------------------Predicting Syntactic Dependencies')
    # Iterate over the tokens
    df = pd.DataFrame({'Text'            : [token.text for token in doc],
                       'pos_'            : [token.pos_ for token in doc],
                       'pos (explained)' : [spacy.explain(token.pos_) for token in doc],
                       'dep_'            : [token.dep_ for token in doc],
                       'dep (explained)' : [spacy.explain(token.dep_) for token in doc],
                       '.head.text'      : [token.head.text for token in doc]})
    print(df,'\n')
    
    print('---------------------------Predicting Named Entities')
    # Process a text
    doc = nlp(u"Apple is looking at buying U.K. startup for $1 billion")
    print(doc)
    
    # Iterate over the predicted entities
    df = pd.DataFrame({'ent.text'               : [ent.text for ent in doc.ents],
                       'ent.label_'             : [ent.label_ for ent in doc.ents],
                       'ent.label_ (explained)' : [spacy.explain(ent.label_) for ent in doc.ents]})
    print(df,'\n')
    
    print('---------------Visualizing Dependencies and Entities')
    doc = nlp(u"Apple is looking at buying U.K. startup for $1 billion")
    
    # generate the dependency diagram
    svg = displacy.render(doc, style="dep", jupyter=False)
    
    # generate the entity representation
    html = displacy.render(doc, style="ent", jupyter=False)

    # Save the svg file
    filename = "01_05_statistical_models.svg"
    with open(filename, 'w') as f: print(svg, file=f)
    
    # open an HTML file on my own (Windows) computer
    filename = 'file:///' + os.getcwd().replace('\\','/').replace(' ','%20') + '/' + filename
    tmp=tempfile.NamedTemporaryFile(delete=False)
    path=tmp.name+'.html'
    print(path)
    with open(path, 'w') as f: 
        f.write(f"""<html>
                  <head><title>{topic}</title></head>
                  <body>
                     <h1>{topic}</h1>
                     <h2>Visualizing entities</h2>
                     {html}
                     <h2>Visualizing dependencies</h2>
                     <img src="{filename}" width='100%'/>
                  </body>
               """)
    webbrowser.open('file:///' + path, new=2)
    
    
    
def Loading_models(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "7. Loading models"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('---------------------------------------------English')
    # Load the 'en_core_web_sm' model – spaCy is already imported
    nlp_en = spacy.load('en_core_web_sm')
    
    text = "It’s official: Apple is the first U.S. public company to reach a $1 trillion market value"
    
    # Process the text
    doc = nlp_en(text)
    
    # Print the document text
    print(doc.text)
    
    print('----------------------------------------------German')
    # Load the 'de_core_news_sm' model – spaCy is already imported
    nlp_de = spacy.load('de_core_news_sm')
    
    text = "Als erstes Unternehmen der Börsengeschichte hat Apple einen Marktwert von einer Billion US-Dollar erreicht"
    
    # Process the text
    doc = nlp_de(text)

    # Print the document text
    print(doc.text)
        
    
def Predicting_linguistic_annotations(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "8. Predicting linguistic annotations"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Load the 'en_core_web_sm' model – spaCy is already imported
    nlp = spacy.load('en_core_web_sm')
    
    print('-------------------------------Dependency predicting')
    text = "It’s official: Apple is the first U.S. public company to reach a $1 trillion market value"
    
    # Process the text
    doc = nlp(text)
    print(doc.text, '\n')
    
    for token in doc:
        # Get the token text, part-of-speech tag and dependency label
        # This is for formatting only
        print('{:<12}{:<10}{:<10}'.format(token.text, token.pos_, token.dep_))
    
    # Iterate over the tokens
    df = pd.DataFrame({'Text'            : [token.text for token in doc],
                       'pos_'            : [token.pos_ for token in doc],
                       'pos (explained)' : [spacy.explain(token.pos_) for token in doc],
                       'dep_'            : [token.dep_ for token in doc],
                       'dep (explained)' : [spacy.explain(token.dep_) for token in doc]})
    print(f'\n{df}\n')
    
    print('-----------------------------------Entity predicting')
    text = "It’s official: Apple is the first U.S. public company to reach a $1 trillion market value"
    
    # Process the text
    doc = nlp(text)
    print(doc.text, '\n')
    
    # Iterate over the predicted entities
    for ent in doc.ents:
        # print the entity text and its label
        print('{:<12}{:<8}{}'.format(ent.text, ent.label_, spacy.explain(ent.label_)))

    
    
def Predicting_named_entities_in_context(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "9. Predicting named entities in context"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Load the 'en_core_web_sm' model – spaCy is already imported
    nlp = spacy.load('en_core_web_sm')
    
    text = "New iPhone X release date leaked as Apple reveals pre-orders by mistake"

    # Process the text
    doc = nlp(text)
    print(doc)
    
    print('------------------------------------Finding entities')
    # Iterate over the entities
    print("Entities found:")
    for ent in doc.ents:
        # print the entity text and label
        print('{:<12}{:<8}{}'.format(ent.text, ent.label_, spacy.explain(ent.label_)))
        
    print('---------------------------------------Entity missed')
    # Get the span for "iPhone X"
    iphone_x = doc[1:3]
    
    # Print the span text
    print('Missing entity:', iphone_x.text)
    
    
def Rule_based_matching(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "10. Rule-based matching"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Load a model and create the nlp object
    nlp = spacy.load('en_core_web_sm')
    
    
    print('-------------------------------Using the Matcher (1)')
    # Initialize the matcher with the shared vocab
    matcher = Matcher(nlp.vocab)
    
    #Add the pattern to the matcher
    pattern = [{'ORTH': 'iPhone'}, {'ORTH': 'X'}]
    matcher.add('IPHONE_PATTERN', None, pattern)
    print('pattern = ', pattern)
    
    # Process some text
    doc = nlp("New iPhone X release date leaked")
    print(f"doc = '{doc}'\n")
    
    # Call the matcher on the doc
    matches = matcher(doc)
    
    # Iterate over the matches
    print("Matches found:")
    for match_id, start, end in matches:
        # Get the matched span
        matched_span = doc[start:end]
        print(matched_span.text)
        
        
    print('-------------------------Matching lexical attributes')
    # Initialize the matcher with the shared vocab
    matcher = Matcher(nlp.vocab)
    
    #Add the pattern to the matcher
    pattern = [{'IS_DIGIT': True},
               {'LOWER': 'fifa'},
               {'LOWER': 'world'},
               {'LOWER': 'cup'},
               {'IS_PUNCT': True}]
    matcher.add('FIFA_PATTERN', None, pattern)
    print('pattern = ')
    pprint(pattern)
    
    # Process some text
    doc = nlp("2018 FIFA World Cup: France won!")
    print(f"doc = '{doc}'\n")
    
    # Call the matcher on the doc
    matches = matcher(doc)
    
    # Iterate over the matches
    print("Matches found:")
    for match_id, start, end in matches:
        # Get the matched span
        matched_span = doc[start:end]
        print(matched_span.text)
    
    
    print('---------------------Matching other token attributes')
    # Initialize the matcher with the shared vocab
    matcher = Matcher(nlp.vocab)
    
    #Add the pattern to the matcher
    pattern = [{'LEMMA': 'love', 'POS': 'VERB'},
               {'POS': 'NOUN'}]
    matcher.add('PET_PATTERN', None, pattern)
    print('pattern = ', pattern)
    
    # Process some text
    doc = nlp("I loved dogs but now I love cats more.")
    print(f"doc = '{doc}'\n")
    
    # Call the matcher on the doc
    matches = matcher(doc)
    
    # Iterate over the matches
    print("Matches found:")
    for match_id, start, end in matches:
        # Get the matched span
        matched_span = doc[start:end]
        print(matched_span.text)
    
    
    print('-----------------Using operators and quantifiers (1)')
    # Initialize the matcher with the shared vocab
    matcher = Matcher(nlp.vocab)
    
    #Add the pattern to the matcher
    pattern = [{'LEMMA': 'buy'},
               {'POS': 'DET', 'OP': '?'}, # optional: match 0 or 1 times
               {'POS': 'NOUN'}]
    matcher.add('SHOPPING_PATTERN', None, pattern)
    print('pattern = ')
    pprint(pattern)
    
    # Process some text
    doc = nlp("I bought a smartphone. Now I'm buying apps.")
    print(f"doc = '{doc}'\n")
    
    # Call the matcher on the doc
    matches = matcher(doc)
    
    # Iterate over the matches
    print("Matches found:")
    for match_id, start, end in matches:
        # Get the matched span
        matched_span = doc[start:end]
        print(matched_span.text)
    
    
    
def Using_the_Matcher(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "11. Using the Matcher"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Load a model and create the nlp object
    nlp = spacy.load('en_core_web_sm')
    
    # Process some text
    doc = nlp("New iPhone X release date leaked as Apple reveals pre-orders by mistake")
    print(f"doc = '{doc}'\n")
    
    
    print('----------------------------Initializing the mathcer')
    # Initialize the Matcher with the shared vocabulary
    matcher = Matcher(nlp.vocab)
    
    
    print('-------------------------------Preparing the pattern')
    # Create a pattern matching two tokens: "iPhone" and "X"
    pattern = [{'TEXT': 'iPhone'}, {'TEXT': 'X'}]
    
    # Add the pattern to the matcher
    matcher.add('IPHONE_X_PATTERN', None, pattern)
    print(pattern)

    print('--------------------------------------------Matching')
    # Use the matcher on the doc
    matches = matcher(doc)
    print('Matches:', [doc[start:end].text for match_id, start, end in matches])
    
    
    
def Writing_match_patterns(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "12. Writing match patterns"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Load a model and create the nlp object
    nlp = spacy.load('en_core_web_sm')
    
    print('-----------Matches mentions of the full iOS versions')
    # Process some text
    doc = nlp("After making the iOS update you won't notice a radical system-wide redesign: nothing like the aesthetic upheaval we got with iOS 7. Most of iOS 11's furniture remains the same as in iOS 10. But you will discover some tweaks once you delve a little deeper.")
    print('doc:', doc)
    
    # Initialize the Matcher with the shared vocabulary
    matcher = Matcher(nlp.vocab)
    
    # Write a pattern for full iOS versions ("iOS 7", "iOS 11", "iOS 10")
    pattern = [{'TEXT': "iOS"}, {'IS_DIGIT': True}]
    print('pattern:', pattern)
    
    # Add the pattern to the matcher and apply the matcher to the doc
    matcher.add('IOS_VERSION_PATTERN', None, pattern)
    matches = matcher(doc)
    print('Total matches found:', len(matches))
    
    # Iterate over the matches and print the span text
    print('Matches:', [doc[start:end].text for match_id, start, end in matches], '\n')
    
    
    print('-----matches forms of "download" followed by a PROPN')
    doc = nlp("i downloaded Fortnite on my laptop and can't open the game at all. Help? so when I was downloading Minecraft, I got the Windows version where it is the '.zip' folder and I used the default program to unpack it... do I also need to download Winzip? ")
    print('doc:', doc)
    
    # Initialize the Matcher with the shared vocabulary
    matcher = Matcher(nlp.vocab)
    
    # Write a pattern that matches a form of "download" plus proper noun
    pattern = [{'LEMMA': 'download', 'POS': 'VERB'}, {'POS': 'PROPN'}]
    print('pattern:', pattern)
    
    # Add the pattern to the matcher and apply the matcher to the doc
    matcher.add('DOWNLOAD_THINGS_PATTERN', None, pattern)
    matches = matcher(doc)
    print('Total matches found:', len(matches))
    
    # Iterate over the matches and print the span text
    print('Matches:', [doc[start:end].text for match_id, start, end in matches], '\n')
    
    
    print("----matches adjectives followed by one or two NOUN's")
    doc = nlp("Features of the app include a beautiful design, smart search, automatic labels and optional voice responses.")
    print('doc:', doc)
    
    # Initialize the Matcher with the shared vocabulary
    matcher = Matcher(nlp.vocab)
    
    # Write a pattern for adjective plus one or two nouns
    pattern = [{'POS': 'ADJ'}, {'POS': 'NOUN'}, {'POS': 'NOUN', 'OP': '?'}]
    print('pattern:', pattern)
    
    # Add the pattern to the matcher and apply the matcher to the doc
    matcher.add('ADJ_NOUN_PATTERN', None, pattern)
    matches = matcher(doc)
    print('Total matches found:', len(matches))

    # Iterate over the matches and print the span text
    print('Matches:', [doc[start:end].text for match_id, start, end in matches], '\n')
    
        
def main():
    print("****************************************************")
    print("** BEGIN                                          **")
    
    Introduction_to_spaCy()
    Getting_Started()
    Documents_spans_and_tokens()
    Lexical_attributes()
    Statistical_models()
    Loading_models()
    Predicting_linguistic_annotations()
    Predicting_named_entities_in_context()
    Rule_based_matching()
    Using_the_Matcher()
    Writing_match_patterns()
    
    print("****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    pd.reset_option("display.max_columns")
    plt.style.use('default')
    np.set_printoptions(formatter={'float': None})