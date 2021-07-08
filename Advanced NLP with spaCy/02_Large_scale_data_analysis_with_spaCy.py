# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 2: Large-scale data analysis with spaCy
    In this chapter, you'll use your new skills to extract specific information 
    from large volumes of text. You'll learn how to make the most of spaCy's 
    data structures, and how to effectively combine statistical and rule-based 
    approaches for text analysis.
Source: https://learn.datacamp.com/courses/advanced-nlp-with-spacy
"""
###############################################################################
## Importing libraries
###############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pprint import pprint

import spacy
from spacy.lang.en import English
from spacy.lang.de import German
from spacy.tokens import Doc
from spacy.tokens import Span
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher

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
## NLP configuration
###############################################################################
nlp = spacy.load('en_core_web_lg')

###############################################################################
## Main part of the code
###############################################################################
def Data_Structures_1(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "1. Data Structures (1)"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed)
    
    text = "I love coffee"
    doc = nlp(text)
    
    word = 'coffee'
    print('Text:', doc.text)
    print('Word:', word, '\n')
    
    print('-------------------Shared vocab and string store (1)')
    # String store: lookup table in both directions
    word_hash = nlp.vocab.strings[word]
    word_string = nlp.vocab.strings[word_hash]
    
    print('Hash of the word    :', word_hash)
    print('String from the hash:', word_string)
    
    # Hashes can't be reversed – that's why we need to provide the shared vocab
    try: 
        new_hash = 3197928453018144410
        string = nlp.vocab.strings[new_hash]
        print(string)
    except: print(f'\nString {new_hash} not found!\n')
    
    print('-------------------Shared vocab and string store (2)')
    # Look up the string and hash in nlp.vocab.strings
    print('Using NLP')
    hash_id = nlp.vocab.strings[word]
    print('hash value  :', hash_id)
    print('string value:', nlp.vocab.strings[hash_id], '\n')
    
    #The doc also exposes the vocab and strings
    print('Using DOC')
    print('hash value  :', doc.vocab.strings[word])
    print('string value:', doc.vocab.strings[hash_id], '\n')
    
    print('------------------Lexemes: entries in the vocabulary')
    lexeme = nlp.vocab[word]
    # print the lexical attributes
    print('lexeme.text    :', lexeme.text)
    print('lexeme.orth    :', lexeme.orth)
    print('lexeme.is_alpha:', lexeme.is_alpha)
    
    
    
def Strings_to_hashes(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "2. Strings to hashes"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('-------------------------------------------------cat')
    word = 'cat'
    # Look up the hash for the word "cat"
    word_hash = nlp.vocab.strings[word]
    # Look up the cat_hash to get the string
    word_string = nlp.vocab.strings[word_hash]
    
    print('Word        :', word)
    print('Hash value  :', word_hash)
    print('String value:', word_string, '\n')
    
    print('----------------------------------------------PERSON')
    word = 'PERSON'
    # Look up the hash for the word "cat"
    word_hash = nlp.vocab.strings[word]
    # Look up the cat_hash to get the string
    word_string = nlp.vocab.strings[word_hash]
    
    print('Word        :', word)
    print('Hash value  :', word_hash)
    print('String value:', word_string, '\n')
    
    
    
def Vocab_hashes_and_lexemes(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "3. Vocab, hashes and lexemes"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('-------------------------------------Hash not found!')
    # Create an English and German nlp object
    nlp_en = nlp #English()
    nlp_de = German()
    
    # Get the ID for the string 'Bowie'
    word = 'Bowie'
    bowie_id = nlp_en.vocab.strings[word]
    print('Word              :', word)
    print('Hash_id in English:', bowie_id)
    
    # Look up the ID for 'Bowie' in the vocab
    try: print('Hash in German    :', nlp_de.vocab.strings[bowie_id])
    except: print(f'\nString {bowie_id} not found in German Vocab!\n')
    
    
    
def Data_Structures_2(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "4. Data Structures (2)"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('--------------------------------------The Doc object')
    # Create an nlp object
    #nlp = English()
    
    # The words and spaces to create the doc from
    words = ['Hello', 'world', '!']
    spaces = [True, False, False]
    print('words          :', words)
    print('spaces         :', spaces)
    
    # Create a doc manually
    doc = Doc(nlp.vocab, words=words, spaces=spaces)
    print('doc            :', doc)
    
    print('---------------------------------The Span object (2)')
    # Create a span with a label
    span_with_label = Span(doc, 0, 2, label="GREETING")
    print('span_with_label:', span_with_label)
    print('Original ents  :', [(ent.text, ent.label_) for ent in doc.ents])
    
    # Add span to the doc.ents
    doc.ents = [span_with_label]
    print('Modified ents  :', [(ent.text, ent.label_) for ent in doc.ents])
    
    
    
def Creating_a_Doc(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "5. Creating a Doc"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('-----------------------------------------------Ex. 1')
    # Desired text: "spaCy is cool!"
    words = ['spaCy', 'is', 'cool', '!']
    spaces = [True, True, False, False]
    print('words :', words)
    print('spaces:', spaces)
    
    # Create a Doc from the words and spaces
    doc = Doc(nlp.vocab, words=words, spaces=spaces)
    print('doc   :', doc.text)

    print('-----------------------------------------------Ex. 2')
    # Desired text: "Go, get started!"
    words = ['Go', ',', 'get', 'started', '!']
    spaces = [False, True, True, False, False]
    print('words :', words)
    print('spaces:', spaces)
    
    # Create a Doc from the words and spaces
    doc = Doc(nlp.vocab, words=words, spaces=spaces)
    print(doc.text)
    print('doc   :', doc.text)

    print('-----------------------------------------------Ex. 3')
    # Desired text: "Oh, really?!"
    words = ['Oh', ',', 'really', '?', '!']
    spaces = [False, True, False, False, False]
    print('words :', words)
    print('spaces:', spaces)
    
    # Create a Doc from the words and spaces
    doc = Doc(nlp.vocab, words=words, spaces=spaces)
    print('doc   :', doc.text)
    
    
    
def Docs_spans_and_entities_from_scratch(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "6. Docs, spans and entities from scratch"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('-----------------------------------------------Ex. 1')
    words = ['I', 'like', 'David', 'Bowie']
    spaces = [True, True, True, False]
    print('words        :', words)
    print('spaces       :', spaces)
    
    # Create a doc from the words and spaces
    doc = Doc(nlp.vocab, words=words, spaces=spaces)
    print('doc          :', doc.text)
    print('Original ents:', [(ent.text, ent.label_) for ent in doc.ents])
    
    print('-----------------------------------------------Ex. 2')
    # Create a span for "David Bowie" from the doc and assign it the label "PERSON"
    span = Span(doc, 2, 4, label='PERSON')
    print('Created Span :', (span.text, span.label_))
    
    print('-----------------------------------------------Ex. 3')
    # Add the span to the doc's entities
    doc.ents = [span]
    
    # Print entities' text and labels
    print('Modified ents:', [(ent.text, ent.label_) for ent in doc.ents])
    
    
    
def Data_structures_best_practices(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "7. Data structures best practices"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('---------------------------------------------Explore')
    doc = nlp('Berlin is a nice city')
    print('doc:', doc.text)
    
    df = pd.DataFrame({'.text'  : [token.text for token in doc],
                       '.tag_'   : [token.tag_  for token in doc],
                       '.pos_'  : [token.pos_ for token in doc], # Coarse-grained part-of-speech tags
                       'Explain': [spacy.explain(token.pos_) for token in doc]}) # Fine-grained part-of-speech tags
    print(df)

    print('-------------Finding proper nouns followed by a verb')
    for token in doc:
        # Check if the current token is a proper noun
        if token.pos_ == 'PROPN':
            # Check if the next token is a verb
            if doc[token.i + 1].pos_ in ['VERB','AUX']:
                print('Found a verb after a proper noun!', [token.text, doc[token.i + 1].text])
    
    
def Word_vectors_and_similarity(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "8. Word vectors and similarity"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('-----------------------------Similarity examples (1)')
    # Compare two documents
    doc1 = nlp("I like fast food")
    doc2 = nlp("I like pizza")
    print("doc1       :", doc1.text)
    print("doc2       :", doc2.text)
    print('Simmilarity:', doc1.similarity(doc2), '\n')
    
    # Compare two documents
    doc1 = nlp("I like vegetables")
    doc2 = nlp("I like pizza")
    print("doc1       :", doc1.text)
    print("doc2       :", doc2.text)
    print('Simmilarity:', doc1.similarity(doc2), '\n')
    
    # Compare two tokens
    doc = nlp("I like pizza and pasta")
    token1 = doc[2]
    token2 = doc[4]
    print("doc        :", doc.text)
    print("token1     :", token1.text)
    print("token2     :", token2.text)
    print('Simmilarity:', token1.similarity(token2), '\n')
    
    print('-----------------------------Similarity examples (2)')
    # Compare a document with a token
    doc = nlp("I like pizza")
    token = nlp("soap")[0]
    print("doc        :", doc.text)
    print("token      :", token.text)
    print('Simmilarity:', doc.similarity(token), '\n')
    
    # Compare a span with a document
    span = nlp("I like pizza and pasta")[2:5]
    doc = nlp("McDonalds sells burgers")
    print("span       :", span.text)
    print("doc        :", doc.text)
    print('Simmilarity:', span.similarity(doc), '\n')
    
    print('-------------------------------Word vectors in spaCy')
    doc = nlp("I have a banana")
    
    # Access the vector via the token.vector attribute
    print("doc        :", doc.text)
    print('tokenn     :', doc[3].text)
    print(f"vector     : \n{doc[3].vector} \n({len(doc[3].vector)} dimensions)\n")
    
    print('-------Similarity depends on the application context')
    doc1 = nlp("I like cats")
    doc2 = nlp("I hate cats")
    print("doc1       :", doc1.text)
    print("doc2       :", doc2.text)
    print('Simmilarity:', doc1.similarity(doc2), '\n')
    
    
    
def Inspecting_word_vectors(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "9. Inspecting word vectors"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('-----------------------------------Vector of bananas')
    # Process a text
    doc = nlp("Two bananas in pyjamas")
    
    # Get the vector for the token "bananas"
    bananas = doc[1]
    bananas_vector = bananas.vector
    
    print("doc   :", doc.text)
    print('tokenn:', bananas.text)
    print(f"vector: \n{bananas_vector} \n({len(bananas_vector)} dimensions)\n")
    
    
    
def Comparing_similarities(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "10. Comparing similarities"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('-----------------------------------------------Ex. 1')
    doc1 = nlp("It's a warm summer day")
    doc2 = nlp("It's sunny outside")
    print("doc1       :", doc1.text)
    print("doc2       :", doc2.text)
    
    # Get the similarity of doc1 and doc2
    similarity = doc1.similarity(doc2)
    print('Simmilarity:', similarity, '\n')
    
    print('-----------------------------------------------Ex. 2')
    doc = nlp("TV and books")
    print("doc        :", doc.text)
    
    token1, token2 = doc[0], doc[2]
    print("token1     :", token1.text)
    print("token2     :", token2.text)
    
    # Get the similarity of the tokens "TV" and "books" 
    similarity = token1.similarity(token2)
    print('Simmilarity:', similarity, '\n')
    
    print('-----------------------------------------------Ex. 3')
    doc = nlp("This was a great restaurant. Afterwards, we went to a really nice bar.")
    print("doc        :", doc.text)
    
    # Create spans for "great restaurant" and "really nice bar"
    span1 = doc[3:5]
    span2 = doc[12:15]
    print("span1      :", span1.text)
    print("span2      :", span2.text)
    
    # Get the similarity of the spans
    similarity = span1.similarity(span2)
    print('Simmilarity:', similarity, '\n')
    
    
    
def Combining_models_and_rules(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "11. Combining models and rules"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('--------------------------Recap: Rule-based Matching')
    # Initialize with the shared vocab
    matcher = Matcher(nlp.vocab)
    
    # Patterns are lists of dictionaries describing the tokens
    pattern = [{'LEMMA': 'love', 'POS': 'VERB'}, {'LOWER': 'cats'}]
    matcher.add('LOVE_CATS', None, pattern)
    
    # Operators can specify how often a token should be matched
    pattern = [{'TEXT': 'very', 'OP': '+'}, {'TEXT': 'happy'}]
    matcher.add('SENTIMENT', None, pattern)
    
    # Calling matcher on doc returns list of (match_id, start, end) tuples
    doc = nlp("I love cats and I'm very very happy")
    matches = matcher(doc)
    print('doc    :', doc, '\n')
    print('pattern:', pattern)
    print('matches:')
    pprint(matches)
    for match_id, start, end in matches:
        print('         ', doc[start:end])
    
    print('----------------------Adding statistical predictions')
    matcher = Matcher(nlp.vocab)
    pattern = [{'LOWER': 'golden'}, {'LOWER': 'retriever'}]
    matcher.add('DOG', None, pattern)
    doc = nlp("I have a Golden Retriever")
    print('doc            :', doc, '\n')
    print('pattern        :', pattern)
    for match_id, start, end in matcher(doc):
        span = doc[start:end]
        print('Matched span   :', span.text)
        # Get the span's root token and root head token
        print('Root token     :', span.root.text)
        print('Root head token:', span.root.head.text)
        # Get the previous token and its POS tag
        print('Previous token:', doc[start - 1].text, doc[start - 1].pos_)

    print('------------------------Efficent phrase matching (2)')
    matcher = PhraseMatcher(nlp.vocab)
    
    pattern = nlp("Golden Retriever")
    matcher.add('DOG', None, pattern)
    doc = nlp("I have a Golden Retriever")
    print('doc         :', doc, '\n')
    print('pattern     :', pattern)
    
    # iterate over the matches
    for match_id, start, end in matcher(doc):
        # get the matched span
        span = doc[start:end]
        print('Matched span:', span.text)
    
    
def Debugging_patterns_1(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "12. Debugging patterns (1)"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('---------------------------------Rule-based Matching')
    matcher = Matcher(nlp.vocab)
    pattern = [{'LOWER': 'silicon'}, {'LOWER': 'valley'}]
    doc = nlp("Can Silicon Valley workers rein in big tech from within?")
    matcher.add('TECH', None, pattern)
    matches = matcher(doc)
    print('doc    :', doc, '\n')
    print('pattern:', pattern)
    print('matches:')
    pprint(matches)
    for match_id, start, end in matches:
        print('         ', doc[start:end])
    
    
    
def Debugging_patterns_2(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "13. Debugging patterns (2)"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    doc = nlp("Twitch Prime, the perks program for Amazon Prime members offering free loot, games and other benefits, is ditching one of its best features: ad-free viewing. According to an email sent out to Amazon Prime members today, ad-free viewing will no longer be included as a part of Twitch Prime for new members, beginning on September 14. However, members with existing annual subscriptions will be able to continue to enjoy ad-free viewing until their subscription comes up for renewal. Those with monthly subscriptions will have access to ad-free viewing until October 15.")
    print('doc     :', doc, '\n')
    
    # Create the match pattern1
    pattern1 = [{'LOWER': 'amazon'}, {'IS_TITLE': True, 'POS': 'PROPN'}]
    for my_dict in [{'ORTH': '-'}, {'TEXT': '-'}, {'LOWER': '-'}, {'SHAPE': '-'}]:
        # Create the match pattern2
        pattern2 = [{'LOWER': 'ad'}, my_dict, {'LOWER': 'free'}, {'POS': 'NOUN'}]
        
        print(f'--------------------------------Using {my_dict}')
        
        # Initialize the Matcher and add the patterns
        matcher = Matcher(nlp.vocab)
        matcher.add('PATTERN1', None, pattern1)
        matcher.add('PATTERN2', None, pattern2)
        
        print('pattern1:', pattern1)
        print('pattern2:', pattern2)
        print('matches :')
        
        # Iterate over the matches
        for match_id, start, end in matcher(doc):
            # Print pattern string name and text of matched span
            print('          ', doc.vocab.strings[match_id], doc[start:end].text)
        
    
    
def Efficient_phrase_matching(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "14. Efficient phrase matching"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Read file
    filename = 'countries.csv'
    COUNTRIES = pd.read_csv(filename).Countries.tolist()
    
    doc = nlp("Czech Republic may help Slovakia protect its airspace")
    print('doc     :', doc, '\n')
    
    
    print('---------------------------------------------Explore')
    # Initialize the PhraseMatcher
    matcher = PhraseMatcher(nlp.vocab)
    
    # Create pattern Doc objects and add them to the matcher
    # This is the faster version of: [nlp(country) for country in COUNTRIES]
    patterns = list(nlp.pipe(COUNTRIES))
    matcher.add('COUNTRY', None, *patterns)
    
    # Call the matcher on the test document and print the result
    matches = matcher(doc)
    
    #print('pattern :', patterns)
    print('matches :')
    print([doc[start:end] for match_id, start, end in matches])
    
    
    
def Extracting_countries_and_relationships(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "15. Extracting countries and relationships"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    # Create the nlp object
    nlp = English()

    # Read files
    filename = 'countries.csv'
    COUNTRIES = pd.read_csv(filename).Countries.tolist()
    
    filename = 'cold_war.dat'
    with open(filename, 'r') as f: text = f.read()
    
    # Create a doc and find matches in it
    doc = nlp(text)

    # Initialize the PhraseMatcher
    matcher = PhraseMatcher(nlp.vocab)
    
    # Create pattern Doc objects and add them to the matcher
    # This is the faster version of: [nlp(country) for country in COUNTRIES]
    patterns = list(nlp.pipe(COUNTRIES))
    matcher.add('COUNTRY', None, *patterns)
    
    # Print the entities in the document
    pre_exist = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ == 'GPE']
    print(f"Existed GPE entities (before matching): {len(pre_exist)} entities.")
    pprint(pre_exist)
    
    print('------------------------------------Finding Entities')
    
    # Iterate over the matches
    for match_id, start, end in matcher(doc):
        # Create a Span with the label for "GPE"
        span = Span(doc, start, end, label='GPE')
        
        # Overwrite the doc.ents and add the span
        #doc.ents = tuple(set(doc.ents).union(set(span)))
        try: doc.ents = list(doc.ents) + [span]
        except: pass
    
    # Print the entities in the document
    after_match = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ == 'GPE']
    print(f"GPE entities (after maching): {len(after_match)} entities.")
    pprint(after_match)
    
    print('-------------------------------------------Root Head')
    # Iterate over the matches
    for match_id, start, end in matcher(doc):
        # Create a Span with the label for "GPE" and overwrite the doc.ents
        span = Span(doc, start, end, label='GPE')
        
        # Get the span's root head token
        span_root_head = span.root.head
        
        # Print the text of the span root's head token and the span text
        print(span_root_head.text, '-->', span.text)

    
        
def main():
    print("****************************************************")
    print("** BEGIN                                          **")
    
    Data_Structures_1()
    Strings_to_hashes()
    Vocab_hashes_and_lexemes()
    Data_Structures_2()
    Creating_a_Doc()
    Docs_spans_and_entities_from_scratch()
    Data_structures_best_practices()
    Word_vectors_and_similarity()
    Inspecting_word_vectors()
    Comparing_similarities()
    Combining_models_and_rules()
    Debugging_patterns_1()
    Debugging_patterns_2()
    Efficient_phrase_matching()
    Extracting_countries_and_relationships()
    
    print("****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    pd.reset_option("display.max_columns")
    plt.style.use('default')
    np.set_printoptions(formatter={'float': None})