# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 00:34:44 2021

@author: jaces
"""
# Import libraries
import spacy
import pandas as pd

from pprint import pprint

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')
nlp_es = spacy.load('es_core_news_lg')

# Read data
ted_talk = pd.read_csv('data/ted.csv')
print(f'Head of ted_talk: \n{ted_talk.head()}')

fakenews = pd.read_csv('data/fakenews.csv', index_col=0)
print(f'\n\nHead of fakenews: \n{fakenews.head()}')

print('*********************************************************')
print('BEGIN')
print('*********************************************************')
print('** 2. Text preprocessing, POS tagging and NER')
print('*********************************************************')
print('** 2.1. Tokenization and Lemmatization')
print('*********************************************************')
# Initiliaze string
string = "Hello! I don't know what I'm doing here."
print(string)

# Create a Doc object
doc = nlp(string)

# Generate list of tokens
tokens = [token.text for token in doc]
print('Tokens:', tokens)

# Generate list of lemmas
lemmas = [token.lemma_ for token in doc]
print('Lemmas:', lemmas)

# Repeating in spanish
string = "¡Hola! Yo no se que estoy haciendo aquí."
print(f'\n{string}')

# Create a Doc object
doc = nlp_es(string)

# Generate list of tokens
tokens = [token.text for token in doc]
print('Tokens:', tokens)

# Generate list of lemmas
lemmas = [token.lemma_ for token in doc]
print('Lemmas:', lemmas)

print('*********************************************************')
print('** 2.2. Identifying lemmas')
print('*********************************************************')
print('** 2.3. Tokenizing the Gettysburg Address')
print('*********************************************************')
# Read data
with open('data/gettysburg.dat','r', encoding='utf-8') as f: 
    gettysburg = f.read() 

# Create a Doc object
doc = nlp(gettysburg)

# Generate the tokens
tokens = [token.text for token in doc]
print(tokens)

print('*********************************************************')
print('** 2.4. Lemmatizing the Gettysburg address')
print('*********************************************************')
# Print the gettysburg address
print(f'Before lemmatization: \n{gettysburg}')

# Generate lemmas
lemmas = [token.lemma_ for token in doc]

# Convert lemmas into a string
print('After lemmatization: \n{}'.format(' '.join(lemmas)))

print('*********************************************************')
print('** 2.5. Text cleaning')
print('*********************************************************')
# Removing non-alphabetic characters
string = """
OMG!!!! This is like the best thing ever \t\n.
Wow, such an amazing song! I'm hooked. Top 5 definitely. ?
"""
print(f'String to treat: \n{string}')

# Generate list of tokens
doc = nlp(string)
lemmas = [token.lemma_ for token in doc]
print(f'\nAfter lemmatization: \n{lemmas}')

# Remove tokens that are not alphabetic
a_lemmas = [lemma for lemma in lemmas 
            if lemma.isalpha() or lemma == '-PRON-']
# Print string after text cleaning
print('\nAfter removing not alphabetic tokens:')
print(' '.join(a_lemmas))

# Get list of stopwords
stopwords = spacy.lang.en.stop_words.STOP_WORDS

# Remove stopwords and non-alphabetic tokens
a_lemmas = [lemma for lemma in lemmas
            if lemma.isalpha() and lemma not in stopwords]
# Print string after text cleaning
print('\nAfter removing stopwords:')
print(' '.join(a_lemmas),'\n')

# First 10 stopwords in spanish
print(sorted(list(spacy.lang.es.stop_words.STOP_WORDS))[:10])

print('*********************************************************')
print('** 2.6. Cleaning a blog post')
print('*********************************************************')
# Read data
with open('data/blog.dat','r', encoding='utf-8') as f: 
    blog = f.read() 
print(f'Original data: \n{blog}\n')    

# Create Doc object
doc = nlp(blog)

# Generate lemmatized tokens
lemmas = [token.lemma_ for token in doc]
print(f'After lematization: \n{lemmas}\n')

# Remove stopwords and non-alphabetic tokens
a_lemmas = [lemma for lemma in lemmas 
            if lemma.isalpha() and lemma not in stopwords]

# Print string after text cleaning
print('After removing stopwords and non-alphabetic tokesn: ')
print(' '.join(a_lemmas))

print('*********************************************************')
print('** 2.7. Cleaning TED talks in a dataframe')
print('*********************************************************')
# Read data
ted = ted_talk.head().copy(deep = True)

# Function to preprocess text
def preprocess(text):
    """Lemmatize a text and return it after cleaning stopwords and not alphanumerics tokens."""
    # Create Doc object without ner and parser
    # ner: EntityRecognizer, parser: owers the sentence boundary detection
    # Return lemmas without stopwords and non-alphabetic characters
    return ' '.join([token.lemma_ for token in nlp(text, disable=['ner', 'parser']) 
                     if token.lemma_.isalpha() and token.lemma_ not in stopwords])
  
# Apply preprocess to ted['transcript']
ted['clean_transcript'] = ted['transcript'].apply(preprocess)
print(ted)

print('*********************************************************')
print('** 2.8. Part-of-speech tagging')
print('*********************************************************')
# Initiliaze string
string = "Jane is an amazing guitarist"
print(f'Original string: \n{string}')

# Create a Doc object
doc = nlp(string)

# Generate list of tokens and pos tags
pos = [(token.text, token.pos_) for token in doc]
print('\nPart-of-speech tagging:')
pprint(pos)

# Repeating in Spanish
string = "Ana es una fantástica guitarrista"
print(f'\nOriginal string: \n{string}')

# Create a Doc object
doc = nlp_es(string)

# Generate list of tokens and pos tags
pos = [(token.text, token.pos_) for token in doc]
print('\nPart-of-speech tagging:')
pprint(pos)

print('*********************************************************')
print('** 2.9. POS tagging in Lord of the Flies')
print('*********************************************************')
# Read data
with open('data/lotf.dat','r', encoding='utf-8') as f: 
    lotf = f.read() 
print(f'Original data: \n{lotf}\n')    

# Create a Doc object
doc = nlp(lotf)

# Generate tokens and pos tags
pos = [(token.text, token.pos_) for token in doc]
pprint(pos)

print('*********************************************************')
print('** 2.10. Counting nouns in a piece of text')
print('*********************************************************')
# Returns number of proper nouns
def proper_nouns(text, model=nlp):
    """Return number of total proper nouns."""
    # Create doc object, generate list of POS tags and return number of proper nouns
    return [token.pos_ for token in model(text)].count('PROPN')

# Returns number of other nouns
def nouns(text, model=nlp):
    """Returns number of total other nouns."""
    # Create doc object, generate list of POS tags and return number of other nouns
    return [token.pos_ for token in model(text)].count('NOUN')

# First example in English
string = "Abdul, Bill and Cathy went to the market to buy apples."
print(f'Original string   : {string}')
print( 'Proper nouns found: {}'.format(proper_nouns(string, nlp)))
print( 'Other nouns found : {}\n'.format(nouns(string, nlp)))

# Second example in Spanish
string = 'Gabriela y José fueron al mercado a comprar manzanas.'
print(f'Original string   : {string}')
print( 'Proper nouns found: {}'.format(proper_nouns(string, nlp_es)))
print( 'Other nouns found : {}\n'.format(nouns(string, nlp_es)))

print('*********************************************************')
print('** 2.11. Noun usage in fake news')
print('*********************************************************')
# Read data
headlines = fakenews.copy(deep=True)

# Compute the features: num_propn and num_noun
headlines['num_propn'] = headlines['title'].apply(proper_nouns)
headlines['num_noun'] = headlines['title'].apply(nouns)

# Compute mean of proper nouns
real_propn = headlines[headlines['label'] == 'REAL']['num_propn'].mean()
fake_propn = headlines[headlines['label'] == 'FAKE']['num_propn'].mean()

# Compute mean of other nouns
real_noun = headlines[headlines['label'] == 'REAL']['num_noun'].mean()
fake_noun = headlines[headlines['label'] == 'FAKE']['num_noun'].mean()

# Print results
print("Mean no. of proper nouns in real and fake headlines are %.2f and %.2f respectively"%(real_propn, fake_propn))
print("Mean no. of other nouns in real and fake headlines are %.2f and %.2f respectively"%(real_noun, fake_noun))

print('*********************************************************')
print('** 2.12. Named entity recognition')
print('*********************************************************')
# NER using spaCy with english language
string = "John Doe is a software engineer working at Google. He lives in France."
print('Original string:', string)

# Create Doc object
doc = nlp(string)

# Generate named entities
ne = [(ent.text, ent.label_) for ent in doc.ents]
print(ne)

# NER using spaCy with spanish language
string = "Gabriela Cortez es una arquitecta que trabaja en HDR, ella vive en Irlanda."
print('\nOriginal string:', string)

# Create Doc object
doc = nlp_es(string)

# Generate named entities
ne = [(ent.text, ent.label_) for ent in doc.ents]
print(ne)

print('*********************************************************')
print('** 2.13. Named entities in a sentence')
print('*********************************************************')
# Create a Doc instance 
string = 'Sundar Pichai is the CEO of Google. Its headquarters is in Mountain View.'
doc = nlp(string)
print('Original string:', string)

# Generate named entities
ne = [(ent.text, ent.label_) for ent in doc.ents]
print('Entities identified:', ne)

print('*********************************************************')
print('** 2.14. Identifying people mentioned in a news article')
print('*********************************************************')
# Read the data
with open('data/tc.dat','r', encoding='utf-8') as f: 
    tc = f.read() 
print(f'Original data: \n{tc}\n')   

def find_persons(text, model=nlp):
    """Return the identified persons."""
    # Create Doc object and eturn the identified persons
    return [ent.text for ent in model(text).ents if ent.label_ == 'PERSON']

print("Identified persons: ", find_persons(tc))

print('*********************************************************')
print('END')
print('*********************************************************') 