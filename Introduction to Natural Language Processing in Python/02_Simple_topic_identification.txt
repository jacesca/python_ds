# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 3: Logistic regression
    This chapter will introduce you to topic identification, which you can 
    apply to any text you encounter in the wild. Using basic NLP models, you 
    will identify topics from texts based on term frequencies. You'll 
    experiment and compare two simple methods: bag-of-words and Tf-idf using 
    NLTK, and a new library Gensim.
Source: https://learn.datacamp.com/courses/introduction-to-natural-language-processing-in-python
"""
###############################################################################
## Importing libraries
###############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re

from collections import Counter #To calculate frequencies
from collections import defaultdict #To initialize a dictionary that will assign a default value to non-existent keys.

import itertools #Allows us to iterate through a set of sequences as if they were one continuous sequence. 

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer #Lemmatization of a string
from nltk.tokenize import word_tokenize

from wordcloud import WordCloud #Documentation: https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html#wordcloud.WordCloud

from gensim.corpora.dictionary import Dictionary #To build corpora and dictionaries using simple classes and functions. Documentation: https://radimrehurek.com/gensim/auto_examples/index.html
from gensim.models.tfidfmodel import TfidfModel #To calculate the Term frequency - inverse document frequency

from glob import glob

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
#Get the file names of the images to add
filenames = glob("Wikipedia articles/*.txt")
articles = []
for file_article in filenames:
    with open(file_article,'r', encoding='utf-8') as f: 
        articles.append(f.read())


###############################################################################
## Main part of the code
###############################################################################
def Word_counts_with_bag_of_words(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "1. Word counts with bag-of-words"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    my_string = """
    The cat is in the box. The cat likes the box.
    The box is over the cat.
    """
    print(f'my_string = "{my_string}"')
    
    print('------------------------------Bag-of-words in Python')
    result = Counter(word_tokenize(my_string))
    print(f'Frequency: \n{result}\n')
    print(f"Two more common words: {result.most_common(2)}\n")
    
    print('----------Bag-of-words in Python after preprocessing')
    result = Counter(word_tokenize(my_string.lower()))
    print(f'Frequency: \n{result}\n')
    
    
    
def Building_a_Counter_with_bag_of_words(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "3. Building a Counter with bag-of-words"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------------------------Reading the data')
    with open('Wikipedia articles/wiki_text_debugging.txt','r') as f: 
        article = f.read()
    
    my_pattern = r"\'{3}\w+\'{3}"
    article_title = re.search(my_pattern, article).group()
    
    print('----------------------------------Infering the title')
    # Tokenize the article: tokens
    tokens = word_tokenize(article.lower())
    
    # Create a Counter with the lowercase tokens: bow_simple
    bow_simple = Counter(tokens)
    
    # Print the 10 most common tokens
    print(bow_simple.most_common(15), '\n')
    
    print('-------------------------------------------The title')
    print(article_title, '\n')
    
    
    
def Simple_text_preprocessing(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "4. Simple text preprocessing"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('---------------------------------------------English')
    my_string = """
    The cat is in the box. The cat likes the box.
    The box is over the cat.
    """
    print(f'my_string = "{my_string}"')
    
    # Text preprocessing with Python
    tokens = [w for w in word_tokenize(my_string.lower()) if ((w.isalpha()) & (w not in stopwords.words('english')))]
    
    result = Counter(tokens)
    print(f'Frequency: \n{result}\n')
    print(f"Two more common words: {result.most_common(2)}\n")
    
    print('---------------------------------------------Spanish')
    my_string = """
    El problema del matrimonio es que se acaba todas 
    las noches despues de hacer el amor, 
    y hay que volver a reconstruirlo todas las mananas 
    antes del desayuno.
    """
    print(f'my_string = "{my_string}"')
    
    # Text preprocessing with Python
    tokens = [w for w in word_tokenize(my_string.lower()) if ((w.isalpha()) & (w not in stopwords.words('spanish')))]
    
    result = Counter(tokens)
    print(f'Frequency: \n{result}\n')
    print(f"Two more common words: {result.most_common(2)}\n")
    
    print('----------------------------------Infering the title')
    with open('Wikipedia articles/wiki_text_debugging.txt','r') as f: 
        article = f.read()
    my_pattern = r"\'{3}\w+\'{3}"
    article_title = re.search(my_pattern, article).group()
    
    # Tokenize the article: tokens
    tokens = [w for w in word_tokenize(article.lower()) if ((w.isalpha()) & (w not in stopwords.words('english')))]
    
    # Create a Counter with the lowercase tokens: bow_simple
    bow_simple = Counter(tokens)
    
    # Print the 10 most common tokens
    print(f"Wikipedia article's (most common words): \n{bow_simple.most_common(5)}\n")
    print(f'Title: "{article_title}"\n')
    
    print('-------------------------------------------Wordcloud')
    # Generate and show the word cloud
    wordcloud = WordCloud(width=1360, height=960,         #Width and height of the canvas.
                          collocations=False,             #Not include biagrams
                          #max_words=100,                 #The maximum number of words.
                          #collocation_threshold=1000,    #score greater than this parameter to be counted as bigrams
                          stopwords={},                   #The words that will be eliminated
                          #min_word_length=6,             #Minimum number of letters a word must have to be included.
                          background_color="black", 
                          colormap='gist_rainbow').generate(' '.join(tokens))
    
    # Create a figure of the generated cloud
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Most Common Words', **title_param)
    
    fig.suptitle(topic, **suptitle_param)
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=.5) #To set the margins 
    plt.show()
    
    
    
def Text_preprocessing_practice(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "6. Text preprocessing practice"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------------------------Reading the data')
    with open('Wikipedia articles/wiki_text_debugging.txt','r') as f: 
        article = f.read()
    
    print('-----------------------------------------Lemmatizing')
    # Instantiate the WordNetLemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()
    
    # Lemmatize all tokens in the article
    tokens = [wordnet_lemmatizer.lemmatize(w) for w in word_tokenize(article.lower()) if ((w.isalpha()) & (w not in stopwords.words('english')))]
    
    # Create the bag-of-words: bow
    bow = Counter(tokens)
    
    # Print the 10 most common tokens
    print(f"Wikipedia article's (most common words): \n{bow.most_common(10)}\n")
    
    print('-------------------------------------------Wordcloud')
    # Generate and show the word cloud
    wordcloud = WordCloud(width=1360, height=960,         #Width and height of the canvas.
                          collocations=False,             #Not include biagrams
                          #max_words=100,                 #The maximum number of words.
                          #collocation_threshold=1000,    #score greater than this parameter to be counted as bigrams
                          stopwords={},                   #The words that will be eliminated
                          #min_word_length=6,             #Minimum number of letters a word must have to be included.
                          background_color="black", 
                          colormap='gist_rainbow').generate(' '.join(tokens))
    
    # Create a figure of the generated cloud
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Most Common Lemmatized Words', **title_param)
    
    fig.suptitle(topic, **suptitle_param)
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=.5) #To set the margins 
    plt.show()
    
    
    
def Introduction_to_gensim(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "7. Introduction to gensim"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------------------------Reading the data')
    my_document = ['The movie was about a spaceship and aliens.',
                   'I really liked the movie!',
                   'Awesome action scenes, but boring characters.',
                   'The movie was awful! I hate alien films.',
                   'Space is cool! I liked the movie.',
                   'More space films, please!',]
    print(f'{my_document}\n')
    
    print('------------------------------------------Tokenizing')
    # Instantiate the WordNetLemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()
    
    # Lemmatize all tokens in the article
    tokens = [[wordnet_lemmatizer.lemmatize(my_word) for my_word in word_tokenize(my_line.lower()) if ((my_word.isalpha()) & (my_word not in stopwords.words('english')))] for my_line in my_document]
    print(f'{tokens}\n')
    
    print('-------------------------------------------------BOW')
    # Create a Counter with the lowercase tokens: bow_simple
    bow_simple = [Counter(token) for token in tokens]
    print(f'{bow_simple}\n')
    
    print('-------------------------------------Applying Gensim')
    dictionary = Dictionary(tokens)
    print(f'{dictionary.token2id}\n')
    
    print('----------------------------Creating a gensim corpus')
    # Create a MmCorpus: corpus and order by frequency desc
    #corpora = [dictionary.doc2bow(doc) for doc in tokens]
    corpora = [sorted(dictionary.doc2bow(doc), key=lambda w: w[1], reverse=True) for doc in tokens]
    print(f'{corpora}\n')
    
    print('-----------------------Interpreting the gensim corpus')
    bow_corpora = [{dictionary.get(word_id): word_count for word_id, word_count in corpus} for corpus in corpora]
    print(f'{bow_corpora}\n')
    
    print('-----------------------General Dict of all documents')
    # Create the defaultdict: total_word_count
    total_word_count = defaultdict(int)
    for word_id, word_count in itertools.chain.from_iterable(corpora):
        total_word_count[word_id] += word_count
    
    # Create a sorted list from the defaultdict: sorted_word_count
    sorted_word_count = sorted(total_word_count.items(), key=lambda w: w[1], reverse=True) 
    sorted_word_count = {dictionary.get(word_id): word_count for word_id, word_count in sorted_word_count}
    print(f'{sorted_word_count}')
    
    
    print("****************************************************")
    topic = "11. Tf-idf with gensim"; print("** %s" % topic)
    print("****************************************************")
    print('----------------------------------Getting the Tf-idf')
    tfidf = TfidfModel(corpora)
    print(f'{tfidf}\n')
    
    print('-----------------Transforming a Tf-idf gensim corpus')
    #tfidf_corpora = [tfidf[corpus] for corpus in corpora]
    tfidf_corpora = [sorted(tfidf[corpus], key=lambda w: w[1], reverse=True) for corpus in corpora]
    print(f'{tfidf_corpora}\n')
    
    print('---------------Interpreting the Tf-idf gensim corpus')
    tfidf_corpora_interpreted = [{dictionary.get(word_id): round(word_tfidf, 2) for word_id, word_tfidf in corpus} for corpus in tfidf_corpora]
    print(f'{tfidf_corpora_interpreted}\n')
    
    
    
def Creating_and_querying_a_corpus_with_gensim(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "9. Creating and querying a corpus with gensim"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------------------------Reading the data')
    print(f'Number of articles: {len(articles)}\n')
    
    print('------------------------------------------Tokenizing')
    # Instantiate the WordNetLemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()
    
    # Lemmatize all tokens in the article
    art_tokens = [[wordnet_lemmatizer.lemmatize(word) for word in word_tokenize(article.lower()) if ((word.isalpha()) & (word not in stopwords.words('english')))] for article in articles]
    print(f'Tokens found in each document: {[len(tokens) for tokens in art_tokens]}\n')
    
    print('----------------------------------Infering the title')
    for i, art in enumerate(art_tokens, start=1):
        print(f"Wikipedia article's No. {i} (most common words): \n{Counter(art).most_common(5)}\n")
    
    print('-------------------------------------Applying Gensim')
    # Create a Dictionary from the articles: dictionary
    dictionary = Dictionary(art_tokens)
    #print(f'{dictionary.token2id}\n')
    
    # Select the id for "computer": computer_id
    word_token = "computer"
    word_id = dictionary.token2id.get(word_token)
    
    # Use computer_id with the dictionary to print the word
    print(f'The Id of "{word_token}" is: {word_id}\n')
    
    print('----------------------------Creating a gensim corpus')
    # Create a MmCorpus: corpus and order by frequency desc
    #corpora = [dictionary.doc2bow(tokens) for tokens in art_tokens]
    corpora = [sorted(dictionary.doc2bow(tokens), key=lambda w: w[1], reverse=True) for tokens in art_tokens]
    
    # Print the first 10 word ids with their frequency counts from the fifth document
    print(f'First 10 most frequency words ids from the 5th article: \n{corpora[4][:10]}\n')
    
    print("****************************************************")
    topic = "10. Gensim bag-of-words"; print("** %s" % topic)
    print("****************************************************")
    
    print('-----------------------Gensim - Most Frequency words')
    bow_corpora = [{dictionary.get(word_id): word_count for word_id, word_count in corpus} for corpus in corpora]
    
    for i, corpus in enumerate(bow_corpora, start=1):
        print(f'First 5 most frequency words in article No.{i}: \n{list(itertools.islice(corpus.items(),0,5))}\n')
    
    print('-----------------------General Dict of all documents')
    # Create the defaultdict: total_word_count
    total_word_count = defaultdict(int)
    for word_id, word_count in itertools.chain.from_iterable(corpora):
        total_word_count[word_id] += word_count
    
    # Create a sorted list from the defaultdict: sorted_word_count
    general_bow_corpora = {dictionary.get(word_id): word_count for word_id, word_count in sorted(total_word_count.items(), key=lambda w: w[1], reverse=True)} 
    
    # Print the top 5 words across all documents alongside the count
    print(f'First 5 most frequency words across all articles: \n{list(itertools.islice(general_bow_corpora.items(),0,5))}\n')
    
    
    print("****************************************************")
    topic = "13. Tf-idf with Wikipedia"; print("** %s" % topic)
    print("****************************************************")
    print('-----------------------------Creating the TfidfModel')
    # Create a new TfidfModel using the corpus: tfidf
    tfidf = TfidfModel(corpora)
    print(f'{tfidf}\n')
    
    print('-----------------Transforming a Tf-idf gensim corpus')
    #tfidf_corpora = [tfidf[corpus] for corpus in corpora]
    tfidf_corpora = [{dictionary.get(word_id): round(word_tfidf, 2) for word_id, word_tfidf in sorted(tfidf[corpus], key=lambda w: w[1], reverse=True)} for corpus in corpora]
    for i, corpus in enumerate(tfidf_corpora, start=1):
        print(f'First 5 most frequency words in article No.{i}: \n{list(itertools.islice(corpus.items(),0,5))}\n')
    
    
            
def main():
    print("****************************************************")
    print("** BEGIN                                          **")
    
    Word_counts_with_bag_of_words()
    Building_a_Counter_with_bag_of_words()
    Simple_text_preprocessing()
    Text_preprocessing_practice()
    Introduction_to_gensim()
    Creating_and_querying_a_corpus_with_gensim()
    
    print("****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    pd.reset_option("display.max_columns")
    plt.style.use('default')
    np.set_printoptions(formatter={'float': None})