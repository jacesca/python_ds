# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 2: Numeric Features from Reviews
    Imagine you are in the shoes of a company offering a variety of products. 
    You want to know which of your products are bestsellers and most of all - 
    why. We embark on step 1 of understanding the reviews of products, using a 
    dataset with Amazon product reviews. To that end, we transform the text 
    into a numeric form and consider a few complexities in the process.
Source: https://learn.datacamp.com/courses/sentiment-analysis-in-python
"""
###############################################################################
## Importing libraries
###############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer

from nltk import word_tokenize #Documentation: https://www.nltk.org/api/nltk.tokenize.html
from nltk import sent_tokenize

from langdetect import detect_langs #Documentation: https://github.com/Mimino666/langdetect
from langdetect import DetectorFactory

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
SIZE           = 1000

# Global configuration
sns.set()
pd.set_option("display.max_columns",24)
plt.rcParams.update(**plot_param)
np.random.seed(SEED)

###############################################################################
## Reading the data
###############################################################################
movies = pd.read_csv('IMDB_sample.csv', index_col=0)
amazon = pd.read_csv('amazon_reviews_sample.csv', index_col=0)
non_english_amazon = pd.read_csv('non-english-amazon-reviews.csv', sep=';')


###############################################################################
## Main part of the code
###############################################################################
def Bag_of_words(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "1. Bag-of-words"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('-----------------------------------------Single text')
    my_string = 'This is the best book ever. I loved the book and highly recommend it!!!'
    
    vect = CountVectorizer()
    vect.fit([my_string])
    X = vect.transform([my_string])
    print(vect.get_feature_names())
    print(X.toarray())
    print(vect.vocabulary_)
    
    
    print('----------------------------------------------Movies')
    vect = CountVectorizer(max_features=size)
    X = vect.fit_transform(movies.review)
    # Transform back to a dataframe, assign column names
    X_df = pd.DataFrame(data=X.toarray(), 
                        columns=vect.get_feature_names())
    print(X_df.head(2))
    
    print('----------------------------------------------Amazon')
    print(amazon.head(2))
    vect = CountVectorizer(max_features=size)
    X = vect.fit_transform(amazon.review)
    # Transform back to a dataframe, assign column names
    X_df = pd.DataFrame(data=X.toarray(), 
                        columns=vect.get_feature_names())
    print(X_df.head(2))
    
    
    
def Your_first_BOW(seed=SEED):
    print("****************************************************")
    topic = "3. Your first BOW"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('---------------------------------------Anna Karenina')
    annak = ['Happy families are all alike;', 
             'every unhappy family is unhappy in its own way']
    
    # Build the vectorizer and fit it
    anna_vect = CountVectorizer()
    anna_vect.fit(annak)
    
    # Create the bow representation
    anna_bow = anna_vect.transform(annak)
    
    # Print the bag-of-words result 
    print(anna_vect.get_feature_names())
    print(anna_bow.toarray())
    
    
    
def BOW_using_product_reviews(seed=SEED):
    print("****************************************************")
    topic = "4. BOW using product reviews"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('----------------------------------------------Amazon')
    # Build the vectorizer, specify max features 
    vect = CountVectorizer(max_features=100)
    # Fit the vectorizer
    vect.fit(amazon.review)
    
    # Transform the review column
    X_review = vect.transform(amazon.review)
    
    # Create the bow representation
    X_df=pd.DataFrame(X_review.toarray(), columns=vect.get_feature_names())
    print(X_df.head(2))
    
    
    
def Specify_token_sequence_length_with_BOW(seed=SEED):
    print("****************************************************")
    topic = "6. Specify token sequence length with BOW"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    amazon_sample = amazon.sample(100, random_state=seed) #Get a small sample to avoid MemoryError: Unable to allocate 24.3 GiB for an array with shape (10000, 326726) and data type int64
    
    print('--------------------------------------Counting words')
    # Build the vectorizer, specify token sequence and fit
    vect = CountVectorizer(ngram_range=(1,2))
    vect.fit(amazon_sample.review)
    
    # Transform the review column
    X_review = vect.transform(amazon_sample.review)
    
    print('----------------------------------------------Bow df')
    # Create the bow representation
    X_df = pd.DataFrame(X_review.toarray(), columns=vect.get_feature_names())
    print(X_df.head(2))
    
    
    
def Size_of_vocabulary_of_movies_reviews(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "7. Size of vocabulary of movies reviews"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    movies_sample = movies.sample(size, random_state=seed) #Get a small sample to avoid MemoryError: Unable to allocate 24.3 GiB for an array with shape (10000, 326726) and data type int64
    
    print('----------------------------------------max_features')
    # Build the vectorizer, specify size of vocabulary and fit
    vect = CountVectorizer(max_features=100)
    vect.fit(movies_sample.review)
    
    # Transform the review column
    X_review = vect.transform(movies_sample.review)
    # Create the bow representation
    X_df = pd.DataFrame(X_review.toarray(), columns=vect.get_feature_names())
    print(X_df.head(2))
    
    
    print('----------------------------------------------max_df')
    # Build and fit the vectorizer
    vect = CountVectorizer(max_df=200)
    vect.fit(movies_sample.review)
    
    # Transform the review column
    X_review = vect.transform(movies_sample.review)
    # Create the bow representation
    X_df = pd.DataFrame(X_review.toarray(), columns=vect.get_feature_names())
    print(X_df.head(2))
    
    
    print('----------------------------------------------min_df')
    # Build and fit the vectorizer
    vect = CountVectorizer(min_df=50)
    vect.fit(movies_sample.review)
    
    # Transform the review column
    X_review = vect.transform(movies_sample.review)
    # Create the bow representation
    X_df = pd.DataFrame(X_review.toarray(), columns=vect.get_feature_names())
    print(X_df.head(2))
    
    
    
def BOW_with_n_grams_and_vocabulary_size(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "8. BOW with n-grams and vocabulary size"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    amazon_sample = amazon.sample(100, random_state=seed) #Get a small sample to avoid MemoryError: Unable to allocate 24.3 GiB for an array with shape (10000, 326726) and data type int64
    
    print('----------------------------------ngram_range=(2, 2)')
    # Build the vectorizer, specify max features and fit
    vect = CountVectorizer(max_features=size, ngram_range=(2, 2), max_df=500)
    vect.fit(amazon_sample.review)
    
    # Transform the review
    X_review = vect.transform(amazon_sample.review)
    
    # Create a DataFrame from the bow representation
    X_df = pd.DataFrame(X_review.toarray(), columns=vect.get_feature_names())
    print(X_df.head())
    
    
    
def Build_new_features_from_text(seed=SEED):
    print("****************************************************")
    topic = "9. Build new features from text"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    amazon_sample = amazon.sample(100, random_state=seed) #Get a small sample to avoid MemoryError: Unable to allocate 24.3 GiB for an array with shape (10000, 326726) and data type int64
    
    print('---------------------------------------Single string')
    anna_k = 'Happy families are all alike, every unhappy families are unhappy in its own way.'
    print(word_tokenize(anna_k))
    
    print('---------------------------------Tokens from a column')
    amazon_sample['tokens'] = amazon_sample.review.apply(lambda x: word_tokenize(x.lower()))
    amazon_sample['n_tokens'] = amazon_sample.review.apply(lambda x: len(word_tokenize(x.lower())))
    
    print(amazon_sample.head(2))
    
    
    
def Tokenize_a_string_from_GoT(seed=SEED):
    print("****************************************************")
    topic = "10. Tokenize a string from GoT"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    GoT = 'Never forget what you are, for surely the world will not. Make it your strength. Then it can never be your weakness. Armour yourself in it, and it will never be used to hurt you.'
    
    print('---------------George R.R. Martin\'s Game of Thrones')
    # Transform the GoT string to word tokens
    print(word_tokenize(GoT))
    
    
    
def Word_tokens_from_the_Avengers(seed=SEED):
    print("****************************************************")
    topic = "11. Word tokens from the Avengers"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    avengers = ["Cause if we can't protect the Earth, you can be d*** sure we'll avenge it",
                'There was an idea to bring together a group of remarkable people, to see if we could become something more',
                "These guys come from legend, Captain. They're basically Gods."]
    
    print('-------------------------------------Avengers movies')
    # Tokenize each item in the avengers 
    tokens_avengers = [word_tokenize(item) for item in avengers]
    
    print(tokens_avengers)
    
    
    
def A_feature_for_the_length_of_a_review(seed=SEED):
    print("****************************************************")
    topic = "12. A feature for the length of a review"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    amazon_sample = amazon.sample(100, random_state=seed) #Get a small sample to avoid MemoryError: Unable to allocate 24.3 GiB for an array with shape (10000, 326726) and data type int64
    
    print('----------------------------------Counting sentences')
    print('------------------------------on the sample data set')
    amazon_sample['sentences'] = amazon_sample.review.apply(lambda x: sent_tokenize(x.lower()))
    amazon_sample['n_sentences'] = amazon_sample.review.apply(lambda x: len(sent_tokenize(x.lower())))
    
    print(amazon_sample.head(2))
    
    print('------------------------Counting words and sentences')
    print('----------------------------on the complete data set')
    #Counting words in each review.
    amazon['n_tokens'] = amazon.review.apply(lambda x: len(word_tokenize(x.lower())))
    amazon['n_sentences'] = amazon.review.apply(lambda x: len(sent_tokenize(x.lower())))
    
    print(amazon.head(2))
    
    
def Can_you_guess_the_language(seed=SEED):
    print("****************************************************")
    topic = "13. Can you guess the language?"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    DetectorFactory.seed = seed
    
    print('---------------------------------------Single string')
    foreign = 'Este libro ha sido uno de los mejores libros que he leido.'
    print(foreign)
    print(detect_langs(foreign))
    print(str(detect_langs(foreign)[0]).split(':')[0])
    
    foreign = 'Por supuesto que Ella es muy bonita, She is beautiful!.'
    print(foreign)
    print(detect_langs(foreign))
    print(str(detect_langs(foreign)[0]).split(':')[0])
    
    print('--------------------------------Working on a dataset')
    amazon['languages'] = amazon.review.apply(lambda x: str(detect_langs(x.lower())[0]).split(':')[0])
    
    print(amazon.head(2),'\n\n')
    print(amazon.languages.value_counts())
    
    
    
def Identify_the_language_of_a_string(seed=SEED):
    print("****************************************************")
    topic = "14. Identify the language of a string"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    DetectorFactory.seed = seed
    
    print('---------------------------------------Single string')
    foreign = 'La histoire rendu étai fidèle, excellent, et grand.'
    
    print(foreign)

    # Detect the language of the foreign string
    print(detect_langs(foreign))
    
    print('---------------------------------------------Explore')
    
    
    
def Detect_language_of_a_list_of_strings(seed=SEED):
    print("****************************************************")
    topic = "15. Detect language of a list of strings"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    DetectorFactory.seed = seed
    
    print('-------------------------------Working with an array')
    sentences = ['La histoire rendu étai fidèle, excellent, et grand.',
                 'Excelente muy recomendable.',
                 'It had a leak from day one but the return and exchange process was very quick.']
    
    print(sentences)
    
    # Loop over the sentences in the list and detect their language
    languages = []
    for sentence in range(len(sentences)):
        languages.append(detect_langs(sentences[sentence]))
    
    print('\nThe detected languages are: ', languages)
    
    
def Language_detection_of_product_reviews(seed=SEED):
    print("****************************************************")
    topic = "16. Language detection of product reviews"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    DetectorFactory.seed = seed
    
    print('--------------------------------Working on a dataset')
    non_english_amazon['languages'] = non_english_amazon.review.apply(lambda x: str(detect_langs(x.lower())[0]).split(':')[0])
    
    print(non_english_amazon.head(2),'\n\n')
    print(non_english_amazon.languages.value_counts())
    
    
        
def main():
    print("****************************************************")
    print("** BEGIN                                          **")
    
    Bag_of_words()
    Your_first_BOW()
    BOW_using_product_reviews()
    Specify_token_sequence_length_with_BOW()
    Size_of_vocabulary_of_movies_reviews()
    BOW_with_n_grams_and_vocabulary_size()
    Build_new_features_from_text()
    Tokenize_a_string_from_GoT()
    Word_tokens_from_the_Avengers()
    A_feature_for_the_length_of_a_review()
    Can_you_guess_the_language()
    Identify_the_language_of_a_string()
    Detect_language_of_a_list_of_strings()
    Language_detection_of_product_reviews()
    
    print("****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    pd.reset_option("display.max_columns")
    plt.style.use('default')
    np.set_printoptions(formatter={'float': None})