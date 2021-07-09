# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 1: Sentiment Analysis Nuts and Bolts
    Have you ever checked the reviews or ratings of a product or a service 
    before you purchased it? Then you have very likely came face-to-face with 
    sentiment analysis. In this chapter, you will learn the basic structure 
    of a sentiment analysis problem and start exploring the sentiment of movie 
    reviews.
Source: https://learn.datacamp.com/courses/sentiment-analysis-in-python
"""
###############################################################################
## Importing libraries
###############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from textblob import TextBlob #For processing textual data. Documentation https://textblob.readthedocs.io/en/dev/

from wordcloud import WordCloud #Documentation: https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html#wordcloud.WordCloud
#from wordcloud import STOPWORDS
from stop_words import get_stop_words #Documentation: https://pypi.org/project/stop-words/
from bs4 import BeautifulSoup #Clean html tag fromt text. Documentation: https://www.crummy.com/software/BeautifulSoup/bs4/doc/

from sklearn.feature_extraction.text import CountVectorizer #Documentation: https://scikit-learn.org/stable/index.html

###############################################################################
## Preparing the environment
###############################################################################
#Global variables
suptitle_param      = dict(color='darkblue', fontsize=11)
title_param         = {'color': 'darkred', 'fontsize': 14, 'weight': 'bold'}
plot_param          = {'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                       'legend.fontsize': 8, 'font.size': 8}
figsize             = (12.1, 5.9)
SEED                = 42
SIZE                = 1000
TOKENS_ALPHANUMERIC = '[A-Za-z0-9\']+(?=\\s+)' # Create the token pattern: TOKENS_ALPHANUMERIC

# Global configuration
sns.set()
pd.set_option("display.max_columns",24)
plt.rcParams.update(**plot_param)
np.random.seed(SEED)

###############################################################################
## Reading the data
###############################################################################
movies = pd.read_csv('IMDB_sample.csv', index_col=0)

###############################################################################
## Main part of the code
###############################################################################
def Welcome(seed=SEED):
    print("****************************************************")
    topic = "1. Welcome!"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------------Explore')
    print(movies.shape)
    print(movies.head())
    
    print('-------------How many positive and negative reviews?')
    print(movies.label.value_counts())
    
    print('---------Percentage of positive and negative reviews')
    print(movies.label.value_counts() / len(movies))
    
    print('---------------------How long is the longest review?')
    movies['length_reviews'] = movies.review.str.len()
    # Finding the review with max length
    print(max(movies.length_reviews))
    
    print('--------------------How long is the shortest review?')
    # Finding the review with max length
    print(min(movies.length_reviews))
    
    
    
def How_many_positive_and_negative_reviews_are_there(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "3. How many positive and negative reviews are there?"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------Movies sample')
    df = movies.sample(size, random_state=seed)
    
    print('-------------Number of positive and negative reviews')
    # Find the number of positive and negative reviews
    print('Number of positive and negative reviews: ')
    print(df.label.value_counts())
    
    print('---------Proportion of positive and negative reviews')
    # Find the proportion of positive and negative reviews
    print('Proportion of positive and negative reviews: ')
    print(df.label.value_counts() / len(df))
    
    
    
def Longest_and_shortest_reviews(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "4. Longest and shortest reviews"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------------------Movies sample')
    df = movies.sample(size, random_state=seed)
    
    print('--------------------------------------Longest review')
    # How long is the longest review
    print(max(df.length_reviews))
    
    print('-------------------------------------Shortest review')
    # How long is the shortest review
    print(min(df.length_reviews))
    
    
    
def Sentiment_analysis_types_and_approaches(seed=SEED):
    print("****************************************************")
    topic = "5. Sentiment analysis types and approaches"; print("** %s" % topic)
    print("****************************************************")
    
    print('------------------------------------------------Data')
    text = "Today was a good day."
    
    print('--------------------------------Working with English')
    txt_en = TextBlob(text)
    print(type(txt_en))
    print(txt_en)
    print('Language: ', txt_en.detect_language())
    print(txt_en.sentiment)
    
    print('--------------------------------Working with Spahish')
    # Working with spanish: https://github.com/sloria/TextBlob/issues/209
    #                       https://stackoverflow.com/questions/38571713/sentiment-analysis-in-spanish-with-google-cloud-natural-language-api
    #                       http://blog.manugarri.com/sentiment-analysis-in-spanish/
    #                       https://kleiber.me/blog/2018/02/25/top-10-python-nlp-libraries-2018/
    #                       http://blog.manugarri.com/sentiment-analysis-in-spanish/
    txt_es = TextBlob("Hoy fue un buen día") 
    print(txt_es)
    print('Language: ', txt_es.detect_language())
    txt_en = txt_es.translate(to='en')
    print(txt_en)
    print(txt_en.sentiment)
    
    
def Detecting_the_sentiment_of_Tale_of_Two_Cities(seed=SEED):
    print("****************************************************")
    topic = "6. Detecting the sentiment of Tale of Two Cities"; print("** %s" % topic)
    print("****************************************************")
    
    print('--------------------------------Working with English')
    two_cities = "It was the best of times, it was the worst of times, it was the age of wisdom, " +\
                 "it was the age of foolishness, it was the epoch of belief, it was the epoch of " +\
                 "incredulity, it was the season of Light, it was the season of Darkness, it was " +\
                 "the spring of hope, it was the winter of despair, we had everything before us, " +\
                 "we had nothing before us, we were all going direct to Heaven, we were all " +\
                 "going direct the other way – in short, the period was so far like the present " +\
                 "period, that some of its noisiest authorities insisted on its being received, " +\
                 "for good or for evil, in the superlative degree of comparison only."
    print(two_cities)
    # Create a textblob object  
    txt_en = TextBlob(two_cities)
    print('Language: ', txt_en.detect_language())
    # Print out the sentiment 
    print(txt_en.sentiment)
    
    
    print('--------------------------------Working with Spahish')
    margarita = "Esto era un rey que tenía " +\
                 "un palacio de diamantes, " +\
                 "una tienda hecha de día " +\
                 "y un rebaño de elefantes, " +\
                 "un kiosko de malaquita, " +\
                 "un gran manto de tisú, " +\
                 "y una gentil princesita, " +\
                 "tan bonita, " +\
                 "Margarita, " +\
                 "tan bonita, como tú."
    txt_es = TextBlob(margarita) 
    print(txt_es)
    print('Language: ', txt_es.detect_language())
    txt_en = txt_es.translate(to='en')
    print(txt_en)
    print(txt_en.sentiment)
    
    
    
def Comparing_the_sentiment_of_two_strings(seed=SEED):
    print("****************************************************")
    topic = "7. Comparing the sentiment of two strings"; print("** %s" % topic)
    print("****************************************************")
    
    print('------------------------------------------------Data')
    annak = 'Happy families are all alike; every unhappy family is unhappy in its own way'
    catcher = "If you really want to hear about it,the first thing you'll probably want to " +\
                 "know is where I was born, and what my lousy childhood was like, and how " +\
                 "my parents were occupied and all before they had me, and all that David " +\
                 "Copperfield kind of crap, but I don't feel like going into it, if you want " +\
                 "to know the truth."    
    
    print('---------------------------------------------Explore')
    # Create a textblob object 
    blob_annak = TextBlob(annak)
    blob_catcher = TextBlob(catcher)
    
    # Print out the sentiment   
    print('Sentiment of annak: ', blob_annak.sentiment)
    print('Sentiment of catcher: ', blob_catcher.sentiment)
    
    
    
def What_is_the_sentiment_of_a_movie_review(seed=SEED):
    print("****************************************************")
    topic = "8. What is the sentiment of a movie review?"; print("** %s" % topic)
    print("****************************************************")
    
    print('------------------------------------------------Data')
    print(movies.loc[movies.review.str.contains('Titanic directed by James Cameron presents a fictional')])
    titanic = movies.loc[movies.review.str.contains('Titanic directed by James Cameron presents a fictional')].iat[0,0]
    
    print('----------------------------------Sentiment analysis')
    # Create a textblob object  
    blob_titanic = TextBlob(titanic)
    
    # Print out its sentiment  
    print(blob_titanic.sentiment)
    
    
    
def Lets_build_a_word_cloud(seed=SEED):
    print("****************************************************")
    topic = "9. Let's build a word cloud!"; print("** %s" % topic)
    print("****************************************************")
    
    # Set the seed for random    
    np.random.seed(seed)
    
    print('--------------------------------------Longest review')
    longest_review = movies.loc[movies.length_reviews.argmax()].review
    wordcloud = WordCloud().generate(longest_review)
    
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Longest Review', **title_param)
    
    fig.suptitle(topic, **suptitle_param)
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=.5) #To set the margins 
    plt.show()
    
    print('------------------------------------------Two cities')
    two_cities = "It was the best of times, it was the worst of times, it was the age of wisdom, " +\
                 "it was the age of foolishness, it was the epoch of belief, it was the epoch of " +\
                 "incredulity, it was the season of Light, it was the season of Darkness, it was " +\
                 "the spring of hope, it was the winter of despair, we had everything before us, " +\
                 "we had nothing before us, we were all going direct to Heaven, we were all " +\
                 "going direct the other way – in short, the period was so far like the present " +\
                 "period, that some of its noisiest authorities insisted on its being received, " +\
                 "for good or for evil, in the superlative degree of comparison only."
    wordcloud = WordCloud().generate(two_cities)
    
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Two cities', **title_param)
    
    fig.suptitle(topic, **suptitle_param)
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=.5) #To set the margins 
    plt.show()
    
    print('---------------------------------------------Titanic')
    titanic = movies.loc[movies.review.str.contains('Titanic directed by James Cameron presents a fictional')].iat[0,0]
    wordcloud = WordCloud(background_color="white").generate(titanic)
    
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Titanic', **title_param)
    
    fig.suptitle(topic, **suptitle_param)
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=.5) #To set the margins 
    plt.show()
    
    
    
def Your_first_word_cloud(seed=SEED):
    print("****************************************************")
    topic = "10. Your first word cloud"; print("** %s" % topic)
    print("****************************************************")
    
    # Set the seed for random    
    np.random.seed(seed)
    
    print('----------------------------------------East of Eden')
    east_of_eden = "I remember my childhood names for grasses and secret flowers. " +\
                 "I remember where a toad may live and what time the birds awaken " +\
                 "in the summer—and what trees and seasons smelled like—how people " +\
                 "looked and walked and smelled even. The memory of odors is very " +\
                 "rich."
    # Generate the word cloud from the east_of_eden string
    wordcloud = WordCloud(background_color="white", colormap='nipy_spectral').generate(east_of_eden)
    
    # Create a figure of the generated cloud
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('East of Eden', **title_param)
    
    fig.suptitle(topic, **suptitle_param)
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=.5) #To set the margins 
    plt.show()
    
    
    
def Which_words_are_in_the_word_cloud(seed=SEED):
    print("****************************************************")
    topic = "11. Which words are in the word cloud?"; print("** %s" % topic)
    print("****************************************************")
    
    print('---------------------------Everything is Illuminated')
    illuminated = "I am not sad, he would repeat to himself over and over, " +\
                 "I am not sad. As if he might one day convince himself or " +\
                 "convince others -- the only thing worse than being sad is " +\
                 "for others to know that you are sad. I am not sad."
    # Generate the word cloud from the east_of_eden string
    wordcloud = WordCloud(background_color="white", colormap='hsv').generate(illuminated)
    
    # Create a figure of the generated cloud
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Everything is Illuminated', **title_param)
    
    fig.suptitle(topic, **suptitle_param)
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=.5) #To set the margins 
    plt.show()
    
    
    
def Word_Cloud_on_movie_reviews(seed=SEED):
    print("****************************************************")
    topic = "12. Word Cloud on movie reviews"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('----------------------------Top 100 positive reviews')
    # Clean the text from html tags
    movies['review'] = movies.review.apply(lambda x: BeautifulSoup(x, 'html.parser').get_text().lower())
    # Set the polarity for each review
    movies['polarity'] = movies.review.apply(lambda x: TextBlob(x).sentiment.polarity)
    
    # Getting the top 100 positive reviews
    movies.sort_values(by='polarity', ascending=False, inplace=True)
    df_top100 = movies.iloc[:100].reset_index(drop=True)
    
    print('----------------------------------Setting stop words')
    #stop_words = get_stop_words('en')
    stop_words = get_stop_words('en') + ['can', 'movie', 'film']
    
    print('---------------------------------Getting the reviews')
    # Making a unique big string
    top100 = df_top100.review.str.cat(sep=' ')
    
    # Counting words
    vectorizer = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)
    word_counts = vectorizer.fit_transform(df_top100.review.values)
    df_words = pd.DataFrame(data=word_counts.toarray(),
                            columns=vectorizer.get_feature_names())
    
    df_words = pd.concat([df_top100, df_words], axis=1, join='inner')
    cols_to_show = np.setdiff1d(vectorizer.get_feature_names(), stop_words)
    print(df_words[cols_to_show].sum().sort_values(ascending=False).head(30))
    
    print('-------------------------------------------WordCloud')
    # Generate the word cloud from the east_of_eden string
    wordcloud = WordCloud(width=680, height=480,          #Width and height of the canvas.
                          collocations=False,             #Not include biagrams
                          #max_words=100,                 #The maximum number of words.
                          #collocation_threshold=1000,    #score greater than this parameter to be counted as bigrams
                          stopwords=set(stop_words),      #The words that will be eliminated
                          #min_word_length=6,             #Minimum number of letters a word must have to be included.
                          background_color="floralwhite", 
                          colormap='gist_rainbow').generate(top100)
    print(wordcloud.words_)  #Word tokens with associated frequency.
    #print(wordcloud.layout_) #Encodes the fitted word cloud. Encodes for each word the string, font size, position, orientation and color.
    # Create a figure of the generated cloud
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Top 100 positive reviews', **title_param)
    
    fig.suptitle(topic, **suptitle_param)
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=.5) #To set the margins 
    plt.show()
    
    
    
def main():
    print("****************************************************")
    print("** BEGIN                                          **")
    
    Welcome()
    How_many_positive_and_negative_reviews_are_there()
    Longest_and_shortest_reviews()
    Sentiment_analysis_types_and_approaches()
    Detecting_the_sentiment_of_Tale_of_Two_Cities()
    Comparing_the_sentiment_of_two_strings()
    What_is_the_sentiment_of_a_movie_review()
    Lets_build_a_word_cloud()
    Your_first_word_cloud()
    Which_words_are_in_the_word_cloud()
    Word_Cloud_on_movie_reviews()
    
    print("****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    pd.reset_option("display.max_columns")
    plt.style.use('default')
    np.set_printoptions(formatter={'float': None})