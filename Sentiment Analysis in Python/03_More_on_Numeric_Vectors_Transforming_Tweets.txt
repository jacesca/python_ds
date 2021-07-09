# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 3: More on Numeric Vectors: Transforming Tweets
    This chapter continues the process of understanding product reviews. 
    We will cover additional complexities, especially when working with 
    sentiment analysis data from social media platforms such as Twitter. 
    We will also learn other ways to obtain numeric features from the text.
Source: https://learn.datacamp.com/courses/sentiment-analysis-in-python
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

from bs4 import BeautifulSoup #Clean html tag fromt text. Documentation: https://www.crummy.com/software/BeautifulSoup/bs4/doc/

from wordcloud import WordCloud #Documentation: https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html#wordcloud.WordCloud
from wordcloud import STOPWORDS

from textblob import TextBlob #For processing textual data. Documentation https://textblob.readthedocs.io/en/dev/

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer #Score tfidf result like CountVectorizer

from stop_words import get_stop_words #Documentation: https://pypi.org/project/stop-words/

from nltk import word_tokenize #Documentation: https://www.nltk.org/api/nltk.tokenize.html
from nltk.stem import PorterStemmer#Stemming of strings - English             |
from nltk.stem.snowball import SnowballStemmer #To use foreign language stemmers: Danish, Dutch, English, Finnish, French, German, Hungarian,Italian, Norwegian, Portuguese, Romanian, Russian, Spanish, Swedish
from nltk.stem import WordNetLemmatizer #Lemmatization of a string

import spacy #To use lematize function in other language. Documentation: https://spacy.io/usage/models

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
amazon = pd.read_csv('amazon_reviews_sample.csv', index_col=0)
tweets = pd.read_csv('tweets.csv')
movies = pd.read_csv('IMDB_sample.csv', index_col=0)


###############################################################################
## Main part of the code
###############################################################################
def Stop_words(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "1. Stop words"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    movies_sample = movies.sample(size, random_state=seed) #Get a small sample to avoid MemoryError: Unable to allocate 24.3 GiB for an array with shape (10000, 326726) and data type int64
    
    print('-------------------------------WordCloud & Stopwords')
    # Clean the text from html tags
    movies_sample['review'] = movies_sample.review.apply(lambda x: BeautifulSoup(x.lower(), 'html.parser').get_text())
    # Set the polarity for each review
    movies_sample['polarity'] = movies_sample.review.apply(lambda x: TextBlob(x).sentiment.polarity)
    
    # Getting the top 100 positive reviews
    movies_sample.sort_values(by='polarity', ascending=False, inplace=True)
    df_top100 = movies_sample.iloc[:100].reset_index(drop=True)
    # Making a unique big string
    top100 = df_top100.review.str.cat(sep=' ')
    
    # Define the stopwords list
    my_stopwords = set(STOPWORDS)
    my_stopwords.update(["movie", "movies", "film", "films", "watch"])
    
    # Generate and show the word cloud
    wordcloud = WordCloud(width=1360, height=960,         #Width and height of the canvas.
                          collocations=False,             #Not include biagrams
                          #max_words=100,                 #The maximum number of words.
                          #collocation_threshold=1000,    #score greater than this parameter to be counted as bigrams
                          stopwords=set(my_stopwords),    #The words that will be eliminated
                          #min_word_length=6,             #Minimum number of letters a word must have to be included.
                          background_color="floralwhite", 
                          colormap='gist_rainbow').generate(top100)
    
    # Create a figure of the generated cloud
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Top 100 positive reviews', **title_param)
    
    fig.suptitle(topic, **suptitle_param)
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=.5) #To set the margins 
    plt.show()
    
    print('-------------------------CountVectorizer & Stopwords')
    # Define the set of stop words
    my_stop_words = ENGLISH_STOP_WORDS.union(['film', 'movie', 'cinema', 'theatre'])
    
    vect = CountVectorizer(stop_words=my_stop_words, max_features=100)
    vect.fit(movies_sample.review)
    X = vect.transform(movies_sample.review)
    # Create the bow representation
    X_df = pd.DataFrame(X.toarray(), columns=vect.get_feature_names())
    print(X_df.head(2))
    print(X_df.columns)
    
    
    
def Word_cloud_of_tweets(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "2. Word cloud of tweets"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    tweets_sample = tweets.sample(size, random_state=seed) #Get a small sample to avoid MemoryError: Unable to allocate 24.3 GiB for an array with shape (10000, 326726) and data type int64
    
    print('----------------------------------------------Tweets')
    print(tweets_sample.head(2))
    text_tweet = tweets_sample.text.str.cat(sep=' ')
    
    fig, axes = plt.subplots(2,2, figsize=figsize)
    print('-----------------------------------Without stopwords')
    ax = axes[0,0]
    # Generate the word cloud
    wordcloud = WordCloud(width=1360, height=960, collocations=False).generate(text_tweet)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Without Stopwords', **title_param)
    
    print('---------------------------------wordcloud.STOPWORDS')
    ax = axes[0,1]
    # Define and update the list of stopwords
    my_stopwords = set(STOPWORDS)
    my_stopwords.update(['airline', 'airplane'])
    # Generate the word cloud
    wordcloud = WordCloud(width=1360, height=960,         #Width and height of the canvas.
                          collocations=False,             #Not include biagrams
                          #max_words=100,                 #The maximum number of words.
                          #collocation_threshold=1000,    #score greater than this parameter to be counted as bigrams
                          stopwords=set(my_stopwords),    #The words that will be eliminated
                          #min_word_length=6,             #Minimum number of letters a word must have to be included.
                          background_color="floralwhite", 
                          colormap='gist_rainbow').generate(text_tweet)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Using wordcloud.STOPWORDS', **title_param)
    
    print('--------------------------sklearn-ENGLISH_STOP_WORDS')
    ax = axes[1,0]
    # Define and update the list of stopwords
    my_stopwords = ENGLISH_STOP_WORDS.union({'airline', 'airplane'})
    # Generate the word cloud
    wordcloud = WordCloud(width=1360, height=960,         #Width and height of the canvas.
                          collocations=False,             #Not include biagrams
                          #max_words=100,                 #The maximum number of words.
                          #collocation_threshold=1000,    #score greater than this parameter to be counted as bigrams
                          stopwords=set(my_stopwords),    #The words that will be eliminated
                          #min_word_length=6,             #Minimum number of letters a word must have to be included.
                          background_color="floralwhite", 
                          colormap='gist_rainbow').generate(text_tweet)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Using sklearn-ENGLISH_STOP_WORDS', **title_param)
    
    print('------------------------------------------stop_words')
    ax = axes[1,1]
    # Define and update the list of stopwords
    my_stopwords = {'airline', 'airplane'}.union(get_stop_words('en'))
    # Generate the word cloud
    wordcloud = WordCloud(width=1360, height=960,         #Width and height of the canvas.
                          collocations=False,             #Not include biagrams
                          #max_words=100,                 #The maximum number of words.
                          #collocation_threshold=1000,    #score greater than this parameter to be counted as bigrams
                          stopwords=set(my_stopwords),    #The words that will be eliminated
                          #min_word_length=6,             #Minimum number of letters a word must have to be included.
                          background_color="floralwhite", 
                          colormap='gist_rainbow').generate(text_tweet)
    
    # Create a figure of the generated cloud
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Using stop_words module', **title_param)
    
    fig.suptitle(topic, **suptitle_param)
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=.5) #To set the margins 
    plt.show()
    
    
    
def Airline_sentiment_with_stop_words(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "3. Airline sentiment with stop words"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    tweets_sample = tweets.sample(size, random_state=seed) #Get a small sample to avoid MemoryError: Unable to allocate 24.3 GiB for an array with shape (10000, 326726) and data type int64
    
    print('------------------------------------------Tweets BOW')
    # Define the stop words
    my_stop_words = ENGLISH_STOP_WORDS.union({'airline', 'airlines', '@'})
    
    # Build and fit the vectorizer
    vect = CountVectorizer(stop_words=my_stop_words)
    vect.fit(tweets_sample.text)
    
    # Create the bow representation
    bow_tweet = vect.transform(tweets_sample.text)
    # Create the data frame
    bow_df = pd.DataFrame(bow_tweet.toarray(), columns=vect.get_feature_names())
    print(bow_df.head())
    print(bow_df.columns)
    print(bow_df.sum().sort_values(ascending=False))

    
    
def Multiple_text_columns(size=1259, seed=SEED):
    print("****************************************************")
    topic = "4. Multiple text columns"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    tweets_sample = tweets[tweets.negativereason.notnull()].sample(size, random_state=seed) #Get a small sample to avoid MemoryError: Unable to allocate 24.3 GiB for an array with shape (10000, 326726) and data type int64
    #tweets_sample.negativereason.fillna('', inplace=True)
    
    print('-------------------------------------CountVectorizer')
    # Define the stop words
    my_stop_words = ENGLISH_STOP_WORDS.union(['airline', 'airlines', '@', 'am', 'pm'])
    
    # Build and fit the vectorizers
    vect1 = CountVectorizer(stop_words=my_stop_words)
    vect2 = CountVectorizer() 
    vect1.fit(tweets_sample.text)
    vect2.fit(tweets_sample.negativereason)
    
    # Print the last 15 features from the first, and all from second vectorizer
    print(vect1.get_feature_names()[-15:])
    print(vect2.get_feature_names())
    
    
    
def Capturing_a_token_pattern(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "5. Capturing a token pattern"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('--------------------String operators and comparisons')
    df = pd.DataFrame({'my_string': ['carambola', '2020', 'AX100']})
    
    df['isalpha()'] = df.my_string.str.isalpha()
    df['isdigit()'] = df.my_string.str.isdigit()
    df['isalnum()'] = df.my_string.str.isalnum()
        
    print(df)
    
    print('--------------------String operators and numpy array')
    my_array = np.array(['carambola', '2020', 'AX100'])
    print(my_array)
    print('Only letters: ', my_array[np.char.isalpha(my_array)])
    
    print('---------String operators applied to CountVectorizer')
    amazon['n_tokens1'] = amazon.review.apply(lambda x: len(word_tokenize(x.lower())))
    amazon['n_tokens2'] = amazon.review.apply(lambda x: len([word for word in word_tokenize(x.lower()) if word.isalpha()]))
    print(amazon[['n_tokens1','n_tokens2']].head())
    
    print('---------------------------------Regular expressions')
    my_string = "I @think the #bestyear was 2000, memories@family.org"
    print(my_string)
    
    pattern = r'[A-Za-z]+' #Only letters
    print(f'pattern "{pattern}":', re.findall(pattern, my_string))
    
    pattern = r'#[A-Za-z]' #string compoused for letters after # found
    print(f'pattern "{pattern}":', re.findall(pattern, my_string))
    
    pattern = r'\b\w+\b' #Alfanumerics between end words identified (space, points, etc.)
    print(f'pattern "{pattern}":', re.findall(pattern, my_string))
    
    pattern = r'\b\w\w+\b' #words with at least 2 alfanumerics compoused
    print(f'pattern "{pattern}":', re.findall(pattern, my_string))
    
    pattern = r'\b[^\d\W][^\d\W]+\b' #words without numbers
    print(f'pattern "{pattern}":', re.findall(pattern, my_string))
    
    #words that begins with @
    my_string = "@Julieta find @Gabriela eating her icecream, call for help to @mama, problems@family.org"
    pattern =  r'\s*@\w+\b' 
    print(my_string)
    print(f'pattern "{pattern}":', re.findall(pattern, my_string))
    
    print('-------------------Replacing with regular expressions')
    original_text = "@RayFranco is answering to @jjconti, this is a real '@username83' but this is an@email.com, and this is a @probablyfaketwitterusername"
    pattern = r'(^|[^@\w])@(\w{1,15})\b'
    new_text = re.sub(pattern, '\\1<a href="http://twitter.com/\\2">\\2</a>', original_text)
    
    print(f'Original text: \n{original_text}\n')
    print(f'Pattern: {pattern}\n')
    print(f'New text: \n{new_text}')
    
    print('-------------------Token pattern with CountVectorizer')
    tweets_sample = tweets.sample(size, random_state=seed) #Get a small sample to avoid MemoryError: Unable to allocate 24.3 GiB for an array with shape (10000, 326726) and data type int64
    my_stop_words = ENGLISH_STOP_WORDS.union({'airline', 'airlines', '@'})
    token_pattern = r'\b[^\d\W][^\d\W]+\b'
    
    # Build and fit the vectorizers
    vect1 = CountVectorizer(stop_words=my_stop_words)
    vect2 = CountVectorizer(stop_words=my_stop_words, token_pattern = token_pattern) 
    vect1.fit(tweets_sample.text)
    vect2.fit(tweets_sample.text)
    
    # Print the last 15 features from the first, and all from second vectorizer
    print('With default token_pattern:', np.sort(vect1.get_feature_names()))
    print(f'With token_pattern {token_pattern}:', np.sort(vect2.get_feature_names()))
    
    
    
def Specify_the_token_pattern(seed=SEED):
    print("****************************************************")
    topic = "6. Specify the token pattern"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('-----------------------------Setting CountVectorizer')
    token_pattern2 = r'\b[^A-Za-z][^A-Za-z]+\b'
    token_pattern3 = r'\b[A-Za-z][A-Za-z]+\b'
    
    # Build and fit the vectorizers
    vect1 = CountVectorizer().fit(tweets.text)
    vect2 = CountVectorizer(token_pattern = token_pattern2).fit(tweets.text)
    vect3 = CountVectorizer(token_pattern = token_pattern3).fit(tweets.text)
    
    # Print the last 15 features from the first, and all from second vectorizer
    print('With default token_pattern:', len(vect1.get_feature_names()))
    print(np.sort(vect1.get_feature_names()))
    print(f'With token_pattern {token_pattern2}:', len(vect2.get_feature_names()))
    print(np.sort(vect2.get_feature_names()))
    print(f'With token_pattern {token_pattern3}:', len(vect3.get_feature_names()))
    print(np.sort(vect3.get_feature_names()))
    
    
    
def String_operators_with_the_Twitter_data(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "7. String operators with the Twitter data"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    tweets_sample = tweets.sample(size, random_state=seed) #Get a small sample to avoid MemoryError: Unable to allocate 24.3 GiB for an array with shape (10000, 326726) and data type int64
    
    print('------------------------------------String operators')
    print('Original text: ', tweets_sample.iloc[0].text)
    
    tweets_sample['word_tokens'] = tweets_sample.text.apply(lambda x: word_tokenize(x.lower()))
    print('Original tokens: ', tweets_sample.iloc[0].word_tokens)
    
    tweets_sample['cleaned_tokens'] = tweets_sample.text.apply(lambda x: [word for word in word_tokenize(x.lower()) if word.isalpha()])
    print('Cleaned tokens: ', tweets_sample.iloc[0].cleaned_tokens)
    
    print('-----------------------------Finding user referenced')
    pattern = r'\s*@[A-Za-z0-9]+\s*'
    tweets_sample['user_referenced'] = tweets_sample.text.apply(lambda x: re.findall(pattern, x))
    print('User referenced: ', tweets_sample.iloc[0].user_referenced)
    print(f'\n\nSample df: \n{tweets_sample.user_referenced}')
    #print(f'\n\nOnly rowe with 2 or mores user referenced: \n{tweets_sample[tweets_sample.user_referenced.str.len() > 1]}')
    tweets_sample['n_user_referenced'] = tweets_sample.user_referenced.apply(len)
    cols_to_show = ['airline_sentiment', 'text', 'user_referenced']
    print(f'\n\nOnly rows with 2 or mores user referenced: \n{tweets_sample[tweets_sample.n_user_referenced > 1][cols_to_show]}')
    
    
    
def More_string_operators_and_Twitter(seed=SEED):
    print("****************************************************")
    topic = "8. More string operators and Twitter"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('----------------------------------Working with lists')
    tweets_list = ["@VirginAmerica it's really aggressive to blast obnoxious 'entertainment' in your guests' faces &amp; they have little recourse",
                   "@VirginAmerica Hey, first time flyer next week - excited! But I'm having a hard time getting my flights added to my Elevate account. Help?", 
                   '@united Change made in just over 3 hours. For something that should have taken seconds online, I am not thrilled. Loved the agent, though.']
    
    
    # Create a list of lists, containing the tokens from list_tweets
    tokens = [word_tokenize(item) for item in tweets_list]
    
    # Remove characters and digits , i.e. retain only letters
    letters = [[word for word in item if word.isalpha()] for item in tokens]
    # Remove characters, i.e. retain only letters and digits
    let_digits = [[word for word in item if word.isalnum()] for item in tokens]
    # Remove letters and characters, retain only digits
    digits = [[word for word in item if word.isdigit()] for item in tokens]
    
    # Print the last item in each list
    print('\nLast item in alphabetic list: ', letters[2])
    print('\nLast item in list of alphanumerics: ', let_digits[2])
    print('\nLast item in the list of digits: ', digits[2])
    
    
    
def Stemming_and_lemmatization(seed=SEED):
    print("****************************************************")
    topic = "9. Stemming and lemmatization"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('-------------------------English Stemming of strings')
    porter = PorterStemmer()
    my_string = 'wonderful'
    stem_string = porter.stem(my_string)
    print(f'The Stemming of "{my_string}" is: "{stem_string}".\n')
    
    print('--------------------------------Non-English stemmers')
    foreign_string = {'dutch':'beginen', 'spanish':'zapatería'}
    for str_lang in foreign_string:
        ForeignStemmer = SnowballStemmer(str_lang)
        stem_string = ForeignStemmer.stem(foreign_string[str_lang])
        print(f"The Stemming of '{foreign_string[str_lang]}' is: '{stem_string}'")
    
    print('\n------------------How to stem a sentence in English?')
    my_sentence = 'Today is a wonderful day!'
    tokens = word_tokenize(my_sentence)
    stemmed_tokens = [porter.stem(token) for token in tokens]
    print(f'Sentence: "{my_sentence}"')
    print('Tokens: ', tokens)
    print('Stemming:', stemmed_tokens)
    
    print('\n--------How to stem a sentence in another languages?')
    foreign_sentences = {'french': 'hui est une merveilleuse journée!', 'spanish':'¡hoy es un día maravilloso!'}
    for str_lang in foreign_sentences:
        ForeignStemmer = SnowballStemmer(str_lang)
        tokens = word_tokenize(foreign_sentences[str_lang])
        stemmed_tokens = [ForeignStemmer.stem(token) for token in tokens]
        print(f'Sentence: "{foreign_sentences[str_lang]}"')
        print('Tokens: ', tokens)
        print('Stems:', stemmed_tokens,'\n')
    
    print('--------------------English Lemmatization of strings')
    WNlemmatizer = WordNetLemmatizer()
    my_string = 'wonderful'
    lem_string = WNlemmatizer.lemmatize(my_string, pos='a')
    print(f'The lemmatization of "{my_string}" is: "{lem_string}".\n')
    
    print('--------------------Lemmatization in other languages')
    nlp = spacy.load('es_core_news_sm')
    
    my_sentence = "Con estos fines, la Dirección de Gestión y Control Financiero monitorea la posición de capital del Banco y utiliza los mecanismos para hacer un eficiente manejo del capital."
    print(f"Sentences: {my_sentence}\n")
    
    nlp_sentence = nlp(my_sentence)
    df = pd.DataFrame({'text' : [token.text   for token in nlp_sentence],
                       'lemma': [token.lemma_ for token in nlp_sentence],
                       'pos'  : [token.pos_   for token in nlp_sentence]})
    print(df)
    
    
    
def Stems_and_lemmas_from_GoT(seed=SEED):
    print("****************************************************")
    topic = "10. Stems and lemmas from GoT"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('--------------------------------------List of tokens')
    GoT = 'Never forget what you are, for surely the world will not. ' +\
          'Make it your strength. Then it can never be your weakness. ' +\
          'Armour yourself in it, and it will never be used to hurt you.'
    print(GoT)

    porter = PorterStemmer()
    WNlemmatizer = WordNetLemmatizer()

    # Tokenize the GoT string
    tokens = word_tokenize(GoT) 
    
    print('---------------------------------------------Stemmer')
    # Log the start time
    start_time = time.time()
    
    # Build a stemmed list
    stemmed_tokens = [porter.stem(token) for token in tokens] 
    
    # Log the end time
    end_time = time.time()
    
    print('Time taken for stemming in seconds: ', end_time - start_time)
    print('Stemmed tokens: ', stemmed_tokens) 
    
    print('------------------------------------------Lemmatizer')
    # Log the start time
    start_time = time.time()
    
    # Build a lemmatized list
    lem_tokens = [WNlemmatizer.lemmatize(token) for token in tokens]
    
    # Log the end time
    end_time = time.time()
    
    print('Time taken for lemmatizing in seconds: ', end_time - start_time)
    print('Lemmatized tokens: ', lem_tokens) 
    
    
    
def Stem_Spanish_reviews(size=2000, seed=SEED):
    print("****************************************************")
    topic = "11. Stem Spanish reviews"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    DetectorFactory.seed = seed
    amazon_sample = amazon.sample(size, random_state=seed) #Get a small sample to avoid MemoryError: Unable to allocate 24.3 GiB for an array with shape (10000, 326726) and data type int64
    
    print('------------------------------------Languages detect')
    # Log the start time
    start_time = time.time()
    
    amazon_sample['languages'] = amazon_sample.review.apply(lambda x: str(detect_langs(x.lower())[0]).split(':')[0])
    amazon_sample = amazon_sample[amazon_sample.languages == 'es']
    # Log the end time
    end_time = time.time()
    
    print(amazon_sample)
    print('Used time: {} ({} processed rows)'.format(end_time - start_time, size))
    
    print('---------------------------------------------Stemmer')
    # Import the Spanish SnowballStemmer
    SpanishStemmer = SnowballStemmer("spanish")
    
    # Create a list of tokens
    amazon_sample['tokens'] = amazon_sample.review.apply(lambda x: word_tokenize(x.lower()))
    amazon_sample['cleaned_tokens'] = amazon_sample.review.apply(lambda x: [word for word in word_tokenize(x.lower()) if word.isalpha()])
    
    # Log the start time
    start_time = time.time()
    
    amazon_sample['stemmed_tokens'] = amazon_sample.review.apply(lambda x: [SpanishStemmer.stem(word) for word in word_tokenize(x.lower()) if word.isalpha()])
    
    # Log the end time
    end_time = time.time()
    print('Used time: {} ({} processed rows)'.format(end_time - start_time, len(amazon_sample)))
    
    print('---------------------------------------Lemmatization')
    # Log the start time
    start_time = time.time()
    
    nlp = spacy.load('es_core_news_sm')
    
    amazon_sample['lematized_tokens'] = amazon_sample.review.apply(lambda x: [word.lemma_ for word in nlp(x.lower()) if str(word).isalpha()])
    
    # Log the end time
    end_time = time.time()
    print('Used time: {} ({} processed rows)'.format(end_time - start_time, len(amazon_sample)))
    
    # Print the head of the results
    cols_to_show = ['review', 'tokens', 'cleaned_tokens', 'stemmed_tokens', 'lematized_tokens']
    print(amazon_sample[cols_to_show])

    
    
def Stems_from_tweets(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "12. Stems from tweets"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    DetectorFactory.seed = seed
    amazon_sample = amazon.sample(size, random_state=seed) #Get a small sample to avoid MemoryError: Unable to allocate 24.3 GiB for an array with shape (10000, 326726) and data type int64
    
    print('------------------------------------Languages detect')
    # Log the start time
    start_time = time.time()
    
    amazon_sample['languages'] = amazon_sample.review.apply(lambda x: str(detect_langs(x.lower())[0]).split(':')[0])
    amazon_sample = amazon_sample[amazon_sample.languages == 'en']
    
    # Log the end time
    end_time = time.time()
    
    print(amazon_sample.head(2))
    print('Used time: {} ({} processed rows)'.format(end_time - start_time, size))
    
    print('---------------------------------------------Stemmer')
    # Import the Spanish SnowballStemmer
    EnglishStemmer = PorterStemmer()
    
    # Create a list of tokens
    amazon_sample['tokens'] = amazon_sample.review.apply(lambda x: word_tokenize(x.lower()))
    amazon_sample['cleaned_tokens'] = amazon_sample.review.apply(lambda x: [word for word in word_tokenize(x.lower()) if word.isalpha()])
    
    # Log the start time
    start_time = time.time()
    
    amazon_sample['stemmed_tokens'] = amazon_sample.review.apply(lambda x: [EnglishStemmer.stem(word) for word in word_tokenize(x.lower()) if word.isalpha()])
    
    # Log the end time
    end_time = time.time()
    print('Used time: {} ({} processed rows)'.format(end_time - start_time, len(amazon_sample)))
    
    print('---------------------------------------Lemmatization')
    # Log the start time
    start_time = time.time()
    
    WNlemmatizer = WordNetLemmatizer()

    amazon_sample['lematized_tokens'] = amazon_sample.review.apply(lambda x: [WNlemmatizer.lemmatize(word) for word in word_tokenize(x.lower()) if str(word).isalpha()])
    
    # Log the end time
    end_time = time.time()
    print('Used time: {} ({} processed rows)'.format(end_time - start_time, len(amazon_sample)))
    
    # Print the head of the results
    cols_to_show = ['review', 'tokens', 'cleaned_tokens', 'stemmed_tokens', 'lematized_tokens']
    print(amazon_sample[cols_to_show].head(2))
    
    
    
def TfIdf_More_ways_to_transform_text(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "13. TfIdf: More ways to transform text"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    DetectorFactory.seed = seed
    tweets_sample = tweets.sample(size, random_state=seed) #Get a small sample to avoid MemoryError: Unable to allocate 24.3 GiB for an array with shape (10000, 326726) and data type int64
    print(tweets_sample.head(2))    
    
    print('---Tfldf - Term Frequency Inverse Document Frequency')
    # Log the start time
    start_time = time.time()
    
    vect = TfidfVectorizer(max_features=100).fit(tweets_sample.text)
    X = vect.transform(tweets_sample.text)
    
    # Log the end time
    end_time = time.time()
    print('Used time: {} ({} processed rows)'.format(end_time - start_time, len(tweets_sample)))
    
    X_df = pd.DataFrame(data    = X.toarray(), 
                        columns = vect.get_feature_names(),
                        index   = tweets_sample.index)
    print(X_df.head(2))    
    
    
def Your_first_TfIdf(seed=SEED):
    print("****************************************************")
    topic = "14. Your first TfIdf"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('-----------------------------------Defining the data')
    annak = ['Happy families are all alike;', 'every unhappy family is unhappy in its own way']
    
    print('-------------------------------------TfidfVectorizer')
    # Call the vectorizer and fit it
    anna_vect = TfidfVectorizer().fit(annak)
    
    # Create the tfidf representation
    anna_tfidf = anna_vect.transform(annak)
    
    # Print the result 
    X_df = pd.DataFrame(data    = anna_tfidf.toarray(), 
                        columns = anna_vect.get_feature_names())
    # Log the end time
    print(X_df)    
    
    
def TfIdf_on_Twitter_airline_sentiment_data(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "15. TfIdf on Twitter airline sentiment data"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    DetectorFactory.seed = seed
    tweets_sample = tweets.sample(size, random_state=seed) #Get a small sample to avoid MemoryError: Unable to allocate 24.3 GiB for an array with shape (10000, 326726) and data type int64
    print(tweets_sample.head(2))    
    
    print('---Tfldf - Term Frequency Inverse Document Frequency')
    # Log the start time
    start_time = time.time()
    
    # Define the stop words
    my_stop_words = ENGLISH_STOP_WORDS.union(['airline', 'airlines', 'airplane', 'am', 'pm', '@'])

    # Define the vectorizer and specify the arguments
    my_pattern = r'\b[^\d\W][^\d\W]+\b'    
    vect = TfidfVectorizer(ngram_range=(1, 2), max_features=100, token_pattern=my_pattern, stop_words=my_stop_words).fit(tweets_sample.text)
    
    # Transform the vectorizer
    X = vect.transform(tweets_sample.text)

    # Log the end time
    end_time = time.time()
    
    # Log the end time
    end_time = time.time()
    print('Used time: {} ({} processed rows)'.format(end_time - start_time, len(tweets_sample)))
    
    # Transform to a data frame and specify the column names
    X_df = pd.DataFrame(data    = X.toarray(), 
                        columns = vect.get_feature_names(),
                        index   = tweets_sample.index)
    print(X_df.head(2))
    
    
def Tfidf_and_a_BOW_on_same_data(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "16. Tfidf and a BOW on same data"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    DetectorFactory.seed = seed
    amazon_sample = amazon.sample(size, random_state=seed) #Get a small sample to avoid MemoryError: Unable to allocate 24.3 GiB for an array with shape (10000, 326726) and data type int64
    print(amazon_sample.head(2))    
    
    print('-----------------------------------------Tfldf & BOW')
    # Define the stop words
    my_stop_words = ENGLISH_STOP_WORDS.union(['airline', 'airlines', 'airplane', 'am', 'pm', '@'])
    my_pattern = r'\b[^\d\W][^\d\W]+\b'    
    
    # Build a BOW and tfidf vectorizers from the review column and with max of 100 features
    vect1 = CountVectorizer(max_features=100, token_pattern=my_pattern, stop_words=my_stop_words).fit(amazon_sample.review)
    vect2 = TfidfVectorizer(max_features=100, token_pattern=my_pattern, stop_words=my_stop_words).fit(amazon_sample.review) 
    
    # Transform the vectorizers
    X1 = vect1.transform(amazon_sample.review)
    X2 = vect2.transform(amazon_sample.review)
    
    # Create DataFrames from the vectorizers 
    X_df1 = pd.DataFrame(X1.toarray(), columns=vect1.get_feature_names())
    X_df2 = pd.DataFrame(X2.toarray(), columns=vect2.get_feature_names())
    
    print('Top 5 rows using BOW: \n', X_df1.head(2))
    print('Top 5 rows using tfidf: \n', X_df2.head(2))
    
    
    
def main():
    print("****************************************************")
    print("** BEGIN                                          **")
    
    Stop_words()
    Word_cloud_of_tweets()
    Airline_sentiment_with_stop_words()
    Multiple_text_columns()
    Capturing_a_token_pattern()
    Specify_the_token_pattern()
    String_operators_with_the_Twitter_data()
    More_string_operators_and_Twitter()
    Stemming_and_lemmatization()
    Stems_and_lemmas_from_GoT()
    Stem_Spanish_reviews()
    Stems_from_tweets()
    TfIdf_More_ways_to_transform_text()
    Your_first_TfIdf()
    TfIdf_on_Twitter_airline_sentiment_data()
    Tfidf_and_a_BOW_on_same_data()
    
    print("****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    pd.reset_option("display.max_columns")
    plt.style.use('default')
    np.set_printoptions(formatter={'float': None})