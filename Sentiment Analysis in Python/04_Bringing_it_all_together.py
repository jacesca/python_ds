# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 4: Let's Predict the Sentiment
    We employ machine learning to predict the sentiment of a review based on 
    the words used in the review. We use logistic regression and evaluate its 
    performance in a few different ways. These are some solid first models!
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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer #Score tfidf result like CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score #Using accuracy score
from sklearn.metrics import classification_report #Precision, recall, f1-score and support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

from wordcloud import WordCloud #Documentation: https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html#wordcloud.WordCloud
from wordcloud import STOPWORDS

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
plot_param     = {'axes.labelsize': 9, 'axes.labelweight': 'bold', 'xtick.labelsize': 9, 'ytick.labelsize': 9, 
                  'legend.fontsize': 8, 'font.size': 8}
cbar_param     = {'fontsize':8, 'labelpad':20, 'color':'maroon'}
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
amazon = pd.read_csv('amazon_reviews_sample.csv', index_col=0) #label: 1-->Positive, 0-->Negative

###############################################################################
## Main part of the code
###############################################################################
print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
topic = "12. Bringing it all together"; print("** %s" % topic)
print("****************************************************")

# Initialize seed and parameters
np.random.seed(SEED) 
DetectorFactory.seed = SEED
    
topic = "13. Step 1: Word cloud and feature creation"; print("** %s" % topic)
print("****************************************************")
# Log the start time
start_time = time.time()

positive_reviews = amazon[amazon.score==1].review.str.cat(sep=' ')
negative_reviews = amazon[amazon.score!=1].review.str.cat(sep=' ')

# Stopwords, 
my_stop_words = sorted(STOPWORDS.union(ENGLISH_STOP_WORDS.union({"lot", "book", "movie", "doe", "read", "need", "Amazon", 'time'})))

print('-------------------------------------------WordCloud')
# Create a figure of the generated cloud
fig, axes = plt.subplots(1, 2, figsize=figsize)
for ax, title, back_color, front_color, reviews in zip(axes, ['Positive','Negative'], ['floralwhite', 'black'], ['gist_rainbow', 'Pastel1'], [positive_reviews, negative_reviews]):
    # Create and generate a word cloud image
    wordcloud = WordCloud(width=680, height=480,          #Width and height of the canvas.
                          collocations=False,             #Not include biagrams
                          max_words=100,                  #The maximum number of words.
                          #collocation_threshold=1000,    #score greater than this parameter to be counted as bigrams
                          stopwords=my_stop_words,        #The words that will be eliminated
                          #min_word_length=6,             #Minimum number of letters a word must have to be included.
                          background_color=back_color, 
                          colormap=front_color).generate(reviews)
    print(f"{title} words: \n{wordcloud.words_.keys()}")  #Word tokens with associated frequency.
    
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Top 100 {title} reviews', **title_param)

fig.suptitle(topic, **suptitle_param)
#plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=.5) #To set the margins 
plt.show()

print('--------------------------------------Len of reviews')
amazon['len_review'] = amazon.review.str.len()

print('--------------------------------------Counting words')
amazon['n_tokens'] = amazon.review.apply(lambda x: len(word_tokenize(x.lower())))

print('----------------------------------Counting sentences')
amazon['n_sentences'] = amazon.review.apply(lambda x: len(sent_tokenize(x.lower())))

print('---------------------------------Detecting languages')
amazon['languages'] = amazon.review.apply(lambda x: str(detect_langs(x.lower())[0]).split(':')[0])
    
print('---------------------------------------------Explore')
print(amazon.head(2))
    
# Log the end time
print(f'\nUsed time: {time.time()-start_time} seg.')



print("****************************************************")
topic = "14. Step 2: Building a vectorizer"; print("** %s" % topic)
print("****************************************************")
# Log the start time
start_time = time.time()

print('-------------------------------------TfidfVectorizer')
# Build the vectorizer
vect = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, 
                       ngram_range=(1, 2), 
                       max_features=200, 
                       token_pattern=r'\b[^\d\W][^\d\W]+\b').fit(amazon.review)

# Create sparse matrix from the vectorizer
X = vect.transform(amazon.review)

# Create a DataFrame
reviews_transformed = pd.DataFrame(data    = X.toarray(), 
                                   columns = vect.get_feature_names(),
                                   index   = amazon.index)

print('-----------------------------------Adding len_review')
reviews_transformed['len_review'] = amazon.len_review

print('---------------------------------------------Explore')
print(f'Top 5 rows of the DataFrame: \n{reviews_transformed.head()}')    

# Log the end time
print(f'\nUsed time: {time.time()-start_time} seg.')



print("****************************************************")
topic = "15. Step 3: Building a classifier"; print("** %s" % topic)
print("****************************************************")
# Log the start time
start_time = time.time()

print('-------------------------Get the train and test data')
# Define X and y
y = amazon.score
X = reviews_transformed

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Group of data
print(amazon.score.value_counts())
    
print('------------------------------------Define the model')
# Define the model to use
#model = LogisticRegression(solver='newton-cg', random_state=seed)
model = LogisticRegression(max_iter=1500)

# Train a logistic regression
model.fit(X_train, y_train)

print('------------------------------------------Validation')
# Accuracy score with train split
y_pred = model.predict(X_train)
print('Accuracy of model in train data: ', accuracy_score(y_train, y_pred))

# Make the predictions
y_pred = model.predict(X_test)
print('Accuracy of model in test data: ', accuracy_score(y_test, y_pred))

print('------------------------------------Confusion Matrix')
# For Confusion matrix plot. 
fig, axes = plt.subplots(1, 2, figsize=figsize, clear=True)
ax = axes[0]
plot_confusion_matrix(model, X_train, y_train, display_labels=['Negative', 'Positive'], 
                      cmap=plt.cm.Blues, normalize='true', values_format='.1%', 
                      ax=ax)
ax.set_title("Amazon Train Data\nConfusion Matrix", **title_param)
ax.grid(False)
ax = axes[1]
plot_confusion_matrix(model, X_test, y_test, display_labels=['Negative', 'Positive'], 
                      cmap=plt.cm.Blues, normalize='true', values_format='.1%', 
                      ax=ax)
ax.set_title("Amazon Test Data\nConfusion Matrix", **title_param)
ax.grid(False)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.3, hspace=None)
fig.suptitle(topic, **suptitle_param)
plt.show()
        
print('-------------------------------------Complete report')
# Precision, recall, f1-score and support
print(f'Complete model evaluation: \n{classification_report(y_test, y_pred, zero_division=1)}')

# Log the end time
print(f'\nUsed time: {time.time()-start_time} seg.')



print("****************************************************")
print("** END                                            **")
print("****************************************************")

pd.reset_option("display.max_columns")
plt.style.use('default')
np.set_printoptions(formatter={'float': None})