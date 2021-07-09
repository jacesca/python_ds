# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 19:40:45 2021

@author: jaces
"""
# Import libraries
import pandas as pd
import numpy as np
import spacy
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Global configuration
SEED = 42
np.random.seed(SEED) 

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')
nlp_es = spacy.load('es_core_news_lg')

# Get list of stopwords in english
stopwords = spacy.lang.en.stop_words.STOP_WORDS

# Read data
movie_overviews = pd.read_csv('data/movie_overviews.csv', index_col=0)
#print(f'Head of movie_overviews: \n{movie_overviews.head()}')

spam = pd.read_csv('data/spam.csv', encoding='utf-8', usecols=['v1', 'v2'])
#print(f'\n\nHead of spam: \n{spam.head()}')


movie_reviews_clean = pd.read_csv('data/movie_reviews_clean.csv')
#print(f'\n\nHead of movie_reviews_clean: \n{movie_reviews_clean.head()}')
#print('Values of sentiment column:', movie_reviews_clean.sentiment.value_counts()) #movie_reviews_clean.sentiment.unique()

# Global functions
def preprocess(text, model=nlp, stopwords=stopwords):
    """Lemmatize a text and return it after cleaning stopwords and not alphanumerics tokens."""
    # Create Doc object without ner and parser
    # ner: EntityRecognizer, parser: owers the sentence boundary detection
    # Return lemmas without stopwords and non-alphabetic characters
    return ' '.join([token.lemma_ for token in model(text.lower(), disable=['ner', 'parser']) 
                     if token.lemma_.isalpha() and token.lemma_ not in stopwords])  

print('*********************************************************')
print('BEGIN')
print('*********************************************************')
print('** 3. N-Gram models')
print('*********************************************************')
print('** 3.1 Building a bag of words model')
print('*********************************************************')
# Bag of words model using sklearn
lcorpus = pd.Series([
'The lion is the king of the jungle',
'Lions have lifespans of a decade',
'The lion is an endangered species'
])
print(f'Corpus to analize: \n{lcorpus}')

# Create CountVectorizer object
vectorizer = CountVectorizer()

# Generate matrix of word vectors
bow_matrix = vectorizer.fit_transform(lcorpus)
features = vectorizer.get_feature_names()
matrix = bow_matrix.toarray()

print(f'\nFeatures: \n{features}')
print(f'\nVectorized data: \n{matrix}')

print('*********************************************************')
print('** 3.2 Word vectors with a given vocabulary')
print('*********************************************************')
print('** 3.3 BoW model for movie taglines')
print('*********************************************************')
# Read data
corpus = movie_overviews[movie_overviews.tagline.notnull()].tagline
print(corpus.head())

# Generate matrix of word vectors
bow_matrix = vectorizer.fit_transform(corpus)
features = vectorizer.get_feature_names()

# Print the features in the bag of words
n = 25
print(f'\nFirst {n} word in bow: \n{features[:n]}')

# Print the shape of bow_matrix
print(f'\nShape of bow: {bow_matrix.shape}')

print('*********************************************************')
print('** 3.4 Analyzing dimensionality and preprocessing')
print('*********************************************************')
# Read data
lem_corpus = corpus.apply(preprocess)
print(lem_corpus.head())

# Generate matrix of word vectors
bow_lem_matrix = vectorizer.fit_transform(lem_corpus)
lem_features = vectorizer.get_feature_names()

# Print the features in the bag of words
n = 25
print(f'\nFirst {n} word in bow: \n{lem_features[:n]}')

# Print the shape of bow_lem_matrix
print(f'\nShape of bow: {bow_lem_matrix.shape}')

print('*********************************************************')
print('** 3.5 Mapping feature indices with feature names')
print('*********************************************************')
# Showing data
print('Corpus: \n{lcorpus}')

# Generate matrix of word vectors
bow_matrix = vectorizer.fit_transform(lcorpus)

# Convert bow_matrix into a DataFrame
bow_df = pd.DataFrame(bow_matrix.toarray())

# Map the column names to vocabulary 
bow_df.columns = vectorizer.get_feature_names()

# Print bow_df
print(bow_df)

print('*********************************************************')
print('** 3.6 Building a BoW Naive Bayes classifier')
print('*********************************************************')
# Read data
df = spam.copy(deep=True)
df.columns = ['label', 'message']
print(df.head())

# Create CountVectorizer object
vectorizer = CountVectorizer(strip_accents='ascii', stop_words='english', lowercase=False)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.25, random_state=SEED)

# Generate training Bow vectors
X_train_bow = vectorizer.fit_transform(X_train)

# Generate test BoW vectors
X_test_bow = vectorizer.transform(X_test)

# Create MultinomialNB object
clf = MultinomialNB()

# Train clf
clf.fit(X_train_bow, y_train)

# Compute accuracy on test set
accuracy = clf.score(X_test_bow, y_test)
print(f'\n\nAccuracy of the created model to detect spam: {accuracy}')

print('*********************************************************')
print('** 3.7 BoW vectors for movie reviews')
print('*********************************************************')
# Create a CountVectorizer object
vectorizer = CountVectorizer(lowercase=True, stop_words='english')

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(movie_reviews_clean['review'], movie_reviews_clean['sentiment'], 
                                                    test_size=0.25, random_state=SEED)

# Fit and transform X_train
X_train_bow = vectorizer.fit_transform(X_train)

# Transform X_test
X_test_bow = vectorizer.transform(X_test)

# Print shape of X_train_bow and X_test_bow
print(X_train_bow.shape)
print(X_test_bow.shape)

print('*********************************************************')
print('** 3.8 Predicting the sentiment of a movie review')
print('*********************************************************')
# Create a MultinomialNB object
clf = MultinomialNB()

# Fit the classifier
clf.fit(X_train_bow, y_train)

# Measure the accuracy
accuracy = clf.score(X_test_bow, y_test)
print("The accuracy of the classifier on the test set is %.3f" % accuracy)

# Predict the sentiment of a negative review
review = "The movie was terrible. The music was underwhelming and the acting mediocre."
prediction = clf.predict(vectorizer.transform([review]))[0]
print("The sentiment predicted by the classifier is %i" % (prediction))

print('*********************************************************')
print('** 3.9 Building n-gram models')
print('*********************************************************')
# Data example
corpus = [
    'The movie was good and not boring',
    'The movie was not good and boring'
]
print(f'Corpus: \n{corpus}')

ngrams_totest = {(1, 1): 'Default', (2, 2): 'Bigrams', (1, 3): 'Unigrams, bigrams and trigrams'}
for ngram in ngrams_totest:
    # Create CountVectorizer object 
    print(f'\nCountVectorizer with ngram_range = {ngram} - {ngrams_totest[ngram]}')
    vectorizer = CountVectorizer(ngram_range=ngram)
    
    # Generate matrix of word vectors
    bow_matrix = vectorizer.fit_transform(corpus)
    features = vectorizer.get_feature_names()
    matrix = bow_matrix.toarray()
    print(f'Features: \n{features}')
    print(f'Vectorized data: \n{matrix} \n{matrix.shape}')

print('*********************************************************')
print('** 3.10 n-gram models for movie tag lines')
print('*********************************************************')
# Read data
corpus = movie_overviews[movie_overviews.tagline.notnull()].tagline
print(corpus.head())

# Generate n-grams upto n=1
vectorizer_ng1 = CountVectorizer(ngram_range=(1,1))
ng1 = vectorizer_ng1.fit_transform(corpus)

# Generate n-grams upto n=2
vectorizer_ng2 = CountVectorizer(ngram_range=(1,2))
ng2 = vectorizer_ng2.fit_transform(corpus)

# Generate n-grams upto n=3
vectorizer_ng3 = CountVectorizer(ngram_range=(1, 3))
ng3 = vectorizer_ng3.fit_transform(corpus)

# Print the number of features for each model
print("\nng1, ng2 and ng3 have %i, %i and %i features respectively" % (ng1.shape[1], ng2.shape[1], ng3.shape[1]))

print('*********************************************************')
print('** 3.11 Higher order n-grams for sentiment analysis')
print('*********************************************************')
# Create a CountVectorizer object
ng_vectorizer = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 2))

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(movie_reviews_clean['review'], movie_reviews_clean['sentiment'], 
                                                    test_size=0.25, random_state=SEED)

# Fit and transform X_train
X_train_ng = ng_vectorizer.fit_transform(X_train)

# Transform X_test
X_test_ng = ng_vectorizer.transform(X_test)

# Print shape of X_train_bow and X_test_bow
print(X_train_bow.shape)
print(X_test_bow.shape)

# Define an instance of MultinomialNB 
clf_ng = MultinomialNB()

# Fit the classifier 
clf_ng.fit(X_train_ng, y_train)

# Measure the accuracy 
accuracy = clf_ng.score(X_test_ng, y_test)
print("The accuracy of the classifier on the test set is %.3f" % accuracy)

# Predict the sentiment of a negative review
review = "The movie was not good. The plot had several holes and the acting lacked panache."
prediction = clf_ng.predict(ng_vectorizer.transform([review]))[0]
print("The sentiment predicted by the classifier is %i" % (prediction))

print('*********************************************************')
print('** 3.12 Comparing performance of n-gram models')
print('*********************************************************')
# Data example
ngrams_totest = {(1, 1): 'Unigram', (1, 3): 'Unigrams, bigrams and trigrams'}
for ngram in ngrams_totest:
    start_time = time.time()
    # Splitting the data into training and test sets
    train_X, test_X, train_y, test_y = train_test_split(movie_reviews_clean['review'], movie_reviews_clean['sentiment'], 
                                                        test_size=0.25, random_state=SEED, 
                                                        stratify=movie_reviews_clean['sentiment'])
    # Generating ngrams
    print(f'\nCountVectorizer with ngram_range = {ngram} - {ngrams_totest[ngram]}')
    vectorizer = CountVectorizer(lowercase=True, stop_words='english', ngram_range=ngram)
    
    # Generate matrix of word vectors
    train_X = vectorizer.fit_transform(train_X)
    test_X = vectorizer.transform(test_X)
    
    # Fit classifier
    clf = MultinomialNB()
    clf.fit(train_X, train_y)
    
    # Print accuracy, time and number of dimensions
    print("""
    The program took %.3f seconds to complete. 
    The accuracy on the test set is %.2f. 
    The ngram representation had %i features.
    """ % (time.time() - start_time, clf.score(test_X, test_y), train_X.shape[1])
    )
    
print('*********************************************************')
print('END')
print('*********************************************************')