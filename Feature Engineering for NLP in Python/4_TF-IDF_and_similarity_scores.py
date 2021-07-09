# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:20:18 2021

@author: jaces
"""
# Import libraries
import pandas as pd
import numpy as np
import time
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from pprint import pprint

# Load model
nlp = spacy.load('en_core_web_lg')

# Read data
#ted_talk = pd.read_csv('data/ted.csv')
#print(f'Head of ted_talk: \n{ted_talk.head()}')

movie_overviews = pd.read_csv('data/movie_overviews.csv', index_col=0)
#print(f'Head of movie_overviews: \n{movie_overviews.head()}')

ted_main = pd.read_csv('data/ted_main.csv.zip', compression='zip', index_col='url',
                       usecols=['title', 'name', 'languages', 'published_date', 'event', 'url'])
ted_transcripts = pd.read_csv('data/ted_transcripts.csv.zip', compression='zip', index_col='url',
                              usecols=['transcript', 'url'])

ted_talk = ted_main.join(ted_transcripts, how='right').reset_index()
#print(f'\n\nHead of ted_talk: \n{ted_talk.head()}')

print('*********************************************************')
print('BEGIN')
print('*********************************************************')
print('** 4. TF-IDF and similarity scores')
print('*********************************************************')
print('** 4.1 Building tf-idf document vectors')
print('*********************************************************')
# Bag of words model using sklearn
lcorpus = pd.Series([
'The lion is the king of the jungle',
'Lions have lifespans of a decade',
'The lion is an endangered species'
])
print(f'Corpus to analize: \n{lcorpus}')

# Create TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Generate matrix of word vectors
tfidf_matrix = vectorizer.fit_transform(lcorpus)
features = vectorizer.get_feature_names()
matrix = tfidf_matrix.toarray()

print(f'\nFeatures: \n{features}')
print(f'\nVectorized data: \n{matrix}')

print('*********************************************************')
print('** 4.2 tf-idf weight of commonly occurring words')
print('*********************************************************')
print('** 4.3 tf-idf vectors for TED talks')
print('*********************************************************')
# Read data
ted = ted_talk.transcript.copy(deep = True)

# Create TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Generate matrix of word vectors
tfidf_matrix = vectorizer.fit_transform(ted)

# Print the shape of tfidf_matrix
print(tfidf_matrix.shape)

print('*********************************************************')
print('** 4.4 Cosine similarity')
print('*********************************************************')
# Define two 3-dimensional vectors A and B
A = (4,7,1)
B = (5,2,3)

# Compute the cosine score of A and B
score = cosine_similarity([A], [B])

# Print the cosine score
print(score)

print('*********************************************************')
print('** 4.5 Range of cosine scores')
print('*********************************************************')
print('** 4.6 Computing dot product')
print('*********************************************************')
# Initialize numpy vectors
A = np.array([1, 3])
B = np.array([-2, 2])

# Compute dot product
dot_prod = np.dot(A, B)

# Print dot product
print(dot_prod)

print('*********************************************************')
print('** 4.7 Cosine similarity matrix of a corpus')
print('*********************************************************')
# Read data
corpus = ['The sun is the largest celestial body in the solar system', 
          'The solar system consists of the sun and eight revolving planets', 
          'Ra was the Egyptian Sun God', 
          'The Pyramids were the pinnacle of Egyptian architecture', 
          'The quick brown fox jumps over the lazy dog']

# Initialize an instance of tf-idf Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Generate the tf-idf vectors for the corpus
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# Compute and print the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(cosine_sim)

print('*********************************************************')
print('** 4.8 Building a plot line based recommender')
print('*********************************************************')
# Compute and print the cosine similarity matrix
start_time = time.time()
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(cosine_sim)
print("\nWith TfidfVectorizer, the program took %.8f seconds to complete.\n\n" % (time.time() - start_time))

# Compute and print the cosine similarity matrix
start_time = time.time()
linear_ker = linear_kernel(tfidf_matrix, tfidf_matrix)
print(linear_ker)
print("\nWith cosine_similarity, the program took %.8f seconds to complete.\n\n" % (time.time() - start_time))

assert (cosine_sim == linear_ker).all()

print('*********************************************************')
print('** 4.9 Comparing linear_kernel and cosine_similarity')
print('*********************************************************')
# Read data
corpus = movie_overviews[movie_overviews.overview.notnull()].overview
print(corpus.head())

# Create TfidfVectorizer object
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')

# Generate matrix of word vectors
tfidf_matrix = vectorizer.fit_transform(corpus)

# Print the shape of tfidf_matrix
print(f'\nShape of tfidf_matrix: {tfidf_matrix.shape}\n\n')

methods = {cosine_similarity: 'cosine_similarity', linear_kernel: 'linear_kernel'}
for method in methods:
    # Record start time
    start = time.time()
    
    # Compute cosine similarity matrix
    cosine_sim = method(tfidf_matrix, tfidf_matrix)
    
    # Print cosine similarity matrix
    #print(cosine_sim)
    
    # Print time taken
    print("Time taken in method %s: %s seconds" %  (methods[method], (time.time() - start)))
    
print('*********************************************************')
print('** 4.10 Plot recommendation engine')
print('*********************************************************')
# Function to retrieve movie recomendation
def get_recommendations(title, cosine_sim, indices, metadata):
    """Retrieve movies recomendation."""
    # Get the index of the movie that matches the title
    idx = indices[title]
    #print('\n\nidx:', idx)
    
    # Get the pairwsie similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    #print('\nsim_scores:', cosine_sim)
    
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores for 10 most similar movies
    sim_scores = sim_scores[1:11]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top 10 most similar movies
    return metadata['title'].iloc[movie_indices]


# Read data
metadata = movie_overviews[movie_overviews.overview.notnull()].reset_index()
movie_plots = metadata.overview
indices = metadata.reset_index().set_index('title')['index']
print(f'Metadata: \n{metadata.head()}')
print(f'\n\nMovie_plots: \n{movie_plots.head()}')
print(f'\n\nIndices: \n{indices.head()}')


# Initialize the TfidfVectorizer 
tfidf = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 2))

# Construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(movie_plots)
#print(list(tfidf_matrix.toarray()))

# Generate the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Generate recommendations
movies = ['Toy Story', 'The Dark Knight Rises']
for movie in movies:
    print(f'\n\nBase on movie: "{movie}", our recomendations are:')
    print(get_recommendations(movie, cosine_sim, indices, metadata))


print('*********************************************************')
print('** 4.11 The recommender function')
print('*********************************************************')
# Read data
metadata = movie_overviews[movie_overviews.tagline.notnull()].reset_index(drop=True)[['title', 'tagline']]

# Generate mapping between titles and index
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

print(f'Metadata: \n{metadata.head()}')
print(f'\n\nIndices: \n{indices.head()}')


# Function to retrieve movie recomendation
def get_new_recommendations(title, cosine_sim, indices, metadata, num_recomendation=10):
    """Retrieve movies recomendation."""    # Get index of movie that matches title
    idx = indices[title]

    # Sort the movies based on the similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores for 10 most similar movies
    sim_scores = sim_scores[1:num_recomendation+1]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top 10 most similar movies
    return metadata['title'].iloc[movie_indices]


print('*********************************************************')
print('** 4.12 TED talk recommender')
print('*********************************************************')
# Read data
metadata = ted_talk[ted_talk.transcript.notnull()].reset_index(drop=True)[['title', 'transcript', 'url']]

# Generate mapping between titles and index
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

print(f'Metadata: \n{metadata.head()}')
print(f'\n\nIndices: \n{indices.head()}')

# Initialize the TfidfVectorizer 
tfidf = TfidfVectorizer(stop_words='english')

# Construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(metadata.transcript)

# Generate the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix)
 
# Generate recommendations 
talks = ['5 ways to kill your dreams', 'The mothers who found forgiveness, friendship']
for talk in talks:
    print(f'\n\nBase on ted talk: "{talk}", our recomendations are:')
    print(get_new_recommendations(talk, cosine_sim, indices, metadata))
    
    
print('*********************************************************')
print('** 4.13 Beyond n-grams: word embeddings')
print('*********************************************************')
# Create Doc object
doc = nlp('I am happy')

# Generate word vectors for each token
pprint([token.vector for token in doc])

# Word similarities
doc = nlp("happy joyous sad")

for token1 in doc:
    for token2 in doc:
        print('{:>6} - {:<6} : {}'.format(token1.text, token2.text, token1.similarity(token2)))

# Document similarities
sent1 = nlp("I am happy")
sent2 = nlp("I am sad")
sent3 = nlp("I am joyous")

# Compute similarity between sent1 and sent2
print('{} - {:<11} : {}'.format(sent1, sent2.text, sent1.similarity(sent2)))

# Compute similarity between sent1 and sent3
print('{} - {:} : {:<11}'.format(sent1, sent3.text, sent1.similarity(sent3)))

print('*********************************************************')
print('** 4.14 Generating word vectors')
print('*********************************************************')
# Read data
sent = 'I like apples and oranges'

# Create the doc object
doc = nlp(sent)

# Compute pairwise similarity scores
for token1 in doc:
    for token2 in doc:
        print('{:>8} - {:<8} : {}'.format(token1.text, token2.text, token1.similarity(token2)))
        
print('*********************************************************')
print('** 4.15 Computing similarity of Pink Floyd songs')
print('*********************************************************')
with open('data/mother.dat', 'r', encoding='utf-8') as f: mother = f.read()
with open('data/hopes.dat' , 'r', encoding='utf-8') as f: hopes  = f.read()
with open('data/hey.dat'   , 'r', encoding='utf-8') as f: hey    = f.read()
    
# Create Doc objects
mother_doc = nlp(mother)
hopes_doc = nlp(hopes)
hey_doc = nlp(hey)

# Print similarity between mother and hopes
print('mother - hopes :', mother_doc.similarity(hopes_doc))

# Print similarity between mother and hey
print('mother - hey   :', mother_doc.similarity(hey_doc))

print('*********************************************************')
print('** 4.16 Congratulations!')
print('*********************************************************')
print('END')
print('*********************************************************')