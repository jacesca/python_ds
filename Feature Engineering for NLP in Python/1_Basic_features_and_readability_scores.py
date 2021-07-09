# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 20:19:57 2021

@author: jaces
"""
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from textatistic import Textatistic


# Global configuration
suptitle_param = dict(color='darkblue', fontsize=11)
title_param    = {'color': 'darkred', 'fontsize': 14, 'weight': 'bold'}
plot_param     = {'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                  'legend.fontsize': 8, 'font.size': 8}
figsize        = (12.1, 5.9)
SEED = 42
np.random.seed(SEED) 
sns.set()

plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                     'legend.fontsize': 8, 'font.size': 8})
suptitle_param   = dict(color='darkblue', fontsize=10)
title_param      = {'color': 'darkred', 'fontsize': 12}

# Read data
russian_tweets = pd.read_csv('data/russian_tweets.csv', index_col=0)
print(f'Head of russian_tweets: \n{russian_tweets.head()}')

ted_talk = pd.read_csv('data/ted.csv')
print(f'Head of ted_talk: \n{ted_talk.head()}')


print('*********************************************************')
print('BEGIN')
print('*********************************************************')
print('** 1. Basic features and readability scores')
print('*********************************************************')
print('** 1.1 Introduction to NLP feature engineering')
print('*********************************************************')
df = pd.DataFrame({'sex': ['female', 'male', 'female', 'male', 'female']})
print(df)

# Perform one-hot encoding on the 'sex' feature of df
df = pd.get_dummies(df, columns=['sex'])
print(df)

print('*********************************************************')
print('** 1.2 Data format for ML algorithms')
print('*********************************************************')
print('** 1.3 One-hot encoding')
print('*********************************************************')
df1 = pd.DataFrame({'feature 1': np.random.uniform(low=0, high=80, size=(15,)),
                    'feature 2': np.random.randint(2, size=15),
                    'feature 3': np.random.choice([0, 2], size=15),
                    'feature 4': np.random.uniform(low=0, high=230, size=(15,)),
                    'feature 5': np.random.choice(['female', 'male'], size=15),
                    'label'    : np.random.randint(2, size=15)
                   })
print(df1)

# Print the features of df1
print(df1.columns)

# Perform one-hot encoding
df1 = pd.get_dummies(df1, columns=['feature 5'])

# Print the new features of df1
print(df1.columns)

# Print first five rows of df1
print(df1.head())

print('*********************************************************')
print('** 1.4 Basic feature extraction')
print('*********************************************************')
#tweets = russian_tweets.copy(deep=True)
df = pd.DataFrame({})

# Number of characters
df['num_chars'] = russian_tweets['content'].apply(len)

# Number of words
df['num_words'] = russian_tweets['content'].apply(lambda x: len(x.split()))

# Average word length
df['avg_word_length'] = russian_tweets['content'].apply(lambda x: sum([len(w) for w in x.split()])/len(x.split()))

# Hashtags and mentions
df['hashtag_count'] = russian_tweets['content'].apply(lambda x: len([w for w in x.split() if (w.startswith('#') and len(w)>1)]))

# Hashtags and mentions
df['mentions_count'] = russian_tweets['content'].apply(lambda x: len([w for w in x.split() if (w.startswith('@') and len(w)>1)]))

print(df.head())

print('*********************************************************')
print('** 1.5 Character count of Russian tweets')
print('*********************************************************')
# Prepare data
tweets = russian_tweets.copy(deep=True)

# Create a feature char_count
tweets['char_count'] = tweets['content'].apply(len)

# Print the average character count
print(tweets['char_count'].mean())

print('*********************************************************')
print('** 1.6 Word count of TED talks')
print('*********************************************************')
# Prepare data
ted = ted_talk.copy(deep=True)

# Number of words
ted['num_words'] = ted['transcript'].apply(lambda x: len(x.split()))

# Print the average character count
print(ted['num_words'].mean())

print('*********************************************************')
topic = '1.7 Hashtags and mentions in Russian tweets'; print(f'** {topic}')
print('*********************************************************')
# Prepare data
tweets = russian_tweets.copy(deep=True)

# Create a feature hashtag_count 
tweets['hashtag_count'] = tweets['content'].apply(lambda x: len([w for w in x.split() if (w.startswith('#') and len(w)>1)]))

# Create a feature mention_count
tweets['mention_count'] = tweets['content'].apply(lambda x: len([w for w in x.split() if (w.startswith('@') and len(w)>1)]))

# Display distribution
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
for ax, c, t in zip(axes.flatten(), ['hashtag_count', 'mention_count'], ['Hashtag', 'Mention']):
    tweets[c].hist(ax=ax)
    ax.set_xlabel(c)
    ax.set_ylabel('Frequency')
    ax.set_title(f'{t} count distribution', **title_param)
        
fig.suptitle(topic, **suptitle_param)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.4, hspace=None); #To set the margins 
plt.show()


print('*********************************************************')
print('** 1.8 Readability tests')
print('*********************************************************')
my_texts = ["This is a short sentence.", 
            "This is longer sentence with more words and it is harder to follow than the first sentence.",
            "I live in my home.",
            "I reside in my domicile.",
            "A neuron has three main parts: dendrites, an axon, and a cell body or soma."]

def flesh_interpretation(punctuation):
    """Return Flesch reading ease score interpretation."""
    if punctuation >= 90  : return '5th grade level'
    elif punctuation >= 80: return '6th grade level'
    elif punctuation >= 70: return '7th grade level'
    elif punctuation >= 60: return '8th or 9th grade level'
    elif punctuation >= 50: return '10th to 12th grade level'
    elif punctuation >= 30: return 'College level'
    else: return 'College graduate level'

def gunning_fog_interpretation(punctuation):
    """Return Gunning fog reading ease score interpretation."""
    if punctuation >= 17  : return 'College graduate level'
    elif punctuation >= 16: return 'College senior level'
    elif punctuation >= 15: return 'College junior level'
    elif punctuation >= 14: return 'College sophomore level'
    elif punctuation >= 13: return 'College freshman level'
    elif punctuation >= 12: return 'High school senior level'
    elif punctuation >= 11: return 'High school junior level'
    elif punctuation >= 10: return 'High school sophomore level'
    elif punctuation >=  9: return 'High school freshman level'
    elif punctuation >=  8: return 'Eighth grade level'
    elif punctuation >=  7: return 'Seventh grade level'
    elif punctuation >=  6: return 'Sixth grade level'
    else: return 'Fifth grade level'

for text in my_texts:
    # Create a Textatistic Object
    readability_scores = Textatistic(text).scores
    
    # Generate scores
    fs = readability_scores['flesch_score']
    gs = readability_scores['gunningfog_score']
    print('{} \nFlesch score: {} ({})'.format(text, fs, flesh_interpretation(fs)))
    print('Gunning fog score: {} ({})\n'.format(gs, gunning_fog_interpretation(gs)))
    
print('*********************************************************')
print("** 1.9 Readability of 'The Myth of Sisyphus'")
print('*********************************************************')
# Read the essay to analize
with open('data/sisyphus_essay.dat','r') as f: 
    sisyphus_essay = f.read() 
    
# Compute the readability scores 
readability_scores = Textatistic(sisyphus_essay).scores

# Print the flesch reading ease score
flesch = readability_scores['flesch_score']
print("The Flesch Reading Ease is %.2f (%s)." % (flesch, flesh_interpretation(flesch)))

print('*********************************************************')
print('** 1.10 Readability of various publications')
print('*********************************************************')
# List of excerpts
articles = ['forbes', 'harvard_law', 'r_digest', 'time_kids']

# Loop through excerpts and compute gunning fog index
for article in articles:
    # Read the essay to analize
    with open(f'data/{article}.dat','r', encoding='utf-8') as f: 
        excerpt = f.read() 
    readability_scores = Textatistic(excerpt).scores
    gunning_fog = readability_scores['gunningfog_score']
    
    # Print the gunning fog indices
    print('The Gunning Fog "%s" essay is %.2f (%s).' % (article, gunning_fog, gunning_fog_interpretation(gunning_fog)))
    
print('*********************************************************')
print('END')
print('*********************************************************')