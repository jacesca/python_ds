# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 1: Regular expressions & word tokenization
    This chapter will introduce some basic NLP concepts, such as word 
    tokenization and regular expressions to help parse text. You'll also 
    learn how to handle non-English text and more difficult tokenization you 
    might find.
Source: https://learn.datacamp.com/courses/introduction-to-natural-language-processing-in-python
"""
###############################################################################
## Importing libraries
###############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns

import re

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize #tokenize a document into sentences
from nltk.tokenize import regexp_tokenize #tokenize a string or document based on a regular expression pattern
from nltk.tokenize import TweetTokenizer #special class just for tweet tokenization, allowing you to separate hashtags, mentions and lots of exclamation points!!!
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import blankline_tokenize

from wordcloud import WordCloud #Documentation: https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html#wordcloud.WordCloud


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
with open('grail.txt','r') as f: 
    scene_one = f.read()

df_tweets = pd.read_csv('tweets.csv')


###############################################################################
## Main part of the code
###############################################################################
def Introduction_to_regular_expressions(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "1. Introduction to regular expressions"; print("** %s" % topic)
    print("****************************************************")
    
    print('--------------------------------------------re.match')
    my_pattern = r'abc'
    my_string = 'abcdef'
    result = re.match(my_pattern, my_string)
    print(f'Finding first coincidence with "abc" pattern: \n{result}\n')
    
    my_pattern = r'\w+'
    result = re.match(my_pattern, 'hi there!')
    print(f'Finding first word: \n{result}\n')
    
    print('--------------------------------------------re.split')
    my_pattern = r'\s+'
    result = re.split(my_pattern, 'Split on spaces.')
    print(f'Splitting on space: \n{result}\n')
    
    
def Which_pattern(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "2. Which pattern?"; print("** %s" % topic)
    print("****************************************************")
    
    print('------------------------------------------re.findall')
    my_pattern = r"\w+"
    result = re.findall(my_pattern, "Let's write RegEx!")
    print(f'Finding all words in the sentences: \n{result}\n')
    
    
    
def Practicing_regular_expressions_resplit_and_refindall(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "3. Practicing regular expressions: re.split() and re.findall()"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize variables
    my_string = "Let's write RegEx!  Won't that be fun?  I sure think so.  Can you find 4 sentences?  Or perhaps, all 19 words?"
    print(f'{my_string}\n')
    
    print('--------------------------------------------re.split')
    # Write a pattern to match sentence endings: sentence_endings
    sentence_endings = r"[.?!]"
    
    # Split my_string on sentence endings and print the result
    result = re.split(sentence_endings, my_string)
    print(f'Finding sentences (splitting on "." or "?" or "!"): \n{result}\n')

    # Split my_string on spaces and print the result
    spaces = r"\s+"
    result = re.split(spaces, my_string)
    print(f'Splitting on spaces: \n{result}\n')
    
    print('------------------------------------------re.findall')
    # Find all capitalized words in my_string and print the result
    capitalized_words = r"[A-Z]\w+"
    result = re.findall(capitalized_words, my_string)
    print(f'Finding all words with capital letters: \n{result}\n')
    
    # Find all digits in my_string and print the result
    digits = r"\d+"
    result = re.findall(digits, my_string)
    print(f'Finding all digits in the strings: \n{result}\n')
    
    
    
def Introduction_to_tokenization(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "4. Introduction to tokenization"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('---------------------------------------word_tokenize')
    result = word_tokenize("Hi there!")
    print(f'Word tokenize: {result}')
    result = word_tokenize("I don't like Sam's shoes.")
    print(f'Word tokenize: {result}')
    s = "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\n\nThanks."
    result = word_tokenize(s)
    print(f'Word tokenize: {result}\n')
    
    print('---------------------------------------sent_tokenize')
    my_string = "¿Qué es poesía?, dices mientras clavas en mi pupila tu pupila azul. ¡Qué es poesía! ¿Y tú me lo preguntas? Poesía... eres tú."
    result = sent_tokenize(my_string)
    print(f"Sent tokenize: {result}")
    result = word_tokenize(s)
    print(f"Sent tokenize: {result}\n")
    
    print('-------------------------------------regexp_tokenize')
    p = r'\w+|\$[\d\.]+|\S+'
    result = regexp_tokenize(s, pattern=p)
    print(f'RegExp tokenize: {result}\n')
    
    print('----------------------------------wordpunct_tokenize')
    result = wordpunct_tokenize(s)
    print(f'Wordpunct tokenize: {result}\n')
    
    print('----------------------------------blankline_tokenize')
    result = blankline_tokenize(s)
    print(f'Blankline tokenize: {result}\n')
    
    print('--------------------------------------TweetTokenizer')
    my_tweet = "@Julianne take what is yours and vote for #GameofThrones in this year's #Webbys: https://itsh.bo/3bbRab3 !!!!!"
    
    tknzr = TweetTokenizer()
    result = tknzr.tokenize(my_tweet)
    print(f'Tweet tokenize: {result}\n')
    
    tknzr = TweetTokenizer(strip_handles=True)
    result = tknzr.tokenize(my_tweet)
    print(f'Tweet tokenize: {result}\n')
    
    print('--------------------------------------------re.match')
    my_string = 'abcdefabcdef'
    my_pattern = r'abc'
    result = re.match(my_pattern, my_string)
    print(f'Finding coincidence since begininng with "abc" pattern: \n{result}\n')
    
    my_pattern = r'cd'
    result = re.match(my_pattern, my_string)
    print(f'Finding coincidence since begininng with "cd" pattern: \n{result}\n')
    
    print('-------------------------------------------re.search')
    my_string = 'abcdefabcdef'
    my_pattern = r'abc'
    result = re.search(my_pattern, my_string)
    print(f'Finding first coincidence with "abc" pattern: \n{result}\n')
    
    my_pattern = r'cd'
    result = re.search(my_pattern, my_string)
    print(f'Finding first coincidence with "cd" pattern: \n{result}\n')
    
    
    
def Word_tokenization_with_NLTK(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "5. Word tokenization with NLTK"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('---------------------------------------sent_tokenize')
    # Split scene_one into sentences: sentences
    sentences = sent_tokenize(scene_one)
    print(f"First five sentences: \n{sentences[:5]}")
    
    print('---------------------------------------word_tokenize')
    # Use word_tokenize to tokenize the fourth sentence: tokenized_sent
    tokenized_sent = word_tokenize(sentences[3])
    print(f"Words in the 4th sentence: \n{tokenized_sent}")
    
    print('--------------------------------unique word_tokenize')
    # Make a set of unique tokens in the entire scene: unique_tokens
    unique_tokens = set(word_tokenize(scene_one))
    print(f"10 first unique words: \n{list(unique_tokens)[:10]}")
    
    
    print("****************************************************")
    topic = "6. More regex with re.search()"; print("** %s" % topic)
    print("****************************************************")
    print('------------------------------------simple re.search')
    my_pattern = r'coconuts'
    
    # Search for the first occurrence of "coconuts" in scene_one: match
    match = re.search(my_pattern, scene_one)
    
    # Print the start and end indexes of match
    print(f"Finding the first coincidence of {my_pattern} in scene_one")
    print(match.start(), match.end())
    print(f'{match}\n')
    
    print('------------------------complex pattern - non greedy')
    # Write a regular expression to search for anything in square brackets: pattern1
    my_pattern = r"\[.*?]"
    # Use re.search to find the first text in square brackets
    match = re.search(my_pattern, scene_one)
    print(f'With pattern "{my_pattern}": \n{match}\n')
    
    print('----------------------------complex pattern - greedy')
    # Write a regular expression to search for anything in square brackets: pattern1
    my_pattern = r"\[.*]"
    # Use re.search to find the first text in square brackets
    match = re.search(my_pattern, scene_one)
    print(f'With pattern "{my_pattern}": \n{match}\n')
    
    print('---------------------using complex pattern re.search')
    # Find the script notation at the beginning of the fourth sentence and print it
    my_pattern = r"[\w\s]+:"
    print(f"4th Sentence: \n{sentences[3]}")
    match = re.search(my_pattern, sentences[3])
    print(f'With pattern "{my_pattern}": \n{match}\n')
    
    
    
def Advanced_tokenization_with_NLTK_and_regex(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "7. Advanced tokenization with NLTK and regex"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------------------Match digits and words')
    my_string = 'He has 11 cats.'
    match_digits_and_words = r'(\d+|\w+)'
    result = re.findall(match_digits_and_words, my_string)
    print(f'Token founds in "{my_string}": \n{result}\n')
    
    print('----Match all lowercase ascii, any digits and spaces')
    my_string = 'match lowercase spaces nums like 12, but no commas'
    my_pattern = r'[a-z0-9 ]+'
    result = re.match(my_pattern, my_string)
    print(f'Token founds in "{my_string}": \n{result}\n')
    print(f'Getting the result only: \n+"{result.group()}"\n')
    
    
def Choosing_a_tokenizer(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "8. Choosing a tokenizer"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('-------------------------------------regexp_tokenize')
    my_string = "SOLDIER #1: Found them? In Mercea? The coconut's tropical!"
    print(f'my_string = "{my_string}"\n')
    
    my_pattern = r'(\w+|\?|!)'
    result = regexp_tokenize(my_string, pattern=my_pattern)
    print(f'With pattern "{my_pattern}": \n{result}\n')
    
    my_pattern = r'(\w+|#\d|\?|!)'
    result = regexp_tokenize(my_string, pattern=my_pattern)
    print(f'With pattern "{my_pattern}": \n{result}\n')
    
    my_pattern = r'(#\d\w+\?!)'
    result = regexp_tokenize(my_string, pattern=my_pattern)
    print(f'With pattern "{my_pattern}": \n{result}\n')
    
    my_pattern = r'\s+'
    result = regexp_tokenize(my_string, pattern=my_pattern)
    print(f'With pattern "{my_pattern}": \n{result}\n')
    
    
    
def Regex_with_NLTK_tokenization(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "9. Regex with NLTK tokenization"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    tweets = ['This is the best #nlp exercise ive found online! #python',
              '#NLP is super fun! <3 #learning',
              'Thanks @datacamp :) #nlp #python']
    print(tweets, '\n')
    
    print('------------------------------------Finding Hashtags')
    # Define a regex pattern to find hashtags: pattern1
    pattern1 = r"#\w+"
    
    # Use the pattern on the first tweet in the tweets list
    hashtags = [regexp_tokenize(t, pattern1) for t in tweets]
    print(np.array(hashtags).flatten(), '\n')
    
    print('------------------------Finding Hastags and mentions')
    # Write a pattern that matches both mentions (@) and hashtags
    pattern2 = r"[@|#]\w+"
    
    # Use the pattern on the last tweet in the tweets list
    mentions_hashtags = [regexp_tokenize(t, pattern2) for t in tweets]
    print(list(np.concatenate(mentions_hashtags)), '\n')
    
    print('------------------------------------------All Tokens')
    # Use the TweetTokenizer to tokenize all tweets into one list
    tknzr = TweetTokenizer()
    all_tokens = [tknzr.tokenize(t) for t in tweets]
    print(list(np.concatenate(all_tokens)), '\n')
    
    print('-----------------------------Wordcloud Hashtags only')
    df_tweets['hashtags'] = df_tweets.text.apply(lambda x: ' '.join(regexp_tokenize(x, pattern1))) 
    only_hashtags = df_tweets.hashtags.str.cat(sep=' ')
    
    # Generate and show the word cloud
    wordcloud = WordCloud(width=1360, height=960,         #Width and height of the canvas.
                          collocations=False,             #Not include biagrams
                          #max_words=100,                 #The maximum number of words.
                          #collocation_threshold=1000,    #score greater than this parameter to be counted as bigrams
                          stopwords={},                   #The words that will be eliminated
                          #min_word_length=6,             #Minimum number of letters a word must have to be included.
                          background_color="floralwhite", 
                          colormap='gist_rainbow').generate(only_hashtags)
    
    # Create a figure of the generated cloud
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Only Hashtags', **title_param)
    
    fig.suptitle(topic, **suptitle_param)
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=.5) #To set the margins 
    plt.show()
    
    
def Non_ascii_tokenization(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "10. Non-ascii tokenization"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    german_text = ('Wann gehen wir Pizza essen? 🍕 Und fährst du mit Über? 🚕')
    print(german_text)
    
    print('---------------------------------------Word Tokenize')
    # Tokenize and print all words in german_text
    all_words = word_tokenize(german_text)
    print(all_words)
    
    print('-----------------------regexp_tokenize Capital Words')
    # Tokenize and print only capital words
    capital_words = r"[A-ZÄËÏÖÜ]\w+"
    print(regexp_tokenize(german_text, capital_words))
    
    print('-------------------------------regexp_tokenize Emoji')
    # Tokenize and print only emoji
    emoji = "['\U0001F300-\U0001F5FF'|'\U0001F600-\U0001F64F'|'\U0001F680-\U0001F6FF'|'\u2600-\u26FF\u2700-\u27BF']"
    print(regexp_tokenize(german_text, emoji))
    
    
    
def Charting_word_length_with_NLTK(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "11. Charting word length with NLTK"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    my_string = 'This is a pretty cool tool!'
    print(my_string)
    
    print('---------------------------------------Word Tokenize')
    # Tokenize and print all words in german_text
    words = word_tokenize(my_string)
    print(words)
    
    print('----------------------Transforming to a numeric list')
    word_lengths = [len(w) for w in words]
    
    print('--------------------------------------------Plotting')
    fig, ax = plt.subplots()
    ax.hist(word_lengths)
    ax.set_xlabel('Lengt of words'); ax.set_ylabel('Frequency');
    ax.set_title(f'Word tokenize in phrase:\n"{my_string}"', **title_param)
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.8, wspace=None, hspace=None) #To set the margins 
    plt.show()
    
    print('---------------Hist of token in the Tweets dataframe')
    my_pattern = r"[@|#]?\w+"
    
    df_tweets['tokens'] = df_tweets.text.apply(lambda x: ' '.join(regexp_tokenize(x, my_pattern))) 
    tokens = df_tweets.tokens.str.cat(sep=' ').split(' ')
    word_lengths = [len(w) for w in tokens]
    
    fig, ax = plt.subplots()
    ax.hist(word_lengths, density=True)
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0%}'))
    ax.set_xlabel('Lengt of words'); ax.set_ylabel('Frequency');
    ax.set_title(f'Word tokenize in the Tweets_df', **title_param)
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=.15, bottom=None, right=None, top=.8, wspace=None, hspace=None) #To set the margins 
    plt.show()
    
    
    
def Charting_practice(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "12. Charting practice"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('----------------------------------------Tokenization')
    # Split the script into lines: lines
    lines = scene_one.splitlines()
    
    # Replace all script lines for speaker (Delete the prompt)
    pattern = "[A-Z]{2,}(\s)?(#\d)?([A-Z]{2,})?:"
    lines = [re.sub(pattern, '', l) for l in lines]
    
    # Tokenize each line: tokenized_lines
    tokenized_lines = [regexp_tokenize(s, '\w+') for s in lines]
    
    # Make a frequency list of lengths: line_num_words
    line_num_words = [len(t_line) for t_line in tokenized_lines]
    
    print('--------------------------------------------Plotting')
    fig, ax = plt.subplots()
    ax.hist(line_num_words)
    #ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0%}'))
    ax.set_xlabel('Lengt of words'); ax.set_ylabel('Frequency');
    ax.set_title(f'Word tokenize in the "Holy Grail script"', **title_param)
    fig.suptitle(topic, **suptitle_param)
    plt.subplots_adjust(left=.15, bottom=None, right=None, top=.8, wspace=None, hspace=None) #To set the margins 
    plt.show()
    
    
        
def main():
    print("****************************************************")
    print("** BEGIN                                          **")
    
    Introduction_to_regular_expressions()
    Which_pattern()
    Practicing_regular_expressions_resplit_and_refindall()
    Introduction_to_tokenization()
    Word_tokenization_with_NLTK()
    Advanced_tokenization_with_NLTK_and_regex()
    Choosing_a_tokenizer()
    Regex_with_NLTK_tokenization()
    Non_ascii_tokenization()
    Charting_word_length_with_NLTK()
    Charting_practice()

    
    print("****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    pd.reset_option("display.max_columns")
    plt.style.use('default')
    np.set_printoptions(formatter={'float': None})