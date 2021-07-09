# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 23:19:59 2020

@author: jacesca

Part-Of-Speech tagging (or POS tagging, for short) is one of the main components of 
almost any NLP analysis. The task of POS-tagging simply implies labelling words with 
their appropriate Part-Of-Speech (Noun, Verb, Adjective, Adverb, Pronoun, …).

Penn Treebank Tags
The most popular tag set is Penn Treebank tagset. Most of the already trained taggers 
for English are trained on this tag set. Examples of such taggers are:
(*) NLTK default tagger
(*) Stanford CoreNLP tagger

Documentation:
    Source of code: https://nlpforhackers.io/training-pos-tagger/
    List of tags: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

Alphabetical list of part-of-speech tags used in the Penn Treebank Project:
Number Tag Description
1.	   CC	Coordinating conjunction
2.	   CD	Cardinal number
3.	   DT	Determiner
4.	   EX	Existential there
5.	   FW	Foreign word
6.	   IN	Preposition or subordinating conjunction
7.	   JJ	Adjective
8.	   JJR	Adjective, comparative
9.	   JJS	Adjective, superlative
10.	   LS	List item marker
11.	   MD	Modal
12.	   NN	Noun, singular or mass
13.	   NNS	Noun, plural
14.	   NNP	Proper noun, singular
15.	   NNPS	Proper noun, plural
16.	   PDT	Predeterminer
17.	   POS	Possessive ending
18.	   PRP	Personal pronoun
19.	   PRP$	Possessive pronoun
20.	   RB	Adverb
21.	   RBR	Adverb, comparative
22.	   RBS	Adverb, superlative
23.	   RP	Particle
24.	   SYM	Symbol
25.	   TO	to
26.	   UH	Interjection
27.	   VB	Verb, base form
28.	   VBD	Verb, past tense
29.	   VBG	Verb, gerund or present participle
30.	   VBN	Verb, past participle
31.	   VBP	Verb, non-3rd person singular present
32.	   VBZ	Verb, 3rd person singular present
33.	   WDT	Wh-determiner
34.	   WP	Wh-pronoun
35.	   WP$	Possessive wh-pronoun
36.	   WRB	Wh-adverb
"""

print("****************************************************")
step = "BEGIN"; print("** %s" % step)
print("****************************************************")
step = "Importing libraries.."; print("** %s" % step)

import pprint 
import numpy as np

import nltk
from nltk import word_tokenize
from nltk import pos_tag
 
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
 
print("****************************************************")
step = "Preparing the environment...\n"; print("** %s" % step)
SEED           = 42
np.random.seed(SEED)

print("****************************************************")
step = "What is POS tagging\n"; print("** %s" % step)

sentence = "I'm learning NLP"
tags = pos_tag(word_tokenize(sentence))

print(f'My sentence: "{sentence}" \nPOS tagging: {tags}\m')


print("****************************************************")
step = "POS tagging tools in NLTK\n"; print("** %s" % step)
###########################################################################################################
# There are some simple tools available in NLTK for building your own POS-tagger.                         #
# You can read the documentation here: NLTK Documentation Chapter 5 , section 4: “Automatic Tagging”.     #
# http://www.nltk.org/book/ch05.html                                                                      #
# You can build simple taggers such as:                                                                   #
# - DefaultTagger that simply tags everything with the same tag                                           #
# - RegexpTagger that applies tags according to a set of regular expressions                              # 
# - UnigramTagger that picks the most frequent tag for a known word                                       #  
# - BigramTagger, TrigramTagger working similarly to the UnigramTagger but also taking some of the        #  
#   context into consideration                                                                            #
###########################################################################################################
print("****************************************************")
step = "Picking a corpus to train the POS tagger\n"; print("** %s" % step)
###########################################################################################################
# Resources for building POS taggers are pretty scarce, simply because annotating a                       #
# huge amount of text is a very tedious task. One resource that is in our reach and                       #
# that uses our prefered tag set can be found inside NLTK.                                                # 
###########################################################################################################
tagged_sentences = nltk.corpus.treebank.tagged_sents()
 
print(tagged_sentences[0])
print("Tagged sentences: ", len(tagged_sentences))
print("Tagged words:", len(nltk.corpus.treebank.tagged_words()))


print("****************************************************")
step = "Showing how 'DictVectorizer' works"; print("** %s" % step)

v = DictVectorizer(sparse=False)
D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
X = v.fit_transform(D)

print(f"""
My list of dict: {D}
Features extracted: {v.get_feature_names()}
Transformed into: \n {X}
""")

D_again = v.inverse_transform(X)
print(f"Returning to the normality: {D_again}")

new_data = {'foo': 4, 'unseen_feature': 3}
new_X = v.transform(new_data)
print(f"""
Working with new data: {new_data}
Transformed in: \n{new_X}
""")

print("****************************************************")
step = "Training our own POS Tagger using scikit-learn\n"; print("** %s" % step)
###########################################################################################################
# Before starting training a classifier, we must agree first on what features to use.                     #
# Most obvious choices are: the word itself, the word before and the word after.                          #
# That’s a good start, but we can do so much better. For example, the 2-letter suffix                     # 
# is a great indicator of past-tense verbs, ending in “-ed”. 3-letter suffix helps                        #
# recognize the present participle ending in “-ing”.                                                      #
###########################################################################################################
def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }

sentence = ['This', 'is', 'a', 'sentence']
word_index = 2
feature_sent = features(sentence, word_index)

print(f'My sentence: "{sentence}" \nFeatures in {word_index+1}° word "{sentence[1]}" are:')
pprint.pprint(feature_sent)

v = DictVectorizer(sparse=False)
X = v.fit_transform([feature_sent])

print(f"""
Features extracted: {v.get_feature_names()}
Prediction: \n {X}
""")

###########################################################################################################
# Small helper function to strip the tags from our tagged corpus and feed it to our classifier:           #
###########################################################################################################
def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]


###########################################################################################################
# Let’s now build our training set. Our classifier should accept features for a single word,              #
# but our corpus is composed of sentences. We’ll need to do some transformations:                         #
###########################################################################################################
# Split the dataset for training and testing
cutoff = int(.75 * len(tagged_sentences))
training_sentences = tagged_sentences[:cutoff]
test_sentences = tagged_sentences[cutoff:]
 
print(f'\n\nFor training: {len(training_sentences)}')   # 2935
print(f'For testing: {len(test_sentences)}\n')         # 979

print(f'First sentence in corpus: \n"{tagged_sentences[0]}"\n')
 
def transform_to_dataset(tagged_sentences):
    X, y = [], []
 
    for tagged in tagged_sentences:
        untagged = untag(tagged)
        for index in range(len(untagged)):
            X.append(features(untagged, index))
            y.append(tagged[index][1])
 
    return X, y
 
X, y = transform_to_dataset(training_sentences)
print(f'First register of the data (X, y): \n{X[0]} \n{y[0]}\n\n')

###########################################################################################################
# We’re now ready to train the classifier. I’ve opted for a DecisionTreeClassifier. Feel free to play     #
# with others:
###########################################################################################################
clf = Pipeline([('vectorizer', DictVectorizer(sparse=False)),
                ('classifier', DecisionTreeClassifier(criterion='entropy'))])
 
# Use only the first 10K samples if you're running it multiple times. It takes a fair bit :)
clf.fit(X[:30000], y[:30000])   # 75,784 total records
print('Training completed...')
 
X_test, y_test = transform_to_dataset(test_sentences)
print("Accuracy:", clf.score(X_test, y_test))



print("****************************************************")
step = "Training our own POS Tagger using scikit-learn\n"; print("** %s" % step)
###########################################################################################################
# We can now use our classifier like this:                                                                #
###########################################################################################################
def pos_tag(sentence):
    tags = clf.predict([features(sentence, index) for index in range(len(sentence))])
    return list(zip(sentence, tags))

sentence = 'This is my friend, John.'
transformed_sent = pos_tag(word_tokenize(sentence))
print(f'My sentence: \"{sentence}" \nTransformed into: \n{transformed_sent}')



print("****************************************************")
step = "Conclusion\n"; print("** %s" % step)
###########################################################################################################
# - Training your own POS tagger is not that hard
# - All the resources you need are right there
# - Hopefully this article sheds some light on this subject, that can sometimes be considered extremely 
#   tedious and “esoteric”
###########################################################################################################


print("****************************************************")
step = "END"; print("** %s" % step)
print("****************************************************")