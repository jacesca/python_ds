# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 21:08:50 2020

@author: jacesca
Aditional documentation:
    Source of the code: https://nlpforhackers.io/named-entity-extraction/
    Classes: https://www.w3schools.com/python/python_classes.asp
Download the 2.2.0 version of the corpus here: https://gmb.let.rug.nl/data.php
"""

print("****************************************************")
step = "BEGIN"; print("** %s" % step)
print("****************************************************")
step = "Importing libraries.."; print("** %s" % step)

#import os
import collections
import string
import pickle

from zipfile import ZipFile 
from pprint import pprint
from collections.abc import Iterable

from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk
from nltk.chunk import conlltags2tree
from nltk.chunk import tree2conlltags
from nltk.stem.snowball import SnowballStemmer
from nltk.tag import ClassifierBasedTagger
from nltk.chunk import ChunkParserI

print("****************************************************")
step = "NLTK NER Chunker\n"; print("** %s" % step)
# NLTK has a standard NE annotator so that we can get started pretty quickly.

sentence = "Mark and John are working at Google."
print(f'my sentence: \n"{sentence}"\n')
chunk_sent = ne_chunk(pos_tag(word_tokenize(sentence)))
print(f'{chunk_sent}\n')

# ne_chunk needs part-of-speech annotations to add NE labels to the sentence. 
# The output of the ne_chunk is a nltk.Tree object.
# The ne_chunk function acts as a chunker, meaning it produces 2-level trees:
#    Nodes on Level-1: Outside any chunk
#    Nodes on Level-2: Inside a chunk – The label of the chunk is denoted by the label of the subtree
# In this example, Mark/NNP is a level-2 leaf, part of a PERSON chunk. and/CC 
# is a level-1 leaf, meaning it’s not part of any chunk.


print("****************************************************")
step = "IOB tagging\n"; print("** %s" % step)
# nltk.Tree is great for processing such information in Python, but it’s not the 
# standard way of annotating chunks. Maybe this can be an article on its own but 
# we’ll cover this here really quickly.
# The IOB Tagging system contains tags of the form:
#    B-{CHUNK_TYPE} – for the word in the Beginning chunk
#    I-{CHUNK_TYPE} – for words Inside the chunk
#    O – Outside any chunk
# A sometimes used variation of IOB tagging is to simply merge the B and I tags:
#    {CHUNK_TYPE} – for words inside the chunk
#    O – Outside any chunk
# We usually want to work with the proper IOB format.
# Here’s how to convert between the nltk.Tree and IOB format:

 
sentence = "Mark and John are working at Google."
ne_tree = ne_chunk(pos_tag(word_tokenize(sentence)))
 
iob_tagged = tree2conlltags(ne_tree)
print(f'Return a list of 3-tuples containing (word, tag, IOB-tag):\n {iob_tagged}\n')
 
ne_tree = conlltags2tree(iob_tagged)
print(f'Convert the CoNLL IOB format to a tree:\n {ne_tree}\n')


print("****************************************************")
step = "GMB corpus\n"; print("** %s" % step)
# NLTK doesn’t have a proper English corpus for NER. It has the CoNLL 2002 Named 
# Entity CoNLL but it’s only for Spanish and Dutch. You can definitely try the 
# method presented here on that corpora. In fact doing so would be easier because 
# NLTK provides a good corpus reader. We are going with Groningen Meaning Bank (GMB) 
# though. 
# GMB is a fairly large corpus with a lot of annotations. Unfortunately, GMB is 
# not perfect. It is not a gold standard corpus, meaning that it’s not completely 
# human annotated and it’s not considered 100% correct. The corpus is created by 
# using already existed annotators and then corrected by humans where needed.


###############################################################################
## If you have uncompress data into a folder
###############################################################################
"""
ner_tags = collections.Counter()
corpus_root = "gmb-2.2.0"   # Make sure you set the proper path to the unzipped corpus
for root, dirs, files in os.walk(corpus_root):
    #print(root, dirs, files)
    for filename in files:
        if filename.endswith(".tags"):
            with open(os.path.join(root, filename), 'rb') as file_handle:
                file_content = file_handle.read().decode('utf-8').strip()
                annotated_sentences = file_content.split('\n\n')   # Split sentences
                for annotated_sentence in annotated_sentences:
                    annotated_tokens = [seq for seq in annotated_sentence.split('\n') if seq]  # Split words
                    for idx, annotated_token in enumerate(annotated_tokens):
                        annotations = annotated_token.split('\t')   # Split annotations
                        word, tag, ner = annotations[0], annotations[1], annotations[3]
 
                        ner_tags[ner] += 1  
"""

###############################################################################
## If you have the data in a zipfile
###############################################################################
ner_tags = collections.Counter()
zip_file = 'gmb-2.2.0.zip'
with ZipFile(zip_file) as myzip:
    for filename in myzip.namelist():
        if filename.endswith(".tags"):
            with myzip.open(filename, 'r') as file_handle:
                file_content = file_handle.read().decode('utf-8').strip()
                #print(f'"{filename}" read...')
                annotated_sentences = file_content.split('\n\n')   # Split sentences
                for annotated_sentence in annotated_sentences:
                    annotated_tokens = [seq for seq in annotated_sentence.split('\n') if seq]  # Split words
                    for idx, annotated_token in enumerate(annotated_tokens):
                        annotations = annotated_token.split('\t')   # Split annotations
                        word, tag, ner = annotations[0], annotations[1], annotations[3]
                        ner_tags[ner] += 1  
print(f'Words = {sum(ner_tags.values())}')
print(f'{ner_tags}\n\n') 

## Out:
## Counter({'O': 1146068, 'geo-nam': 58388, 'org-nam': 48034, 'per-nam': 23790, 'gpe-nam': 20680, 
##          'tim-dat': 12786, 'tim-dow': 11404, 'per-tit': 9800, 'per-fam': 8152, 'tim-yoc': 5290, 
##          'tim-moy': 4262, 'per-giv': 2413, 'tim-clo': 891, 'art-nam': 866, 'eve-nam': 602, 
##          'nat-nam': 300, 'tim-nam': 146, 'eve-ord': 107, 'per-ini': 60, 'org-leg': 60, 
##          'per-ord': 38, 'tim-dom': 10, 'per-mid': 1, 'art-add': 1})

print('** FIRST IMPROVEMENT...')
# Let’s interpret the tags a bit. We can observe that the tags are composed (Except for O of course)
# as such: {TAG}-{SUBTAG}. Here’s what the top-level categories mean:
#     geo = Geographical Entity
#     org = Organization
#     per = Person
#     gpe = Geopolitical Entity
#     tim = Time indicator
#     art = Artifact
#     eve = Event
#     nat = Natural Phenomenon
# The subcategories are pretty unnecessary and pretty polluted. per-ini for example tags the Initial of 
# a person’s name. This tag, kind of makes sense. On the other hand, it’s unclear what the difference 
# between per-nam (person name) and per-giv (given name), per-fam (family-name), per-mid (middle-name).
# I decided to just remove the subcategories and focus only on the main ones. Let’s modify the code a bit: 
ner_tags = collections.Counter()
zip_file = 'gmb-2.2.0.zip'
with ZipFile(zip_file) as myzip:
    for filename in myzip.namelist():
        if filename.endswith(".tags"):
            with myzip.open(filename, 'r') as file_handle:
                file_content = file_handle.read().decode('utf-8').strip()
                annotated_sentences = file_content.split('\n\n')   # Split sentences
                for annotated_sentence in annotated_sentences:
                    annotated_tokens = [seq for seq in annotated_sentence.split('\n') if seq]  # Split words
                    for idx, annotated_token in enumerate(annotated_tokens):
                        annotations = annotated_token.split('\t')   # Split annotation
                        word, tag, ner = annotations[0], annotations[1], annotations[3]
                        # Get only the primary category
                        if ner != 'O':
                            ner = ner.split('-')[0]
                        ner_tags[ner] += 1 
print(f'Words = {sum(ner_tags.values())}')
print(f'{ner_tags}\n\n')

# This looks much better. You might decide to drop the last few tags because they are not well
# represented in the corpus. We’ll keep them … for now.


print("****************************************************")
step = "Knowing the data...\n"; print("** %s" % step)
print('Example of annotated_tokens: ')
pprint(annotated_tokens[:5])
print(f'\nLast word, tag, ner: {word}, {tag}, {ner}\n')


print("****************************************************")
step = "Training your own system\n"; print("** %s" % step)

def features(tokens, index, history):
    """
    `tokens`  = a POS-tagged sentence [(w1, t1), ...]
    `index`   = the index of the token we want to extract features for
    `history` = the previous predicted IOB tags
    """
    # init the stemmer
    stemmer = SnowballStemmer('english')
 
    # Pad the sequence with placeholders
    tokens = [('[START2]', '[START2]'), ('[START1]', '[START1]')] + list(tokens) + [('[END1]', '[END1]'), ('[END2]', '[END2]')]
    history = ['[START2]', '[START1]'] + list(history)
 
    # shift the index with 2, to accommodate the padding
    index += 2
 
    word, pos = tokens[index]
    prevword, prevpos = tokens[index - 1]
    prevprevword, prevprevpos = tokens[index - 2]
    nextword, nextpos = tokens[index + 1]
    nextnextword, nextnextpos = tokens[index + 2]
    
    extracted_features = {'word': word,
                          'lemma'           : stemmer.stem(word),
                          'pos'             : pos,
                          'all-ascii'       : all([True for c in word if c in string.ascii_lowercase]),
                          'contains-dash'   : '-' in word,
                          'contains-dot'    : '.' in word,
                          'all-caps'        : word == word.capitalize(),
                          'capitalized'     : word[0] in string.ascii_uppercase,
                          
                          'prev-word'       : prevword,
                          'prev-lemma'      : stemmer.stem(prevword),
                          'prev-pos'        : prevpos,
                          'prev-iob'        : history[index - 1],
                          'prev-all-caps'   : prevword == prevword.capitalize(),
                          'prev-capitalized': prevword[0] in string.ascii_uppercase,
                          
                          'prev-prev-word'  : prevprevword,
                          'prev-prev-pos'   : prevprevpos,
                          
                          'next-word'       : nextword,
                          'next-lemma'      : stemmer.stem(nextword),
                          'next-pos'        : nextpos,
                          'next-all-caps'   : nextword == prevword.capitalize(),
                          'next-capitalized': nextword[0] in string.ascii_uppercase,
                          
                          'next-next-word'  : nextnextword,
                          'next-next-pos'   : nextnextpos,}
    return extracted_features

# The feature extraction works almost identical as the one implemented in the Training a Part-Of-Speech Tagger, 
# (https://nlpforhackers.io/training-pos-tagger/) except we added the history mechanism. Since the previous IOB 
# tag is a very good indicator of what the current IOB tag is going to be, we have included the previous IOB tag 
# as a feature.

# Let’s create a few utility functions to help us with the training and move the corpus reading stuff into a 
# function, read_gmb:

def to_conll_iob(annotated_sentence):
    """
    `annotated_sentence` = list of triplets [(w1, t1, iob1), ...]
    Transform a pseudo-IOB notation: O, PERSON, PERSON, O, O, LOCATION, O
    to proper IOB notation: O, B-PERSON, I-PERSON, O, O, B-LOCATION, O
    """
    proper_iob_tokens = []
    for idx, annotated_token in enumerate(annotated_sentence):
        word, tag, ner = annotated_token
 
        if ner != 'O':
            if idx == 0:
                ner = "B-" + ner
            elif annotated_sentence[idx - 1][2] == ner:
                ner = "I-" + ner
            else:
                ner = "B-" + ner
        proper_iob_tokens.append((word, tag, ner))
    return proper_iob_tokens
 
 
def read_gmb(corpus_root):
    with ZipFile(zip_file) as myzip:
        for filename in myzip.namelist():
            if filename.endswith(".tags"):
                with myzip.open(filename, 'r') as file_handle:
                    file_content = file_handle.read().decode('utf-8').strip()
                    annotated_sentences = file_content.split('\n\n')
                    for annotated_sentence in annotated_sentences:
                        annotated_tokens = [seq for seq in annotated_sentence.split('\n') if seq]
 
                        standard_form_tokens = []
 
                        for idx, annotated_token in enumerate(annotated_tokens):
                            annotations = annotated_token.split('\t')
                            word, tag, ner = annotations[0], annotations[1], annotations[3]
 
                            if ner != 'O':
                                ner = ner.split('-')[0]
 
                            if tag in ('LQU', 'RQU'):   # Make it NLTK compatible
                                tag = "``"
 
                            standard_form_tokens.append((word, tag, ner))
 
                        conll_tokens = to_conll_iob(standard_form_tokens)
 
                        # Make it NLTK Classifier compatible - [(w1, t1, iob1), ...] to [((w1, t1), iob1), ...]
                        # Because the classfier expects a tuple as input, first item input, second the class
                        yield [((w, t), iob) for w, t, iob in conll_tokens]
 

zip_file = 'gmb-2.2.0.zip'
reader = read_gmb(zip_file)

# Check the output:
print('Cheking the output: ')
pprint(reader.__next__())

# We managed to read sentences from the corpus in a proper format. We can now start to 
# actually train a system. NLTK offers a few helpful classes to accomplish the task. 
# nltk.chunk.ChunkParserI is a base class for building chunkers/parsers. Another useful 
# asset we are going to use is the nltk.tag.ClassifierBasedTagger. Under the hood, it 
# uses a NaiveBayes classifier for predicting sequences.

class NamedEntityChunker(ChunkParserI):
    def __init__(self, train_sents, **kwargs):
        assert isinstance(train_sents, Iterable)
 
        self.feature_detector = features
        self.tagger = ClassifierBasedTagger(
            train=train_sents,
            feature_detector=features,
            **kwargs)
 
    def parse(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)
 
        # Transform the result from [((w1, t1), iob1), ...] 
        # to the preferred list of triplets format [(w1, t1, iob1), ...]
        iob_triplets = [(w, t, c) for ((w, t), c) in chunks]
 
        # Transform the list of triplets to nltk.Tree format
        return conlltags2tree(iob_triplets)

# Let’s build the datasets:
zip_file = 'gmb-2.2.0.zip'
reader = read_gmb(zip_file)
data = list(reader)
training_samples = data[:int(len(data) * 0.9)]
test_samples = data[int(len(data) * 0.9):]
 
print("\nTraining samples = %s" % len(training_samples))    # training samples = 55809
print("Test samples = %s\n" % len(test_samples))            # test samples = 6201

# We built everything up to this point so beautifully such that the training can be
# expressed as simply as:
chunker = NamedEntityChunker(training_samples[:2000])

# It probably took a while. Let’s take it for a spin:
sentence = "I'm going to Germany this Monday."
nktl_tree_chunk = chunker.parse(pos_tag(word_tokenize(sentence)))
print(f'Testing: \nMy sentence: "{sentence}"')
print(nktl_tree_chunk)
nktl_tree_chunk.pretty_print()

# The system you just trained did a great job at recognizing named entities:
# - Named Entity “Germany” – Geographical Entity
# - Named Entity “Monday” – Time Entity

print("****************************************************")
step = "Testing the system\n"; print("** %s" % step)

# Let’s see how the system measures up. Because we followed to good patterns in NLTK, we can test 
# our NE-Chunker as simple as this:
score = chunker.evaluate([conlltags2tree([(w, t, iob) for (w, t), iob in iobs]) for iobs in test_samples[:500]])
print(score.accuracy())


print("****************************************************")
step = "Conclusion\n"; print("** %s" % step)

# (*) Chunking can be reduced to a tagging problem.
# (*) Named Entity Recognition is a form of chunking.
# (*) We explored a freely available corpus that can be used for real-world applications.
# (*) The NLTK classifier can be replaced with any classifier you can think about. Try replacing it 
#     with a scikit-learn classifier.
# If you loved this tutorial, you should definitely check out the sequel: Training a NER system on a 
# large dataset. It builds upon what you already learned, it uses a scikit-learn classifier and pushes 
# the accuracy to 97%.
# 
# Notes
# - I’ve used NLTK version 3.2.1
# - You can find the entire code here: Python NER Gist

print("\n****************************************************")
step = "END"; print("** %s" % step)
print("****************************************************")