# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 13:12:41 2020

@author: jacesca@gmail.com
Testing SpaCy
"""

print("****************************************************")
topic = "BEGIN"; print("** %s" % topic)
print("****************************************************")
topic = "01. Importig libraries"; print("** %s" % topic)
print("****************************************************")
import pandas as pd
print ("pandas imported...")
import matplotlib.pyplot as plt
print ("matplotlib.pyplot imported...")
import webbrowser
print("webbrowser imported...")
import os
print("os imported...")
import numpy as np
print("numpy imported...")

from pprint import pprint
print("pprint.pprint imported")
from cairosvg import svg2png #Transform svg file into png file
print("cairosvg.svg2png imported...")

import spacy
from spacy import displacy
from spacy.tokens import Span
from spacy.tokens import Doc
from spacy.tokens import Token
from spacy.matcher import Matcher # Matcher is initialized with the shared vocab
print ("spacy imported...")


print("****************************************************")
topic = "02. Set the configuration"; print("** %s" % topic)
print("****************************************************")
np.set_printoptions(threshold=8)


print("****************************************************")
topic = "03. Statistical models"; print("** %s" % topic)
print("****************************************************")
# Load the installed model "en_core_web_sm"
nlp = spacy.load("en_core_web_lg")
print("en_core_web_lg loaded...")


print("****************************************************")
topic = "04. Documents and tokens"; print("** %s" % topic)
print("****************************************************")
# Processing text
doc = nlp("This is a text")

# Accessing token attributes
doc = nlp("This is a text")
# Token texts
print([token.text for token in doc])
# ['This', 'is', 'a', 'text']


print("****************************************************")
topic = "05. Spans"; print("** %s" % topic)
print("****************************************************")
print("-------------------------------------Accessing spans")
# Accessing spans
doc = nlp("This is a text")
span = doc[2:4]
print(span.text)
# 'a text'

print("----------------------------Creating a span manually")
# Create a Doc object
doc = nlp("I live in New York")
print(doc)
# Span for "New York" with label GPE (geopolitical)
span = Span(doc, 3, 5, label="GPE")
print(span.text)
print(span.label, span.label_)
# 'New York'


print("****************************************************")
topic = "06. Linguistic features"; print("** %s" % topic)
print("****************************************************")
doc = nlp("This is a text.")
print(doc,'\n')

print('---------------------------------Part-of-speech tags')
print("Attributes return label IDs. For string labels, use the attributes with an underscore.\n")
df = pd.DataFrame({'.text': [token.text for token in doc],
                   '.pos' : [token.pos  for token in doc],
                   '.pos_': [token.pos_ for token in doc], # Coarse-grained part-of-speech tags
                   '.tag' : [token.tag  for token in doc],
                   '.tag_': [token.tag_ for token in doc],
                   'TAG EXPLAIN': [spacy.explain(token.tag_) for token in doc]}) # Fine-grained part-of-speech tags
print(df)

print('------------------------------Syntactic dependencies')
df = pd.DataFrame({'.text'      : [token.text       for token in doc],
                   '.dep'       : [token.dep        for token in doc],
                   '.dep_'      : [token.dep_       for token in doc], # Dependency labels
                   '.head.text' : [token.head.text  for token in doc],
                   'DEP EXPLAIN': [spacy.explain(token.dep_) for token in doc]}) # Syntactic head token (governor)
print(df)

print("--------------------------------------Named entities")
doc = nlp("Larry Page founded Google in New York")
print(doc)
# Text and label of named entity span
print([(ent.text, ent.label_) for ent in doc.ents])
# [('Larry Page', 'PERSON'), ('Google', 'ORG')]


print("****************************************************")
topic = "07. Syntax iterators"; print("** %s" % topic)
print("****************************************************")
print("-------------------------------------------Sentences")
doc = nlp("This a sentence. This is another one.")
print(doc)
# doc.sents is a generator that yields sentence spans
print([sent.text for sent in doc.sents])
# ['This is a sentence.', 'This is another one.']

print("-----------------------------------Base noun phrases")
doc = nlp("I have a red car")
print(doc)
# doc.noun_chunks is a generator that yields spans
print([chunk.text for chunk in doc.noun_chunks])
# ['I', 'a red car']


print("****************************************************")
topic = "08. Label explanations"; print("** %s" % topic)
print("****************************************************")
print("RB --->", spacy.explain("RB"))
# 'adverb'
print("GPE -->", spacy.explain("GPE"))
# 'Countries, cities, states'


print("****************************************************")
topic = "09. Visualizing"; print("** %s" % topic)
print("****************************************************")
# If you're in a Jupyter notebook, use displacy.render .
# Otherwise, use displacy.serve to start a web server and
# show the visualization in your browser.

print("------------------------------Visualize dependencies")
doc = nlp("This is a sentence")

# generate the diagram
svg = displacy.render(doc, style="dep", jupyter=False)

# Save the svg file
filename = "spacy_dep.svg"
with open(filename, 'w') as f: print(svg, file=f)

# open an HTML file on my own (Windows) computer
filename = 'file:///' + os.getcwd().replace('\\','/').replace(' ','%20') + '/' + filename
webbrowser.open(filename, new=2)

# tranform to png
filename = "spacy_dep.png"
svg2png(bytestring=svg, write_to=filename, dpi=200)

# plot the diagram
fig, ax = plt.subplots()
spacy_diagram   = plt.imread(filename)
ax.imshow(spacy_diagram)
ax.set_title("Spacy Dependencies Visualization")
ax.axis('off')
plt.show()



print("----------------------------Visualize named entities")
doc = nlp("Larry Page founded Google")

# generate the representation
html = displacy.render(doc, style="ent", jupyter=False)

# Save the svg file
filename = "spacy_ent.html"
with open(filename, 'w') as f: print(html, file=f)

# open an HTML file on my own (Windows) computer
filename = 'file:///' + os.getcwd().replace('\\','/').replace(' ','%20') + '/' + filename
webbrowser.open(filename, new=2)


print("****************************************************")
topic = "10. Word vectors and similarity"; print("** %s" % topic)
print("****************************************************")
print("--------------------------------Comparing similarity")
doc1 = nlp("I like cats")
doc2 = nlp("I like dogs")
print(f'doc1 = "{doc1}"')
print(f'doc2 = "{doc2}"')

# Compare 2 documents
print(f'"{doc1}" = "{doc2}":', doc1.similarity(doc2))

# Compare 2 tokens
print(f'"{doc1[2]}" = "{doc2[2]}":', doc1[2].similarity(doc2[2]))

# Compare tokens and spans
print(f'"{doc1[0]}" = "{doc2[1:3]}":', doc1[0].similarity(doc2[1:3]))


print("------------------------------Accessing word vectors")
# Vector as a numpy array
doc = nlp("I like cats")
print(doc)

# The L2 norm of the token's vector
print(f'Vector of "{doc[2]}" (dimensions: {len(doc[2].vector)}):\n{doc[2].vector}')
print(f'Logitud (L2 Norm) of "{doc[2]}":', doc[2].vector_norm)


print("****************************************************")
topic = "11. Pipeline components"; print("** %s" % topic)
print("****************************************************")
# Functions that take a Doc object, modify it and return it.

print("--------------------------------Pipeline information")
nlp_sm = spacy.load("en_core_web_sm")

print(nlp_sm.pipe_names)
# ['tagger', 'parser', 'ner']

pprint(nlp_sm.pipeline)
# [('tagger', <spacy.pipeline.Tagger>),
# ('parser', <spacy.pipeline.DependencyParser>),
# ('ner', <spacy.pipeline.EntityRecognizer>)]

print("-----------------------------------Custom components")
# Components can be added first , last (default), or before or after an existing component.
# Function that modifies the doc and returns it
def custom_component(doc):
    print("Do something to the doc here!")
    return doc

# Add the component first in the pipeline
nlp_sm.add_pipe(custom_component, first=True)
doc = nlp_sm("This is a sentence")
print("New doc:", doc)

print(nlp_sm.pipe_names)
# ['tagger', 'parser', 'ner']

pprint(nlp_sm.pipeline)
# [('tagger', <spacy.pipeline.Tagger>),
# ('parser', <spacy.pipeline.DependencyParser>),
# ('ner', <spacy.pipeline.EntityRecognizer>)]



print("****************************************************")
topic = "12. Extension attributes"; print("** %s" % topic)
print("****************************************************")
print("""
      Custom attributes that are registered on the global Doc ,
      Token and Span classes and become available as ._ .
      """)
doc = nlp("The sky over New York is blue")
print(doc)
print("--------------------------------Attribute extensions")
# Register custom attribute on Token class
try: Token.set_extension("is_color", default=False)
except: print("._.is_color attribute in Token defined...")

# Overwrite extension attribute with default value
doc[6]._.is_color = True

df = pd.DataFrame({'.text': [token.text for token in doc],
                   '._.is_color' : [token._.is_color  for token in doc]}) # Custom attribute
print(df)

print("---------------------------------Property extensions")
# Register custom attribute on Doc class
get_reversed = lambda doc: doc.text[::-1]
try: Doc.set_extension("reversed", getter=get_reversed)
except: print("._.reversed property in Doc defined...")

# Compute value of extension attribute with getter
print(f"Reverse: \n{doc._.reversed}")
# 'eulb si kroY weN revo yks ehT'


print("-----------------------------------Method extensions")
# Register custom attribute on Span class
has_label = lambda span, label: span.label_ == label
try: Span.set_extension("has_label", method=has_label)
except: print("._.has_label method in Doc defined...")

# Compute value of extension attribute with method
custom_label = 'GPE'
span = Span(doc, 3, 5, label="GPE")
print(f'"{span}" has "{custom_label}":', span._.has_label("GPE"))
# True


print("****************************************************")
topic = "13. Rule-based matching"; print("** %s" % topic)
print("****************************************************")
print("-----------------------------------Using the matcher")
# Each dict represents one token and its attributes
matcher = Matcher(nlp.vocab)

# Add with ID, optional callback and pattern(s)
pattern = [{"LOWER": "new"}, {"LOWER": "york"}]
matcher.add("CITIES", None, pattern)

# Match by calling the matcher on a Doc object
doc = nlp("I live in New York")
print(doc)
matches = matcher(doc)

# Matches are (match_id, start, end) tuples
for match_id, start, end in matches:
    print(match_id, start, end)
    # Get the matched span by slicing the Doc
    span = doc[start:end]
    print(span.text)
    # 'New York'



print("****************************************************")
topic = "14. Rule-based matching"; print("** %s" % topic)
print("****************************************************")
print("--------------------------------------Token patterns")
# "love cats", "loving cats", "loved cats"
pattern1 = [{"LEMMA": "love"}, {"LOWER": "cats"}]

# "10 people", "twenty people"
pattern2 = [{"LIKE_NUM": True}, {"TEXT": "people"}]

# "book", "a cat", "the sea" (noun + optional article)
pattern3 = [{"POS": "DET", "OP": "?"}, {"POS": "NOUN"}]



print("****************************************************")
topic = "END"; print("** %s" % topic)
print("****************************************************")
