# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:20:22 2020

@author: jacesca@gmail.com
Chapter 3: Named-entity recognition
    This chapter will introduce a slightly more advanced topic: 
    named-entity recognition. 
    You'll learn how to identify the who, what, and where of your texts using 
    pre-trained models on English and non-English text. You'll also learn how 
    to use some new libraries, polyglot and spaCy, to add to your NLP toolbox.
Source: https://learn.datacamp.com/courses/introduction-to-natural-language-processing-in-python
Aditional documentation:
    https://explosion.ai/demos/displacy-ent/?text=In%20New%20York%2C%20I%20like%20to%20ride%20the%20Metro%20to%20visit%20MOMA%20and%20some%20restaurants%20rated%20well%20by%20Ruth%20Reichl.&model=en_core_web_sm&ents=person%2Cper%2Cnorp%2Cfacility%2Corg%2Cgpe%2Cloc%2Cproduct%2Cevent%2Cwork_of_art%2Clanguage%2Cdate%2Ctime%2Cpercent%2Cmoney%2Cquantity%2Cordinal%2Ccardinal%2Cmisc%2Cdrv%2Cevt%2Cgpe_loc%2Cgpe_org%2Cprod
"""
###############################################################################
## Importing libraries
###############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import chain #To flat a list
from collections import defaultdict #To initialize a dictionary that will assign a default value to non-existent keys.

import requests
from bs4 import BeautifulSoup

import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

import spacy

from polyglot.text import Text #To Named entity recognition. Documentation: https://polyglot.readthedocs.io/en/latest/Installation.html , https://polyglot.readthedocs.io/en/latest/Download.html
from polyglot.downloader import downloader #To download and review existing modules
from polyglot.transliteration import Transliterator #To make traductions

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



###############################################################################
## Main part of the code
###############################################################################
def Named_Entity_Recognition(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "1. Named Entity Recognition"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('-------------Using nltk for Named Entity Recognition')
    sentence = '''
    In New York, I like to ride the Metro to visit MOMA
    and some restaurants rated well by Ruth Reichl.
    '''
    print(f'{sentence}\n')
    
    tokenized_sent = nltk.word_tokenize(sentence)
    print(f'First 3 token found: {tokenized_sent[:3]}\n')
    
    tagged_sent = nltk.pos_tag(tokenized_sent)
    print(f'First 3 tagged word: {tagged_sent[:3]}\n')
    
    print('-------------------------------------NLTK tree found')
    chunk_sent = nltk.ne_chunk(tagged_sent)
    print(f'{chunk_sent}\n')
    
    
def NER_with_NLTK(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "2. NER with NLTK"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------------------------Reading the data')
    with open('News articles/uber_apple.txt','r', encoding='utf-8') as f: 
        article = f.read()
    
    print('-------------Using nltk for Named Entity Recognition')
    # Tag each tokenized sentence into parts of speech for each tokenize sent: pos_sentences
    pos_sentences = [nltk.pos_tag(word_tokenize(sent)) for sent in sent_tokenize(article)]
    
    # Create the named entity chunks: chunked_sentences
    chunked_sentences = nltk.ne_chunk_sents(pos_sentences, binary=True)
    
    print('----------------------Name Entity found - 1st method')
    print("Everything for each sentence...")
    # Test for stems of the tree with 'NE' tags
    NE_found = [[chunk for chunk in sent if (hasattr(chunk, "label") and (chunk.label() == "NE"))] for sent in chunked_sentences]
    print(f'{NE_found}\n')
    
    print('----------------------Name Entity found - 2nd method')
    print("Only Name entity found...")
    chunked_sentences = nltk.ne_chunk_sents(pos_sentences, binary=True)
    for sent in chunked_sentences:
        for chunk in sent:
            if hasattr(chunk, "label") and chunk.label() == "NE":
                print(chunk)
    
    print('----------------------Name Entity found - Representation')
    chunked_sentences = nltk.ne_chunk_sents(pos_sentences, binary=True)
    for sent in chunked_sentences:
        for chunk in sent:
            if hasattr(chunk, "label") and chunk.label() == "NE":
                chunk.pretty_print()
    
    print('------------------------------------------Tags found')
    chunked_sentences = nltk.ne_chunk_sents(pos_sentences, binary=True)
    tags = [[chunk.label() for chunk in sent if hasattr(chunk, 'label')] for sent in chunked_sentences]
    tags = set(list(chain(*tags)))
    print(f'{tags}')
    
    
    
def Charting_practice(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "3. Charting practice"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------------------------Reading the data')
    with open('News articles/articles.txt','r', encoding='utf-8') as f: 
        article = f.read()
    
    # Tag each tokenized sentence into parts of speech for each tokenize sent: pos_sentences
    pos_sentences = [nltk.pos_tag(word_tokenize(sent)) for sent in sent_tokenize(article)]
    
    # Create the named entity chunks: chunked_sentences
    chunked_sentences = list(nltk.ne_chunk_sents(pos_sentences, binary=False))
    print(f'{len(chunked_sentences)} leaves in the tree')  
    
    print('-----------------------------------Name Entity found')
    # Create the defaultdict: ner_categories
    ner_categories = defaultdict(int)
    # Create the nested for loop
    for sent in chunked_sentences:
        for chunk in sent:
            if hasattr(chunk, 'label'):
                ner_categories[chunk.label()] += 1
    print(f'{ner_categories}\n')
                
    print('----------------------------Plotting the information')
    print('---------------------------------------------Explore')
    # Create a list from the dictionary keys for the chart labels: labels
    labels = list(ner_categories.keys())
    # Create a list of the values: values
    #values = [ner_categories.get(v) for v in labels]
    values = list(ner_categories.values())
    
    print(f'Labels: {labels} \nValues: {values}\n')
    
    # Create the pie chart
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.set_title('Categories found in the article - [1st graph]', **title_param)
    fig.suptitle(topic, **suptitle_param)
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=.5) #To set the margins 
    plt.show()
    
    # Create the pie chart
    fig, ax = plt.subplots()
    ax.pie(values, None, startangle=90)
    ax.legend(labels=[f'{l}, {v:1.1%}' for l, v in zip(labels, list(np.array(values)/sum(values)))], loc="upper right", bbox_to_anchor=(.8, 0, 0.5, 1))
    #plt.setp(autotexts, size=8, weight="bold")
    ax.set_title('Categories found in the article - [2nd graph]', **title_param)
    fig.suptitle(topic, **suptitle_param)
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=.85, wspace=None, hspace=.5) #To set the margins 
    plt.show()
    
    
def Introduction_to_SpaCy(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "5. Introduction to SpaCy"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('---------------------------------------------English')
    nlp = spacy.load('en_core_web_sm')
    print(f'Spacy entity: {nlp.entity}')
    
    sentence = """
    Berlin is the capital of Germany;
    and the residence of Chancellor Angela Merkel.
    """
    print(f'My sentence: "{sentence}"')
    doc = nlp(sentence)
    
    print(f'Named entities found: {doc.ents}')
    print('Looking the attribute of the entities found: ')
    for i in range(len(doc.ents)):
        print(doc.ents[i], doc.ents[i].label_)
    
    print('---------------------------------------------Spanish')
    nlp = spacy.load('es_core_news_sm')
    print(f'Spacy entity: {nlp.entity}')
    
    sentence = """
    Berlin es la capital de Alemania;
    y la residencia de la canciller Angela Merkel.
    """
    print(f'My sentence: "{sentence}"')
    doc = nlp(sentence)
    
    print(f'Named entities found: {doc.ents}')
    print('Looking the attribute of the entities found: ')
    for i in range(len(doc.ents)):
        print(doc.ents[i], doc.ents[i].label_)
    
    print('-------------------------Exploring a little bit more')
    print(f'Sentences: {list(doc.sents)}')
    print(f'Dependency labels: {list(doc.noun_chunks)}')
    
    
def Comparing_NLTK_with_spaCy_NER(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "6. Comparing NLTK with spaCy NER"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------------------------Reading the data')
    with open('News articles/uber_apple.txt','r', encoding='utf-8') as f: 
        article = f.read()
    
    print('------------Using spacy for Named Entity Recognition')
    # Instantiate the English model: nlp
    nlp = spacy.load('en_core_web_sm', tagger=False, parser=False, matcher=False) #See for explanations
    
    # Create a new document: doc
    doc = nlp(article)

    print('----------------------Name Entity found - 1st method')
    # Print all of the found entities and their labels
    for ent in doc.ents:
        print(ent.label_, ent.text)
    
    
def spaCy_NER_Categories(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "7. spaCy NER Categories"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('---------------------------------------------Explore')
    print('---------------------------------------------Explore')
    
    
    
def Multilingual_NER_with_polyglot(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "8. Multilingual NER with polyglot"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('-------------------------Familiarizing with Polyglot')
    print('Modules availables: ')
    print(downloader.list(show_packages=False))
    
    print('Languages Coverage in Translation:')
    print(downloader.supported_languages_table("transliteration2"))
    
    print('Task supported in Spanish: ')
    print(downloader.supported_tasks(lang="es"))
    
    print('----------------------------------Detecting language')
    sentences = ['Man acts as though he were the shaper and master of language, while in fact language remains the master of man.',
                 'Der Mensch gebärdet sich, als sei er Bildner und Meister der Sprache, während doch sie die Herrin der Menschen bleibt.']
    for sentence in sentences:
        print(f'My sentence: \n"{sentence}"')
        txt_sentence = Text(sentence)
        print(f'Language detected: \n{txt_sentence.language.code} - {txt_sentence.language.name}\n')
    
    print('-----------------------------Making some traductions')
    sentence = "We will meet at eight o'clock on Thursday morning."
    txt_sentence = Text(sentence)
    print(f'My sentence: \n"{sentence}"')
    
    print('Translate to Arabic:')
    print(' '.join(txt_sentence.transliterate("ar")))
    print('Translate to Russian:')
    print(' '.join(txt_sentence.transliterate("ru")))
    
    """
    ## DOES NOT WORK WITH SPANISH, ITALIAN, FRENCH
    print('Translate to Spanish:')
    print(' '.join(txt_sentence.transliterate("es")))
    print('Translate to Italian:')
    print(' '.join(txt_sentence.transliterate("it")))
    print('Translate to French:')
    print(' '.join(txt_sentence.transliterate("fr")))
    for x in txt_sentence.transliterate("es"):
        print(x)
    """
    #For one word only
    transliterator = Transliterator(source_lang="en", target_lang="ru")
    print('"Love" in russian: ', transliterator.transliterate(u"love"))
    
    print('-------------------------Making sentimental analysis')
    """
    paragraph = [("Barack Obama gave a fantastic speech last night. " 
                  "Reports indicate he will move next to New Hampshire."),
                 ('José Lisandro Cortez obtuvo la mejor calificacion del curso. ' 
                  'Él tiene garantizado el éxito.'),
                 ('EL Parque Daniel Hernández está mal ubicado. '
                  'Tiene espacios sin grama y no suficientes basureros.')]
    """
    paragraph = ["Barack Obama gave a fantastic speech last night. Reports indicate he will move next to New Hampshire.",
                 'José Lisandro Cortez obtuvo la mejor calificacion del curso. Él tiene garantizado el éxito.',
                 'EL Parque Daniel Hernández está mal ubicado. Tiene espacios sin grama y no suficientes basureros.']
    for sentences in paragraph:
        print(f'\nCASE: \n"{sentences}"')
        text = Text(sentences)
        print(f'Language detected: \n{text.language.code} - {text.language.name}')
        for i, sentence in enumerate(text.sentences, start=1):
            for entity in sentence.entities:
                try: 
                    print(f'Entity: {entity}, Positive sentimental: {entity.positive_sentiment}, Negative sentimental: {entity.negative_sentiment}')
                except:
                    print(f'No feeling for entity: {entity}')
    
    print('------------------------------Part of Speech Tagging')
    sentences = [["We will meet at eight o'clock on Thursday morning.", 'en'],
                 ["Der Mensch gebärdet sich, als sei er Bildner und Meister der Sprache, während doch sie die Herrin der Menschen bleibt.", 'de'],
                 ["Él que quiere interesar a los demás tiene que provocarlos. ", 'es']]
    for sentence, lang in sentences:
        #text = Text(sentence)
        # We can also specify language of that text by using
        text = Text(sentence, hint_language_code=lang)
        print(f'My sentence: \n"{sentence}" \n{text.pos_tags}\n')
        
    print('-----------------------------------Word Tokenization')
    sentences = ["""
                 两个月前遭受恐怖袭击的法国巴黎的犹太超市在装修之后周日重新开放，法国内政部长以及超市的管理者都表示，这显示了生命力要比野蛮行为更强大。
                 该超市1月9日遭受枪手袭击，导致4人死亡，据悉这起事件与法国《查理周刊》杂志社恐怖袭击案有关。
                 ""","""
                 José Lisandro Cortez obtuvo la mejor calificacion del curso.
                 """]
    for sentence in sentences:
        print(f'My sentence: \n"{sentence.strip()}"')
        text = Text(sentence)
        print(f'Language detected: \n{text.language.code} - {text.language.name}')
        print(f'Word tokenizer: \n{text.words}\n')
    
    print('---------------------------Spanish NER with polyglot')
    my_text = """
    El presidente de la Generalitat de Cataluña,
    Carles Puigdemont, ha afirmado hoy a la alcaldesa
    de Madrid, Manuela Carmena, que en su etapa de
    alcalde de Girona (de julio de 2011 a enero de 2016)
    hizo una gran promoción de Madrid.
    """
    ptext = Text(my_text)
    print(f'My text: {my_text} \nEntities found: \n{ptext.entities}')
    
    
    
def French_NER_with_polyglot_I(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "9. French NER with polyglot I"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------------------------Reading the data')
    with open('News articles/french.txt','r', encoding='utf-8') as f: 
        article = f.read()
    
    print('-----------------------------------NER with polyglot')
    # Create a new text object using Polyglot's Text class: txt
    txt = Text(article)
    print(f'Language detected: \n{txt.language.code} - {txt.language.name}\n')
    
    # Print each of the entities found
    for ent in txt.entities: print(f'{ent.tag}: {ent}')
    
    # Print the type of ent
    print('Type of each entity found: ', type(ent))
    
    
    
def French_NER_with_polyglot_II(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "10. French NER with polyglot II"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------------------------Reading the data')
    with open('News articles/french.txt','r', encoding='utf-8') as f: 
        article = f.read()
    
    print('-----------------------------------NER with polyglot')
    # Create a new text object using Polyglot's Text class: txt
    txt = Text(article)
    print(f'Language detected: \n{txt.language.code} - {txt.language.name}\n')
    
    # Create the list of tuples: entities
    entities = [(ent.tag, ' '.join(ent)) for ent in txt.entities]
    
    # Print entities
    print(entities)
    
    
def Spanish_NER_with_polyglot(size=SIZE, seed=SEED):
    print("****************************************************")
    topic = "11. Spanish NER with polyglot"; print("** %s" % topic)
    print("****************************************************")
    
    # Initialize seed and parameters
    np.random.seed(seed) 
    
    print('------------------------------------Reading the data')
    ## Text file --> It is not exactly the same
    #with open('spanish_blog.txt','r', encoding='utf-8') as f: 
    #    article = f.read()
    
    # URL data
    # Specify url: url
    url="https://sites.utexas.edu/ransomcentermagazine/2015/10/27/los-mundos-politicos-de-gabriel-garcia-marquez-con-lina-del-castillo/"
    
    # Package the request, send the request and catch the response: r
    r = requests.get(url)

    # Extracts the response as html: html_doc
    html_doc = r.text

    # Create a BeautifulSoup object from the HTML: soup
    article = BeautifulSoup(html_doc, 'html.parser').get_text()
    
    print('-----------------------------------NER with polyglot')
    # Create a new text object using Polyglot's Text class: txt
    txt = Text(article)
    print(f'Language detected: \n{txt.language.code} - {txt.language.name}\n')
    
    # Print each of the entities found
    for ent in txt.entities: print(f'{ent.tag}: {ent}')
    
    print('-----------Checkinf if contains "Márquez" or "Gabo".')
    entities_filtered = [ent for ent in txt.entities if (('Márquez' in ent) or ('Gabo' in ent))]
    count = len(entities_filtered)
    # Print count
    print(f'{count} entities found.')
    
    # Calculate the percentage of entities that refer to "Gabo": percentage
    percentage = count / len(txt.entities)
    print(f'{percentage:.2%}')

    
        
def main():
    print("****************************************************")
    print("** BEGIN                                          **")
    
    Named_Entity_Recognition()
    NER_with_NLTK()
    Charting_practice()
    
    Introduction_to_SpaCy()
    Comparing_NLTK_with_spaCy_NER()
    spaCy_NER_Categories()
    Multilingual_NER_with_polyglot()
    French_NER_with_polyglot_I()
    French_NER_with_polyglot_II()
    Spanish_NER_with_polyglot()
    
    print("****************************************************")
    print("** END                                            **")
    print("****************************************************")
    
    
    
if __name__ == '__main__':
    main()
    pd.reset_option("display.max_columns")
    plt.style.use('default')
    np.set_printoptions(formatter={'float': None})