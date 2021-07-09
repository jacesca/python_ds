# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 22:54:52 2021

@author: jaces
"""

# Importing the requires libraries
import os
import shutil
import numpy as np

import speech_recognition as sr

from pydub import AudioSegment
from pydub.playback import play
from pydub.effects import normalize
from pydub.silence import split_on_silence

#import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

from punctuator import Punctuator

import spacy
from spacy.pipeline import EntityRuler

from pprint import pprint

import pandas as pd

# Import text classification packages
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split


#Global variables
SEED              = 42
PUNCTUATOR_MODULE = r'C:\Anaconda3\PUNCTUATOR_DATA_DIR\INTERSPEECH-T-BRNN.pcl'

# Global configuration
np.random.seed(SEED)

# Global functions
def convert_to_wav(filename):
    "Takes an audio file of non .wav format and converts to .wav"
    # Import audio file
    audio = AudioSegment.from_file(filename)
    # Increase the volume by 10 dB
    audio = audio + 30
    # Improving the quality
    audio = normalize(audio)
    # Create new filename
    new_filename = filename.split(".")[0] + ".wav"
    # Export file as .wav
    audio.export(new_filename, format="wav")
    print(f"Converting {filename} to {new_filename}...")
    return new_filename

def show_pydub_stats(filename):
    "Returns different audio attributes related to an audio file."
    # Create AudioSegment instance
    audio_segment = AudioSegment.from_file(filename)
    # Print attributes
    print(f"Channels: {audio_segment.channels}")
    print(f"Sample width: {audio_segment.sample_width}")
    print(f"Frame rate (sample rate): {audio_segment.frame_rate}")
    print(f"Frame width: {audio_segment.frame_width}")
    print(f"Length (ms): {len(audio_segment)}")
    print(f"Frame count: {audio_segment.frame_count()}")
    return audio_segment

def transcribe_audio(filename, language, energy_threshold=300, duration=None, offset=None, show_all=None, noise=0):
    "Takes a .wav format audio file and transcribes it to text."
    # Setup a recognizer instance
    recognizer = sr.Recognizer()
    # Import the audio file and convert to audio data
    audio_file = sr.AudioFile(filename)
    with audio_file as source:
         # Adjust for ambient noise and record
        if noise>0:
            recognizer.adjust_for_ambient_noise(source, duration=noise)
        # Record the audio
        audio_data = recognizer.record(source,
                                       duration=duration, #Listen from the begining to duration value.
                                       offset=offset) #used to skip over a specific seconds at the start.
    # Set the energy threshold
    recognizer.energy_threshold = energy_threshold
    # Return the transcribed text
    return recognizer.recognize_google(audio_data, language=language, show_all=show_all)

def transcribe_long_audio(file, language, 
                          energy_threshold=300, duration=None, offset=None, show_all=None, noise=0,
                          chunk_folder=r'acme_studios_audio\temp'):
    """
    Splitting the large audio file into chunks and apply speech recognition on each of these chunks
    """
    # create a speech recognition object
    recognizer = sr.Recognizer()
    # open the audio file using pydub
    audio_file = AudioSegment.from_file(file)  
    
    # split audio_file where silence is 700 miliseconds or more and get chunks
    chunks = split_on_silence(audio_file,
                              min_silence_len = 500, # experiment with this value for your target audio file
                              silence_thresh = audio_file.dBFS-14, # adjust this per requirement
                              keep_silence=500, # keep the silence for 1 second, adjustable as well
                             )
    # create a directory to store the audio chunks
    if os.path.isdir(chunk_folder):
        shutil.rmtree(chunk_folder)
    os.mkdir(chunk_folder)
    whole_text = ""
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in the `folder_name` directory.
        chunk_filename = os.path.join(chunk_folder, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        # recognize the chunk
        with sr.AudioFile(chunk_filename) as source:
            # Adjust for ambient noise and record
            if noise>0:
                recognizer.adjust_for_ambient_noise(source, duration=noise)
            # Record the audio
            audio_listened = recognizer.record(source,
                                               duration=duration, #Listen from the begining to duration value.
                                               offset=offset) #used to skip over a specific seconds at the start.
            # Set the energy threshold
            recognizer.energy_threshold = energy_threshold
            # try converting it to text
            try:
                text = recognizer.recognize_google(audio_listened, language=language)
            except sr.UnknownValueError as e:
                print(f"No audio detect {str(e)}")
            else:
                whole_text += f'{text} '
    # return the text for all chunks detected
    return whole_text

def split_to_mono(filename, sel_channel):
    'Split audio and return specified channel'
    # Create AudioSegment instance
    audio_segment = AudioSegment.from_file(filename)
    # Verify the number of channels
    number_of_channels = audio_segment.channels
    if sel_channel < number_of_channels:
        # Split call_1 to mono
        audio_channels = audio_segment.split_to_mono()
        # Export channel 2 (the customer channel)
        file_sel_channel = filename.split('.')[0] + '_channel_2.wav'
        audio_channels[sel_channel].export(file_sel_channel, format='wav')
    else:
        print(f'Error: unmatched audio file, selected channel: {sel_channel+1}°, ' + \
              f'number of channels in the audio file: {number_of_channels}.')
        file_sel_channel = None
    return file_sel_channel

# Transcribe text from wav files
def create_text_list(folder):
    text_list, text_label = [], []
    p = Punctuator(PUNCTUATOR_MODULE)
    # Loop through folder
    for file in os.listdir(folder):
        # Check for .wav extension
        if file.endswith(".wav"):
            # Transcribe audio
            text = transcribe_long_audio(folder + file, language='en-US')
            # Add punctuation marks
            text = p.punctuate(text)
            # Add transcribed text to list
            text_list.append(text)
            text_label.append('post_purchase' if 'post_purchase' in file else 'pre_purchase')                
    return text_label, text_list

print('*********************************************************')
print('BEGIN')
print('*********************************************************')
print('** 04 Processing text transcribed from spoken language**')
print('*********************************************************')
print('** 04.01 Creating transcription helper functions')
print('*********************************************************')
# Check the folder of audio files
folder = "./"
print(os.listdir(folder))

# For the following examples, we use ex4_call_1_stereo_mp3.mp3 audio file
file = "acme_studios_audio/ex4_call_1_stereo_mp3.mp3"
wav_file = "acme_studios_audio/ex4_call_1_stereo_mp3.wav"

# Import audio file
audio = AudioSegment.from_file(file)
play(audio)

print('*********************************************************')
print('** 04.02 Converting audio to the right format')
print('*********************************************************')
file = "acme_studios_audio/ex4_call_1_stereo_formatted_mp3.mp3"

# Using the file format conversion function
wav_file = convert_to_wav(file)


print('*********************************************************')
print('** 04.03 Finding PyDub stats')
print('*********************************************************')
# Using the attribute showing function
audio = show_pydub_stats(wav_file)

print('*********************************************************')
print('** 04.04 Transcribing audio with one line')
print('*********************************************************')
print('Complete audio:')
print(transcribe_audio(wav_file, language='en-GB'))
print('\nAudio divided in chunks:')
print(transcribe_long_audio(wav_file, language='en-US'))

print('*********************************************************')
print("** 04.05 Using the helper functions you've built")
print('*********************************************************')
file = "acme_studios_audio/ex4_call_1_stereo_formatted_mp3.mp3"

# Convert mp3 file to wav
print('Converting to wav format...')
wav_file = convert_to_wav(file)

# Check the stats of new file
print('\nChecking its stats...')
call_1 = show_pydub_stats(wav_file)

# Split call_1 to mono
print('\nGetting the client audio (channnel 2)...')
file_channel2 = call_1_split = split_to_mono(wav_file, 1)

# Transcribe the single channel
print('\nTranscribing the client audio...')
call_1_channel_2_text = transcribe_long_audio(file_channel2, language='en-US')
print(call_1_channel_2_text)

print('*********************************************************')
print("** 04.06 Sentiment analysis on spoken language text")
print('*********************************************************')
# Create sentiment analysis instance
sid = SentimentIntensityAnalyzer()
# Sentiment analysis on customer channel of call_1
print('Sentiment analysis on customer channel of call_1:')
print(sid.polarity_scores(call_1_channel_2_text))

# Trying with puntuator module
print('\nAdding punctuation marks... (English only):')
p = Punctuator(PUNCTUATOR_MODULE)
call_1_channel_2_text_with_punctuation = p.punctuate(call_1_channel_2_text)
print(call_1_channel_2_text_with_punctuation)

# Find sentiment on each sentence
print('\nRepeting sentiment analysis again (sentence by sentence):')
for sentence in sent_tokenize(call_1_channel_2_text_with_punctuation):
    print(sentence)
    print(sid.polarity_scores(sentence))

print('*********************************************************')
print("** 04.07 Analyzing sentiment of a phone call")
print('*********************************************************')
file = "acme_studios_audio/ex4_call_2_stereo_native.wav"

# Check the stats of new file
print('Checking its stats...')
audio = show_pydub_stats(file)

# Play audio file
#print('\nPlaying the audio file...')
#play(audio)

# Transcribe the audio
print('\nTranscribing the audio...')
file_text = transcribe_long_audio(file, language='en-US')
print(file_text)

# Add punctuation to the text
print('\nAdding punctuation marks... (English only):')
p = Punctuator(PUNCTUATOR_MODULE)
file_text_with_punctuation = p.punctuate(file_text)
print(file_text_with_punctuation)

# Find sentiment on each sentence
print('\nSentiment analysis on customer channel of call_1:')
for sentence in sent_tokenize(file_text_with_punctuation):
    print(sentence)
    print(sid.polarity_scores(sentence))

print('*********************************************************')
print("** 04.08 Sentiment analysis on formatted text")
print('*********************************************************')
file = "acme_studios_audio/ex4_call_1_stereo_formatted_mp3.wav"

# Split call to mono
print('\nGetting the client audio (channnel 2)...')
file_one_channel = split_to_mono(wav_file, 1)

# Transcribe customer channel of call 2
print('\nTranscribing the audio...')
file_one_channel_text = transcribe_long_audio(file_one_channel, language='en-US')
print(file_one_channel_text)

# Add punctuation marks
print('\nAdding punctuation marks... (English only):')
p = Punctuator(PUNCTUATOR_MODULE)
file_one_channel_text_with_punctuation = p.punctuate(file_one_channel_text)
print(file_one_channel_text_with_punctuation)

# Find sentiment on each sentence
print('\nSentiment analysis on each sentence:')
for sentence in sent_tokenize(file_one_channel_text_with_punctuation):
    print(sentence)
    print(sid.polarity_scores(sentence))

print('*********************************************************')
print("** 04.09 Named entity recognition on transcribed text")
print('*********************************************************')
# Load spaCy language model
print('Loading the spacy model...')
nlp = spacy.load("en_core_web_sm")

# Create a spaCy doc
texto = "I'd like to talk about a smartphone I ordered on July 31st from your " + \
        "Sydney store, my order number is 40939440. I spoke to Georgia about it last week."
doc = nlp(texto)
print(f'\nText to analize: \n{doc}')
          
# Show different tokens and positions
print('\nTokens found...')
for token in doc:
    print(token.text, token.idx)
    
# Show sentences in doc
print('\nSentences:')
for i, sentence in enumerate(doc.sents, start=1):
    print(f'({i}) {sentence}')
    
# Find named entities in doc
print('\Entities found...')
for entity in doc.ents:
    print(entity.text, entity.label_)

# Check spaCy pipeline
print('\nThe actual pipeline:')
pprint(nlp.pipeline)

# Custom named entities
print('\nCustomizing named entities...')
# Create EntityRuler instance
ruler = EntityRuler(nlp)
# Add token pattern to ruler
ruler.add_patterns([{"label":"PRODUCT", "pattern": "smartphone"}])
# Add new rule to pipeline before ner
nlp.add_pipe(ruler, before="ner")
# Check updated pipeline
pprint(nlp.pipeline)

# Testing the new pipeline
print('\nTesting the new pipeline...')
doc = nlp(texto)
          
# Test new entity rule
for entity in doc.ents:
    print(entity.text, entity.label_)

print('*********************************************************')
print("** 04.10 Named entity recognition in spaCy")
print('*********************************************************')
file = "acme_studios_audio/ex4_call_4_channel_2_formatted.wav"

# Load spaCy language model
print('Loading the spacy model...')
nlp = spacy.load("en_core_web_sm")

# Transcribe the audio
print('\nTranscribing the audio...')
file_text = transcribe_long_audio(file, language='en-US')
print(file_text)

# Create a spaCy doc with call 4 channel 2 text
doc = nlp(file_text)

# Check the type of doc
print('Type:', type(doc))

# Show tokens in doc
print('\nTokens:')
for token in doc:
    print(token.text, token.idx)

# Show sentences in doc
print('\nSentences:')
for i, sentence in enumerate(doc.sents, start=1):
    print(f'({i}) {sentence}')

# Show named entities and their labels
print('\nEntities:')
for entity in doc.ents:
    print(entity.text, entity.label_)

print('*********************************************************')
print("** 04.11 Creating a custom named entity in spaCy")
print('*********************************************************')
# Create EntityRuler instance
ruler = EntityRuler(nlp)

# Add token pattern to ruler
ruler.add_patterns([{"label":"PRODUCT", "pattern": "smart"}])

# Add new rule to pipeline before ner
nlp.add_pipe(ruler, before="ner")
print('The new pipeline:')
print(nlp.pipeline)

# Custom named entities
print('\nApplying the new pipeline...')
doc = nlp(file_text)
          
# Test new entity rule
for entity in doc.ents:
    print(entity.text, entity.label_)
    
print('*********************************************************')
print("** 04.12 Classifying transcribed speech with Sklearn")
print('*********************************************************')
print("** 04.13 Preparing audio files for text classification")
print('*********************************************************')
print("** 04.14 Transcribing phone call excerpts")
print('*********************************************************')
print("** 04.15 Organizing transcribed phone call data")
print('*********************************************************')
print("** 04.16 Create a spoken language text classifier")
print('*********************************************************')
print("** 04.17 Congratulations")
print('*********************************************************')
# Inspect post purchase audio folder
print('Inspecting the data...')
folder = 'purchases_audio/'

post_purchase_audio = os.listdir(folder)
print(post_purchase_audio)

# Loop through mp3 files
print('\nConverting to wav')
for file in post_purchase_audio:
    if not file.endswith(".wav"):
        # Use previously made function to convert to .wav
        wav_file = convert_to_wav(folder + file)

# Transcribing all phone call excerpts
print('\nTranscribing phone calls')
purchase_label, purchase_text = create_text_list(folder)
print(purchase_label)
print(purchase_text)

# Building a text classier
print('\nReading data classifier...')
# Create post purchase dataframe
df = pd.read_csv('customer_call_transcriptions.csv')
print(df.head())

# Building a text classifier
print('\nBuilding a text classier')
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.3, random_state=SEED)

# Create text classifier pipeline
text_classifier = Pipeline([("vectorizer", CountVectorizer()),
                            ("tfidf", TfidfTransformer()),
                            ("classifier", MultinomialNB())
                            ])

# Fit the classifier pipeline on the training data
text_classifier.fit(X_train, y_train)

# Make predictions and compare them to test labels
predictions = text_classifier.predict(X_test)
accuracy = 100 * np.mean(predictions == y_test)
print(f"The model is {accuracy:.2f}% accurate.")

print('*********************************************************')
print('END')
print('*********************************************************')