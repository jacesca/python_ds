# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 21:44:34 2021

@author: jaces
"""

# Importing the requires libraries
import speech_recognition as sr

print('*********************************************************')
print('BEGIN')
print('*********************************************************')
print('** 02. Using the Python SpeechRecognition library')
print('*********************************************************')
print('** 02.01 SpeechRecognition Python library')
print('*********************************************************')
def get_audio_file(wav_file, duration=None, offset=None, noise=0):
    # Setup recognizer instance
    recognizer = sr.Recognizer()
    
    # Read in audio file
    clean_support_call = sr.AudioFile(wav_file)

    # Check type of clean_support_call
    #print('Object type:', type(clean_support_call))
    
    # Convert from AudioFile to AudioData
    with clean_support_call as source:
        # Adjust for ambient noise and record
        if noise>0:
            recognizer.adjust_for_ambient_noise(source, duration=noise)
        
        # Record the audio
        clean_support_call_audio = recognizer.record(source,
                                                     duration=duration, #Listen from the begining to duration value.
                                                     offset=offset) #used to skip over a specific seconds at the start.
        
    return clean_support_call_audio

audio_file = get_audio_file('good_morning.wav')
print('Object type:', type(audio_file))

def get_transcription(file, language, energy_threshold=300, duration=None, offset=None, show_all=None, noise=0):
    # get the audio data
    audio_data = get_audio_file(file, duration=duration, offset=offset, noise=0)

    # Create an instance of Recognizer
    recognizer = sr.Recognizer()
    
    # Set the energy threshold
    recognizer.energy_threshold = energy_threshold
    
    # Transcribe speech using Goole web API
    return recognizer.recognize_google(audio_data=audio_data, language=language, show_all=show_all)

files = ['good_morning_jacesca.wav', 'buenos_dias.wav', 'it_example.wav', 'fr_example.wav']
langs = ['en-US', 'es-US', 'it-IT', 'fr-CA']

for file, lang in zip(files, langs):
    print(f"Language {lang}: ", get_transcription(file, lang))

print('*********************************************************')
print('** 02.02 Pick the wrong speech_recognition API')
print('*********************************************************')
print('** 02.03 Using the SpeechRecognition library')
print('*********************************************************')
print('** 02.04 Using the Recognizer class')
print('*********************************************************')
lang, file = 'en-US', 'clean-support-call.wav'
print(f"Language {lang}: ", get_transcription(file, lang))

print('*********************************************************')
print('** 02.05 Reading audio files with SpeechRecognition')
print('*********************************************************')
# Get first 2-seconds of clean support call
print("First 2 seconds: ", get_transcription(file, lang, duration=2.0))

# Skip first 2-seconds of clean support call
print("After 2 seconds: ", get_transcription(file, lang, offset=2.0))

print('*********************************************************')
print('** 02.06 From AudioFile to AudioData')
print('*********************************************************')
lang, file = 'en-US', 'clean-support-call.wav'
print(f"Language {lang}: ", get_transcription(file, lang))

print('*********************************************************')
print('** 02.07 Recording the audio we need')
print('*********************************************************')
# Get first 2-seconds of clean support call
lang, file = 'en-US', '30-seconds-of-nothing-16k.wav'
print("Duration = 10 seconds: ", get_transcription(file, lang, duration=10))

# Skip first 2-seconds of clean support call
lang, file = 'en-US', 'static-out-of-warranty.wav'
print("Offset   =  3 seconds: ", get_transcription(file, lang, offset=3))

print('*********************************************************')
print('** 02.08 Dealing with different kinds of audio')
print('*********************************************************')
# Pass the audio to recognize_google

print('** What language?')
files = ['good-morning-japanense.wav', 'good-morning-japanense.wav', 'multiple-speakers-16k.wav']
langs = ['en-US', 'ja-JP', 'en-US']

for file, lang in zip(files, langs):
    print(f"Language {lang}: ", get_transcription(file, lang))


print('** Showing all')
files = ['good-morning-japanense.wav', 'leopard.wav']
langs = ['en-US', 'en-US']

for file, lang in zip(files, langs):
    print(f"Language {lang}: ", get_transcription(file, lang, show_all=True))


print('** Multiple speakers')    
files = ['speaker_0.wav', 'speaker_1.wav', 'speaker_2.wav']
langs = ['en-US', 'en-US', 'en-US']

for i, (f, l) in enumerate(zip(files, langs)):
    print(f"Text from speaker {i}: ", get_transcription(f, l))
    

# Import audio file with background nosie
print('** Noisy audio')
lang, file = 'en-US', '2-noisy-support-call.wav'
print(f"Language {lang}: ", get_transcription(file, lang, noise=0.5))


print('*********************************************************')
print('** 02.09 Different kinds of audio')
print('*********************************************************')
# Pass the audio to recognize_google
files = ['good-morning-japanense.wav', 'good-morning-japanense.wav', 'leopard.wav', 'charlie-bit-me-5.wav']
langs = ['en-US', 'ja-JP', 'en-US', 'en-US']

for file, lang in zip(files, langs):
    print(f"Language {lang}: ", get_transcription(file, lang, show_all=True))


print('*********************************************************')
print('** 02.10 Multiple Speakers 1')
print('*********************************************************')
lang, file = 'en-US', 'multiple-speakers-16k.wav'
print(f"Language {lang}: ", get_transcription(file, lang, noise=0.5))


print('*********************************************************')
print('** 02.11 Multiple Speakers')
print('*********************************************************')
files = ['speaker_0.wav', 'speaker_1.wav', 'speaker_2.wav']
lang = 'en-US'

for i, audio in enumerate(files):
    print(f"Text from speaker {i}: ", get_transcription(audio, lang))


print('*********************************************************')
print('** 02.12 Working with noisy audio')
print('*********************************************************')
# Pass the audio to recognize_google
files = ['clean-support-call.wav', '2-noisy-support-call.wav', '2-noisy-support-call.wav', '2-noisy-support-call.wav']
show_all = [False, True, True, True]
noise = [0, 0, 1, 0.5]
lang = 'en-US'

for i, (f, s, n) in enumerate(zip(files, show_all, noise)):
    print(f"Ex.{i}: ", get_transcription(f, lang, show_all=s, noise=n))


print('*********************************************************')
print('END')
print('*********************************************************')
