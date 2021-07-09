# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 20:28:11 2021

@author: jaces
"""

# importing libraries 
import speech_recognition as sr 

import os
import shutil
import sys

from pydub import AudioSegment
from pydub.silence import split_on_silence

import pydub

# a function that splits the audio file into chunks and applies speech recognition
def get_large_audio_transcription(file, language, energy_threshold=300, duration=None, offset=None, show_all=None, noise=0):
    """
    Splitting the large audio file into chunks and apply speech recognition on each of these chunks
    """
    
    # create a speech recognition object
    recognizer = sr.Recognizer()

    # open the audio file using pydub
    if '.wav' in file:
        audio_file = AudioSegment.from_wav(file)
    elif '.mp3' in file:
        audio_file = AudioSegment.from_file(file, "mp3")
    elif '.ogg' in file:
        audio_file = AudioSegment.from_ogg(file)
    elif '.flv' in file:
        audio_file = AudioSegment.from_flv(file)
    elif '.mp4' in file:
        audio_file = AudioSegment.from_file(file, "mp4")
    elif '.wma' in file:
        audio_file = AudioSegment.from_file(file, "wma")
    elif '.aiff' in file:
        audio_file = AudioSegment.from_file(file, "aac")
    else:
        audio_file = AudioSegment.from_file(file)  
    
    # split audio_file where silence is 700 miliseconds or more and get chunks
    chunks = split_on_silence(audio_file,
                              min_silence_len = 500, # experiment with this value for your target audio file
                              silence_thresh = audio_file.dBFS-14, # adjust this per requirement
                              keep_silence=500, # keep the silence for 1 second, adjustable as well
                             )
    folder_name = "audio-chunks-temp"
    
    # create a directory to store the audio chunks
    if os.path.isdir(folder_name):
        shutil.rmtree(folder_name)
    os.mkdir(folder_name)
    
    whole_text = ""
    with open(file.split('.')[0]+'.txt', 'w') as f:
        # process each chunk 
        for i, audio_chunk in enumerate(chunks, start=1):
            # export audio chunk and save it in the `folder_name` directory.
            chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
            
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
                    msg_to_write = f"Error: {str(e)}\n" 
                    print(msg_to_write)
                else:
                    text = f"{text.capitalize()}. "
                    msg_to_write = f"{i}: {text}\n"
                    print(chunk_filename, ":", text)
                    whole_text += text
                
                # Write the text to the file
                f.write(msg_to_write)
    
    # return the text for all chunks detected
    return whole_text

path = "multiple-speakers-16k.wav"
print("\nFull text:", get_large_audio_transcription(path, language='en-US'))

path = "AUD-20180918-WA0000.mp3"
print("\nFull text:", get_large_audio_transcription(path, language='es-US'))