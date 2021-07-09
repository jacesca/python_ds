# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 20:41:07 2021

@author: jaces
"""

# Importing the requires libraries
import wave
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Global variables
suptitle_param = dict(color='darkblue', fontsize=11)
title_param    = {'color': 'darkred', 'fontsize': 14, 'weight': 'bold'}
plot_param     = {'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 
                  'legend.fontsize': 8, 'font.size': 8}
cbar_param     = {'fontsize':12, 'labelpad':20, 'color':'maroon'}
figsize        = (12.1, 5.9)
SEED           = 42

# Global configuration
sns.set()
plt.rcParams.update(**plot_param)
np.random.seed(SEED)

print('*********************************************************')
print('BEGIN')
print('*********************************************************')
print('** 01. Introduction to Spoken Language Processing with Python**')
print('*********************************************************')
print('** 01.01 Introduction to audio data in Python')
print('*********************************************************')
# Import audio file as wave object
good_morning = wave.open("good_morning.wav", "r")
print(f'Audio type: {good_morning}\n')


# Convert wave object to bytes
print('Converting wave object to bytes..')
soundwave_gm = good_morning.readframes(-1)
print(type(soundwave_gm))
print(soundwave_gm[:10])


print('*********************************************************')
print('** 01.02 The right frequency')
print('*********************************************************')
print('** 01.03 Importing an audio file with Python')
print('*********************************************************')

# Create audio file wave object
good_morning = wave.open("good_morning.wav", 'r')

# Read all frames from wave object 
signal_gm = good_morning.readframes(-1)

# View first 10
print(signal_gm[:10])


print('*********************************************************')
print('** 01.04 Converting sound wave bytes to integers')
print('*********************************************************')
# Convert soundwave_gm from bytes to integers
signal_gm = np.frombuffer(soundwave_gm, dtype='int16')

# Show the first 10 items
print('Converting bytes to integers (first 10 numbers):', signal_gm[:10])
print('Number of information peaces (leb) of audio file:', len(signal_gm))

print('Finding the frame rate:')
# Get the frame rate
framerate_gm = good_morning.getframerate()

# Show the frame rate
print(framerate_gm)

#Duration of audio file (seconds) = length of wave objects / frequency (Hz)
print('Duration: ', len(signal_gm) / framerate_gm, 'sec.')

print('Finding sound wave timestamps:')
# Get the timestamps of the good morning sound wave
time_gm = np.linspace(start = 0, 
                      stop  = len(soundwave_gm)/framerate_gm, 
                      num   = len(soundwave_gm))
print(time_gm)


print('*********************************************************')
print('** 01.05 The right data type')
print('*********************************************************')
print('** 01.06 Bytes to integers')
print('*********************************************************')
# Open good morning sound wave and read frames as bytes
good_morning = wave.open('good_morning.wav', 'r')
signal_gm = good_morning.readframes(-1)

# Convert good morning audio bytes to integers
soundwave_gm = np.frombuffer(signal_gm, dtype='int16')

# View the first 10 sound wave values
print('Converting bytes to integers (first 10 numbers):', soundwave_gm[:10])

print('*********************************************************')
print('** 01.07 Finding the time stamps')
print('*********************************************************')
# Read in sound wave and convert from bytes to integers
good_morning = wave.open('good_morning.wav', 'r')
signal_gm = good_morning.readframes(-1)
soundwave_gm = np.frombuffer(signal_gm, dtype='int16')

# Get the sound wave frame rate
framerate_gm = good_morning.getframerate()

# Find the sound wave timestamps
time_gm = np.linspace(start=0,
                      stop=len(soundwave_gm)/framerate_gm,
                      num=len(soundwave_gm))

# Print the first 10 timestamps
print('First ten: ', time_gm[:10])


print('*********************************************************')
topic = '01.08 Visualizing sound waves'; print(f'** {topic}');
print('*********************************************************')
# Read good_afternoon.wav
good_afternoon = wave.open('good_afternoon.wav', 'r')
signal_ga = good_afternoon.readframes(-1)
soundwave_ga = np.frombuffer(signal_ga, dtype='int16')

# Get the sound wave frame rate
framerate_ga = good_afternoon.getframerate()

# Find the sound wave timestamps
time_ga = np.linspace(start=0,
                      stop=len(soundwave_ga)/framerate_ga,
                      num=len(soundwave_ga))

# Print the first 10 timestamps
print('Time stamps, first ten: ', time_ga[:10])

# Initialize figure and setup title
fig, ax = plt.subplots()
ax.set_title("Good Afternoon vs. Good Morning")

# x and y axis labels
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Amplitude")

# Add good morning and good afternoon values
ax.plot(time_ga, soundwave_ga, label ="Good Afternoon", alpha=0.5)
ax.plot(time_gm, soundwave_gm, label="Good Morning", alpha=0.5)

# Create a legend and show our plot
ax.legend()

fig.suptitle(topic, **suptitle_param)
plt.subplots_adjust(left=None, bottom=.45, right=None, top=.85, wspace=None, hspace=None) #To set the margins 
plt.show()

print('*********************************************************')
print('** 01.09 Staying consistent')
print('*********************************************************')
topic = '01.10 Processing audio data with Python'; print(f'** {topic}');
print('*********************************************************')

# Define a funtion to make the transformations
def transform_wav_files(wav_file):
    # Read the wav file
    wf = wave.open(wav_file, 'r')
    signal = wf.readframes(-1)
    soundwave = np.frombuffer(signal, dtype='int16')

    # Get the sound wave frame rate
    framerate = wf.getframerate()

    # Find the sound wave timestamps
    time = np.linspace(start=0, stop=len(soundwave)/framerate, num=len(soundwave))
    
    return wf, soundwave, time


wav_files_to_compare = ['good_morning.wav', 'good_morning_jacesca.wav']


# Initialize figure and setup title
fig, ax = plt.subplots()

# Setup the title
ax.set_title(' vs '.join(wav_files_to_compare))

# x and y axis labels
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Amplitude")

for file in wav_files_to_compare:
    # Get data from file
    wf, soundwave, time = transform_wav_files(file)
    
    # Add data to the plot
    ax.plot(time, soundwave, label=file, alpha=0.5)

# Create a legend and show our plot
ax.legend()

fig.suptitle(topic, **suptitle_param)
plt.subplots_adjust(left=None, bottom=.45, right=None, top=.85, wspace=None, hspace=None) #To set the margins 
plt.show()


print('*********************************************************')
print('END')
print('*********************************************************')
