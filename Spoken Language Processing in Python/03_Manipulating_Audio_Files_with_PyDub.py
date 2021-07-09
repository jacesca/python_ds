# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 22:12:48 2021

@author: jaces
"""

# Importing the requires libraries
import ffmpeg
from pydub import AudioSegment
from pydub.effects import normalize
from pydub.playback import play

import speech_recognition as sr

import os

from pprint import pprint

# Own libraries
from Transcribe_speech_to_text import get_transcription


# Global functions
def make_wav(wrong_folder_path, right_folder_path):
    print("Let's begin!")
    # Loop through wrongly formatted files
    for file in os.scandir(wrong_folder_path):
        
        # Only work with files with audio extensions we're fixing
        if file.path.endswith(".mp3") or file.path.endswith(".flac"):
        
            # Create the new .wav filename
            out_file = right_folder_path + os.path.splitext(os.path.basename(file.path))[0] + ".wav"

            # Read in the audio file and export it in wav format
            AudioSegment.from_file(file.path).export(out_file, format="wav")
            print(f"Creating {out_file}")
    print('End.')


print('*********************************************************')
print('BEGIN')
print('*********************************************************')
print('** 03. Manipulating Audio Files with PyDub**')
print('*********************************************************')
print('** 03.01 Introduction to PyDub')
print('*********************************************************')
# Import an audio file
filename = "good_morning.wav"
wav_file = AudioSegment.from_file(file=filename, format="wav")

print('Type: ', type(wav_file))

# Play audio file
print('Playing the file:', filename)
play(wav_file)

# show you the number of channels, 1 for mono, 2 for stereo audio
print('Channels: ', wav_file.channels)

# Getting the sample of our AudioSegment in Hertz
print('Hertz of the file:', wav_file.frame_rate)

# Find the number of bytes per sample, 1 means 8-bit, 2 means 16-bit
print('Bytes per sample:', wav_file.sample_width)

# Find the max amplitude of your audio file, which can be considered loudness and is useful for normalizing sound levels
print('Max amplitude:', wav_file.max)

# Duration of audio file in milliseconds
print('Duration:', len(wav_file), 'ms')
print('Duration:', len(wav_file)*0.001, 's\n')


print('** Changing audio parameters')

# Change sample width to 1
print('Original width (Bytes per sample):', wav_file.sample_width)
wav_file_width_1 = wav_file.set_sample_width(1)
print('Width after change (Bytes per sample):', wav_file_width_1.sample_width)


# Change sample rate
print('Original sample rate:', wav_file.frame_rate)
wav_file_16k = wav_file.set_frame_rate(16000)
print('Sample rate after change:', wav_file_16k.frame_rate)


# Change number of channels
print('Original number of channels:', wav_file.channels)
wav_file_1_channel = wav_file.set_channels(1)
print('Number of channels after change:', wav_file_1_channel.channels)


print('*********************************************************')
print('** 03.02 Import an audio file with PyDub')
print('*********************************************************')
# Create an AudioSegment instance
wav_file = AudioSegment.from_file(file='wav_file.wav', 
                                  format="wav")

# Check the type
print('Type:', type(wav_file))


print('*********************************************************')
print('** 03.03 Play an audio file with PyDub')
print('*********************************************************')
print('** 03.04 Audio parameters with PyDub')
print('*********************************************************')
# Find the frame rate
print('Frame rate:', wav_file.frame_rate)

# Find the number of channels
print('Number of channels:', wav_file.channels)

# Find the max amplitude
print('Max amplitude:', wav_file.max)

# Find the length
# Duration of audio file in milliseconds
print('Duration:', len(wav_file), 'ms')
print('Duration:', len(wav_file)*0.001, 's')


print('*********************************************************')
print('** 03.05 Adjusting audio parameters')
print('*********************************************************')
# Adjusted frame rate
print("Old frame rate: ", wav_file.frame_rate)
wav_file_16k = wav_file.set_frame_rate(16000)
print("New frame rate: ", wav_file_16k.frame_rate)

# Set number of channels to 1
print("Old number of channels: ", wav_file.channels)
wav_file_1_ch = wav_file.set_channels(1)
print("New number of channels: ", wav_file_1_ch.channels)

# Set sample_width to 1
print(f"Old sample width: {wav_file.sample_width}")
wav_file_sw_1 = wav_file.set_sample_width(1)
print(f"New sample width: {wav_file_sw_1.sample_width}")


print('*********************************************************')
print('** 03.06 Manipulating audio files with PyDub')
print('*********************************************************')
# Create an AudioSegment instance
file, lang = 'wav_file.wav', 'en-US'
# Using the own created function
pprint(get_transcription(file, lang, show_all=True))


# Minus 60 dB
print('\n**Turning it down by 60 dB')
quiet_wav_file = wav_file - 60
# Try to recognize quiet audio
file, lang = 'quiet_wav_file.wav', 'en-US'
_ = quiet_wav_file.export(file, format='wav')
pprint(get_transcription(file, lang, show_all=True))


# Increase the volume by 10 dB
print('\n**Increase the volume by 10 dB')
louder_wav_file = wav_file + 10
# Try to recognize quiet audio
file, lang = 'louder_wav_file.wav', 'en-US'
_ = louder_wav_file.export(file, format='wav')
pprint(get_transcription(file, lang, show_all=True))


print('\n**Normalizing audio files')
# Import uneven sound audio file
loud_quiet = AudioSegment.from_file("ex3_datacamp_loud_then_quiet.wav")
# Normalize the sound levels
normalized_loud_quiet = normalize(loud_quiet)


print('\n**No static')
# Import audio with static at start
static_at_start = AudioSegment.from_file("static-out-of-warranty.wav")
# Remove the static via slicing
no_static_at_start = static_at_start[3000:]


print('\n** Remixing your audio')
# Import two audio files
wav_file_1 = AudioSegment.from_file("speaker_1.wav")
wav_file_2 = AudioSegment.from_file("speaker_2.wav")
# Combine the two audio files
wav_file_3 = wav_file_1 + wav_file_2
# Combine two wav files and make the combination louder
louder_wav_file_3 = wav_file_1 + wav_file_2 + 10


print('\n** Splitting your audio')
# Import phone call audio
phone_call = AudioSegment.from_file("ex3_stereo_call.wav")
# Find number of channels
print(phone_call.channels)
# Split stereo to mono
phone_call_channels = phone_call.split_to_mono()
print(phone_call_channels)

# Find number of channels of first list item
print(phone_call_channels[0].channels)

# Try to recognize the audio
file, lang = 'ex3_stereo_call_c0.wav', 'en-US'
_ = phone_call_channels[0].export(file, format='wav')

pprint(get_transcription(file, lang))

# Find number of channels of first list item
print(phone_call_channels[1].channels)

# Try to recognize the audio
file, lang = 'ex3_stereo_call_c1.wav', 'en-US'
_ = phone_call_channels[1].export(file, format='wav')

pprint(get_transcription(file, lang))


print('*********************************************************')
print('** 03.07 Turning it down... then up')
print('*********************************************************')
# Import audio file
volume_adjusted = AudioSegment.from_file('volume_adjusted.wav')

# Lower the volume by 60 dB
quiet_volume_adjusted = volume_adjusted - 60

# Increase the volume by 15 dB
louder_volume_adjusted = volume_adjusted + 15


print('*********************************************************')
print('** 03.08 Normalizing an audio file with PyDub')
print('*********************************************************')
# Import target audio file
loud_then_quiet = AudioSegment.from_file('ex3_datacamp_loud_then_quiet.wav')

# Normalize target audio file
normalized_loud_then_quiet = normalize(loud_then_quiet)


print('*********************************************************')
print('** 03.09 Chopping and changing audio files')
print('*********************************************************')
# Import part 1 and part 2 audio files
part_1 = AudioSegment.from_wav('ex3_slicing_part_1.wav')
part_2 = AudioSegment.from_file('ex3_slicing_part_2.wav')

# Remove the first four seconds of part 1
part_1_removed = part_1[4000:]

# Add the remainder of part 1 and part 2 together
part_3 = part_1_removed + part_2


print('*********************************************************')
print('** 03.10 Splitting stereo audio to mono with PyDub')
print('*********************************************************')
# Import stereo audio file and check channels
stereo_phone_call = AudioSegment.from_file('ex3_stereo_call.wav')
print(f"Stereo number channels: {stereo_phone_call.channels}")

# Split stereo phone call and check channels
channels = stereo_phone_call.split_to_mono()
print(f"Split number channels: {channels[0].channels}, {channels[1].channels}")

# Save new channels separately
phone_call_channel_1 = channels[0]
phone_call_channel_2 = channels[1]


print('*********************************************************')
print('** 03.11 Converting and saving audio files with PyDub')
print('*********************************************************')
# Call our new function
make_wav(".", "audio_mp3_to_wav/")

print('*********************************************************')
print('** 03.12 Exporting and reformatting audio files')
print('*********************************************************')
# Import the .mp3 file
mp3_file = AudioSegment.from_file('mp3_file.mp3')

# Export the .mp3 file as wav
mp3_file.export(out_f='audio_mp3_to_wav/mp3_file.wav', format='wav')


print('*********************************************************')
print('** 03.13 Manipulating multiple audio files with PyDub')
print('*********************************************************')
folder = ['AUD-20180918-WA0000.mp3', 'mp3_file.mp3', 'CocaCola.mp3', 'AUD-20190504-WA0000.m4a']

# Loop through the files in the folder
for audio_file in folder:
    # Create the new .wav filename
    wav_filename = "audio_mp3_to_wav/" + os.path.splitext(os.path.basename(audio_file))[0] + ".wav"
        
    # Read audio_file and export it in wav format
    AudioSegment.from_file(audio_file).export(out_f=wav_filename, format='wav')
        
    print(f"Creating {wav_filename}...")

print('*********************************************************')
print('** 03.14 An audio processing workflow')
print('*********************************************************')
file_with_static = AudioSegment.from_file('ex3-static-help-with-account.mp3')

# Cut the first 3-seconds of static off
file_without_static = file_with_static[3000:]

# Increase the volume by 10dB
louder_file_without_static = file_without_static + 10

# Multiple files
folder = ['ex3-static-help-with-account.mp3']

for audio_file in folder:
    file_with_static = AudioSegment.from_file(audio_file)

    # Cut the 3-seconds of static off
    file_without_static = file_with_static[3000:]

    # Increase the volume by 10dB
    louder_file_without_static = file_without_static + 10
    
    # Create the .wav filename for export
    wav_filename = "audio_mp3_to_wav/" + os.path.splitext(os.path.basename(audio_file))[0] + ".wav"
    
    # Export the louder file without static as .wav
    louder_file_without_static.export(wav_filename, format='wav')
    print(f"Creating {wav_filename}...")

print('*********************************************************')
print('END')
print('*********************************************************')