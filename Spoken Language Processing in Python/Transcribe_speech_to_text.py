## Functions to translate from audio to text\n
import speech_recognition as sr

def get_transcription(file, language, energy_threshold=300, duration=None, offset=None, show_all=None, noise=0):
    # Setup recognizer instance
    recognizer = sr.Recognizer()
    
    # Convert from AudioFile to AudioData
    with sr.AudioFile(file) as source:
        # Adjust for ambient noise and record
        if noise>0:
            recognizer.adjust_for_ambient_noise(source, duration=noise)
        
        # Record the audio
        audio_data = recognizer.record(source,
                                       duration=duration, #Listen from the begining to duration value.
                                       offset=offset) #used to skip over a specific seconds at the start.        
        
    # Set the energy threshold
    recognizer.energy_threshold = energy_threshold
        
    # Transcribe speech using Goole web API
    return recognizer.recognize_google(audio_data=audio_data, language=language, show_all=show_all)