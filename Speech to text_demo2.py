import os
import librosa
import soundfile as sf
import speech_recognition as sr
import tempfile

def load_audio_file_if_exists(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)

    if os.path.exists(file_path):
        try:
            audio_data, sample_rate = librosa.load(file_path)
            return audio_data, sample_rate
        except Exception as e:
            print(f"An error occurred while loading the audio file: {e}")
    
    return None, None

# Specify the folder path and file name
folder_path = r'/Users/ivansanz/Downloads'
file_name = 'stengah_vocals.mp3'

# Call the load_audio_file_if_exists function
audio_data, sample_rate = load_audio_file_if_exists(folder_path, file_name)

if audio_data is not None:
    print(f"Audio loaded with shape: {audio_data.shape}, Sample rate: {sample_rate}")

    def transcribe_audio_to_text(audio_data, sample_rate):
        # Create a unique temporary WAV file name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav_file:
            temp_wav_filename = temp_wav_file.name

        # Save audio data to the temporary WAV file
        sf.write(temp_wav_filename, audio_data, sample_rate)

        recognizer = sr.Recognizer()

        try:
            with sr.AudioFile(temp_wav_filename) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
                return text
        except sr.UnknownValueError:
            return "Speech recognition could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results from Google Web Speech API; {e}"
        finally:
            # Clean up the temporary WAV file
            os.remove(temp_wav_filename)

    transcribed_text = transcribe_audio_to_text(audio_data, sample_rate)
    print("Transcribed text:")
    print(transcribed_text)

else:
    print(f"File '{file_name}' not found in '{folder_path}'.")
