#speech_key = '80961d156c344c2eaa0b6b8c04b36373'   service_region = 'uksouth'
# /Users/ivansanz/Desktop/Azure Speech

import azure.cognitiveservices.speech as speechsdk
from pydub import AudioSegment
import math
import tempfile
import os
import time

def split_audio(file_path, segment_duration=15):
    audio = AudioSegment.from_file(file_path)

    # Calculate the total duration of the audio in seconds
    total_duration = len(audio) / 1000  # Convert milliseconds to seconds

    # Calculate the number of segments
    num_segments = math.ceil(total_duration / segment_duration)

    # Split the audio into segments
    segments = []
    for i in range(num_segments):
        start_time = i * segment_duration * 1000  # Convert seconds to milliseconds
        end_time = min((i + 1) * segment_duration * 1000, len(audio))
        segment = audio[start_time:end_time]
        segments.append(segment)

    return audio, segments

def continuous_recognition(audio, audio_segments):
    speech_config = speechsdk.SpeechConfig(subscription="80961d156c344c2eaa0b6b8c04b36373", region="uksouth")

    # Print the total duration of the entire audio file
    total_duration = len(audio) / 1000  # Convert milliseconds to seconds
    print(f"Total duration of the audio file: {total_duration:.2f} seconds\n")

    total_start_time = time.time()

    recognized_text = ""

    for i, segment in enumerate(audio_segments):
        # Save each segment to a temporary WAV file
        temp_wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        segment.export(temp_wav_file.name, format="wav")
        temp_wav_file.close()

        try:
            audio_config = speechsdk.AudioConfig(filename=temp_wav_file.name)

            # Create a speech recognizer
            speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

            # Perform recognition
            result = speech_recognizer.recognize_once()
            recognized_text += result.text + " "

        finally:
            # Clean up the temporary WAV file
            os.remove(temp_wav_file.name)

    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"\nTranslation process finished in: {total_time:.2f} seconds")

    # Save the recognized text to a text file in the 'transcriptions' subfolder
    output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'transcriptions')
    os.makedirs(output_folder, exist_ok=True)

    output_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}.txt")
    with open(output_file_path, 'w') as output_file:
        output_file.write(recognized_text)

    print(f"Recognized text saved to: {output_file_path}")

if __name__ == "__main__":
    file_path = "/Users/ivansanz/Desktop/Azure Speech/short.wav"
    audio, audio_segments = split_audio(file_path, segment_duration=15)
    continuous_recognition(audio, audio_segments)
