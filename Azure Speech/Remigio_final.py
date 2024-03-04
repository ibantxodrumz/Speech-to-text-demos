import os
import azure.cognitiveservices.speech as speechsdk
from pydub import AudioSegment
import math
import tempfile
import time

def split_audio(audio, segment_duration=15):
    """
    Split an audio file into segments of a specified duration.

    Parameters:
    - audio (AudioSegment): The input audio segment.
    - segment_duration (int): The duration of each segment in seconds.

    Returns:
    - segments (list): List of audio segments.
    """
    total_duration = len(audio) / 1000  # Convert milliseconds to seconds
    num_segments = math.ceil(total_duration / segment_duration)

    segments = []
    for i in range(num_segments):
        start_time = i * segment_duration * 1000
        end_time = min((i + 1) * segment_duration * 1000, len(audio))
        segment = audio[start_time:end_time]
        segments.append(segment)

    return segments

def continuous_recognition(audio_segments, output_file_path, subscription_key, region):
    """
    Perform continuous speech recognition on audio segments and save the results to a text file.

    Parameters:
    - audio_segments (list): List of audio segments.
    - output_file_path (str): Path to save the recognized text.
    - subscription_key (str): Azure Speech API subscription key.
    - region (str): Azure region.

    Returns:
    - None
    """
    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)
    recognized_text = ""

    for i, segment in enumerate(audio_segments):
        try:
            temp_wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            segment.export(temp_wav_file.name, format="wav")
            temp_wav_file.close()

            audio_config = speechsdk.AudioConfig(filename=temp_wav_file.name)
            speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
            result = speech_recognizer.recognize_once()
            recognized_text += result.text + " "

            os.remove(temp_wav_file.name)

        except Exception as e:
            print(f"Error processing segment {i + 1}: {e}")

    with open(output_file_path, 'w') as output_file:
        output_file.write(recognized_text)

def read_prompts(prompts_file):
    """
    Read prompts from a file and return them as a set.

    Parameters:
    - prompts_file (str): Path to the prompts file.

    Returns:
    - prompts (set): Set of prompts.
    """
    with open(prompts_file, 'r') as file:
        prompts = set(file.read().split())
    return prompts

def flag_files(transcriptions_folder, prompts_file, flagged_folder):
    """
    Flag and move transcriptions that contain specific prompts.

    Parameters:
    - transcriptions_folder (str): Path to the folder containing transcriptions.
    - prompts_file (str): Path to the prompts file.
    - flagged_folder (str): Path to the folder for flagged transcriptions.

    Returns:
    - None
    """
    os.makedirs(flagged_folder, exist_ok=True)
    prompts = read_prompts(prompts_file)

    for file_name in os.listdir(transcriptions_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(transcriptions_folder, file_name)

            with open(file_path, 'r') as file:
                content = file.read()
                if any(prompt in content for prompt in prompts):
                    flagged_path = os.path.join(flagged_folder, file_name)
                    os.rename(file_path, flagged_path)
                    print(f"File '{file_name}' flagged and moved to 'flagged' folder.")
                else:
                    print(f"No flagged content found in '{file_name}'.")

if __name__ == "__main__":
    
    start_time = time.time()  # Record the start time
    
    # Configuration
    subscription_key = "80961d156c344c2eaa0b6b8c04b36373"
    region = "uksouth"
    folder_path = "/Users/ivansanz/Desktop/Azure Speech"
    output_folder = "/Users/ivansanz/Desktop/Azure Speech/transcriptions"
    prompts_file_path = "/Users/ivansanz/Desktop/Azure Speech/prompts.txt"
    flagged_folder_path = "/Users/ivansanz/Desktop/Azure Speech/flagged"

    # Process audio files
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            print(f"\nProcessing file: {file_path}")

            audio = AudioSegment.from_file(file_path)
            audio_segments = split_audio(audio, segment_duration=15)

            output_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")

            continuous_recognition(audio_segments, output_file_path, subscription_key, region)

            print(f"Recognized text saved to: {output_file_path}")

    # Flag files based on prompts
    flag_files(output_folder, prompts_file_path, flagged_folder_path)
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time
    print(f"\nTotal time elapsed: {elapsed_time} seconds")
    
    