import whisper
import time
import os
from pydub import AudioSegment
from datetime import timedelta
import tkinter as tk
from tkinter import filedialog


def select_files():
    print("Please choose one or several files to transcribe")
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_paths = filedialog.askopenfilenames(title="Select audio files",
                                             filetypes=[("Audio Files", "*.mp3 *.wav *.m4a *.mp4")])
    if file_paths:
        print(f"Files selected: {file_paths}")
    return file_paths


def choose_model():
    print("Please choose a model:")
    print("1) ENG Small, VRAM~2GB, Speed 6x, model name 'small.en'")
    print("2) ENG Medium, VRAM~5GB, Speed 2x, model name 'medium.en'")
    print("3) All Small (incl.RU), VRAM~2GB, Speed 6x, model name 'small'")
    print("4) All Medium (incl.RU), VRAM~5GB, Speed 2x, model name 'medium'")

    choice = input("Enter the number of the model you want to use: ")

    model_mapping = {
        "1": "small.en",
        "2": "medium.en",
        "3": "small",
        "4": "medium"
    }

    model_name = model_mapping.get(choice, "small")  # Default to 'small' if invalid choice
    print(f"Model '{model_name}' chosen. Transcription process has started.")
    return model_name


def transcribe_audio(file_path, model, model_name, idx, total_files):
    print(f"Starting transcription of {os.path.basename(file_path)}.")
    # Calculate the duration of the audio file
    audio = AudioSegment.from_file(file_path)
    audio_duration_ms = len(audio)
    audio_duration = round(audio_duration_ms / 1000)  # Duration in seconds, rounded
    audio_duration_formatted = str(timedelta(seconds=audio_duration))

    # Start the timer
    start_time = time.time()

    # Load and transcribe the audio
    result = model.transcribe(file_path)

    # Stop the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = round(end_time - start_time)
    elapsed_time_formatted = str(timedelta(seconds=elapsed_time))

    # Get the transcription text
    transcription_text = result['text']

    # Count the number of symbols (characters)
    symbol_count = len(transcription_text)

    # Print the transcription, audio duration, symbol count, model name, and elapsed time
    print(f"Transcription: {transcription_text}")
    print(f"Audio duration: {audio_duration_formatted} (hh:mm:ss)")
    print(f"Number of symbols: {symbol_count}")
    print(f"Time taken: {elapsed_time_formatted} (hh:mm:ss)")

    # Create a descriptive file name
    base_path, _ = os.path.splitext(file_path)
    filename = f"{base_path}_duration-{audio_duration_formatted}_model-{model_name}_symbols-{symbol_count}_time-{elapsed_time_formatted}.txt"

    # Save the transcription to a .txt file
    with open(filename, "w") as txt_file:
        txt_file.write(transcription_text)

    print(f"Transcription saved to: {filename}")
    print(
        f"Transcription process in progress. {idx + 1} files done, {total_files - idx - 1} still left. Total files {total_files}.")


if __name__ == "__main__":
    # Select the audio files
    audio_file_paths = select_files()
    if not audio_file_paths:
        print("No files selected. Exiting.")
    else:
        # Choose the model
        model_name = choose_model()
        model = whisper.load_model(model_name)

        total_files = len(audio_file_paths)
        for idx, file_path in enumerate(audio_file_paths):
            transcribe_audio(file_path, model, model_name, idx, total_files)