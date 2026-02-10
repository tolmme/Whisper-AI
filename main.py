import json
import importlib
import importlib.util
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import timedelta

import tkinter as tk
import watchdog.events
from pydub import AudioSegment
from tkinter import filedialog
from watchdog.observers.polling import PollingObserver

try:
    import torch
except ImportError:
    torch = None


AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".mp4")


def select_folder():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    folder_path = filedialog.askdirectory(title="Select a folder for automatic transcription")
    return folder_path


def sanitize_filename_part(value):
    return value.replace("/", "_").replace(":", "_").replace(" ", "_")


def format_hhmmss(seconds_value):
    return str(timedelta(seconds=round(seconds_value)))


def format_timestamp(seconds_value):
    total_ms = max(0, int(round(float(seconds_value) * 1000)))
    hours, rem = divmod(total_ms, 3600000)
    minutes, rem = divmod(rem, 60000)
    seconds, milliseconds = divmod(rem, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


def to_plain_dict(obj):
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return json.loads(str(obj))


def save_transcription_outputs(file_path, model_name, result, audio_duration, elapsed_time, source_label, include_timestamps):
    transcription_text = (result.get("text") or "").strip()
    symbol_count = len(transcription_text)
    audio_duration_formatted = format_hhmmss(audio_duration)
    elapsed_time_formatted = format_hhmmss(elapsed_time)

    model_tag = sanitize_filename_part(model_name)
    base_path, _ = os.path.splitext(file_path)

    summary_path = (
        f"{base_path}_duration-{audio_duration_formatted}_model-{model_tag}"
        f"_symbols-{symbol_count}_time-{elapsed_time_formatted}.txt"
    )
    with open(summary_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(transcription_text)

    words_output_exists = False
    timestamps_path = None
    words_path = None
    json_path = None
    if include_timestamps:
        timestamps_path = f"{base_path}_timestamps_model-{model_tag}.txt"
        with open(timestamps_path, "w", encoding="utf-8") as ts_file:
            for segment in result.get("segments", []):
                start = format_timestamp(segment.get("start", 0))
                end = format_timestamp(segment.get("end", 0))
                speaker = segment.get("speaker")
                text = (segment.get("text") or "").strip()
                if speaker:
                    ts_file.write(f"[{start} --> {end}] {speaker}: {text}\n")
                else:
                    ts_file.write(f"[{start} --> {end}] {text}\n")

        words_path = f"{base_path}_words_model-{model_tag}.txt"
        with open(words_path, "w", encoding="utf-8") as words_file:
            for segment in result.get("segments", []):
                for word_info in segment.get("words", []):
                    words_output_exists = True
                    start = format_timestamp(word_info.get("start", 0))
                    end = format_timestamp(word_info.get("end", 0))
                    word = (word_info.get("word") or "").strip()
                    speaker = word_info.get("speaker") or segment.get("speaker")
                    if speaker:
                        words_file.write(f"[{start} --> {end}] {speaker}: {word}\n")
                    else:
                        words_file.write(f"[{start} --> {end}] {word}\n")

        if not words_output_exists:
            os.remove(words_path)
            words_path = None

        json_path = f"{base_path}_full_result_model-{model_tag}.json"
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(result, json_file, ensure_ascii=False, indent=2)

    print(f"Source: {source_label}")
    print(f"Transcription: {transcription_text}")
    print(f"Audio duration: {audio_duration_formatted} (hh:mm:ss)")
    print(f"Number of symbols: {symbol_count}")
    print(f"Time taken: {elapsed_time_formatted} (hh:mm:ss)")
    print(f"Transcription saved to: {summary_path}")
    if include_timestamps:
        print(f"Timestamps saved to: {timestamps_path}")
        if words_output_exists:
            print(f"Word-level timestamps saved to: {words_path}")
        print(f"Full JSON saved to: {json_path}")
    else:
        print("Timestamps are disabled for this run.")


def transcribe_audio_local_whisper(file_path, model, model_name, include_timestamps, backend_label):
    print(f"Starting local transcription of {os.path.basename(file_path)}.")
    audio = AudioSegment.from_file(file_path)
    audio_duration = len(audio) / 1000
    start_time = time.time()
    result = model.transcribe(
        file_path,
        word_timestamps=include_timestamps,
        verbose=False,
    )
    elapsed_time = time.time() - start_time
    save_transcription_outputs(
        file_path,
        model_name,
        result,
        audio_duration,
        elapsed_time,
        backend_label,
        include_timestamps,
    )
    print(f"Transcription process completed for {os.path.basename(file_path)}.\n")


def transcribe_audio_local_mlx(file_path, mlx_transcribe, mlx_repo, model_name, include_timestamps):
    print(f"Starting local transcription of {os.path.basename(file_path)}.")
    audio = AudioSegment.from_file(file_path)
    audio_duration = len(audio) / 1000
    start_time = time.time()

    result = None
    errors = []
    call_variants = [
        lambda: mlx_transcribe(
            file_path,
            path_or_hf_repo=mlx_repo,
            word_timestamps=include_timestamps,
            verbose=False,
        ),
        lambda: mlx_transcribe(
            file_path,
            model=mlx_repo,
            word_timestamps=include_timestamps,
            verbose=False,
        ),
        lambda: mlx_transcribe(
            file_path,
            mlx_repo,
            word_timestamps=include_timestamps,
            verbose=False,
        ),
        lambda: mlx_transcribe(file_path, mlx_repo, verbose=False),
    ]
    for call_fn in call_variants:
        try:
            result = call_fn()
            break
        except TypeError as exc:
            errors.append(str(exc))

    if result is None:
        raise RuntimeError(
            "Failed to call mlx-whisper transcribe(). "
            f"Tried multiple signatures. Last errors: {errors[-2:]}"
        )

    result = to_plain_dict(result)
    elapsed_time = time.time() - start_time
    save_transcription_outputs(
        file_path,
        model_name,
        result,
        audio_duration,
        elapsed_time,
        "local-mlx-whisper(metal)",
        include_timestamps,
    )
    print(f"Transcription process completed for {os.path.basename(file_path)}.\n")


def transcribe_audio_openai(file_path, model_name, include_timestamps):
    print(f"Starting OpenAI transcription of {os.path.basename(file_path)} with model '{model_name}'.")
    print("OpenAI API mode: detailed per-segment progress is not available from the server.")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Export your API key before using OpenAI mode.")

    try:
        from openai import OpenAI
    except ImportError as import_error:
        raise RuntimeError("OpenAI SDK is not installed. Run: pip install openai") from import_error

    audio = AudioSegment.from_file(file_path)
    audio_duration = len(audio) / 1000
    start_time = time.time()

    client = OpenAI()
    with open(file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=model_name,
            file=audio_file,
            response_format="json",
        )

    result = to_plain_dict(response)
    elapsed_time = time.time() - start_time
    save_transcription_outputs(
        file_path,
        model_name,
        result,
        audio_duration,
        elapsed_time,
        "openai-api",
        include_timestamps,
    )
    print(f"Transcription process completed for {os.path.basename(file_path)}.\n")


class TranscriptionStatus:
    def __init__(self):
        self.total_files_added = 0
        self.finished_files = 0
        self.lock = threading.Lock()

    def add_file(self):
        with self.lock:
            self.total_files_added += 1
            self.print_status()

    def finish_file(self):
        with self.lock:
            self.finished_files += 1
            self.print_status()

    def print_status(self):
        remaining = self.total_files_added - self.finished_files
        percentage = (self.finished_files / self.total_files_added) * 100 if self.total_files_added else 0
        print(f"Status Update: {self.finished_files} of {self.total_files_added} files have been transcribed ({percentage:.0f}%).")
        print(f"{remaining} file(s) still left to transcribe.\n")

    def all_transcribed(self):
        with self.lock:
            if self.finished_files == self.total_files_added and self.total_files_added != 0:
                print("All files have been transcribed.\n")


def transcribe_from_queue(transcribe_function, task_queue, status):
    while True:
        file_path = task_queue.get()  # Get the next file from the queue
        if file_path is None:  # Exit signal
            break

        print(f"Transcribing file {status.finished_files + 1} of {status.total_files_added}...")
        try:
            transcribe_function(file_path)
        except Exception as exc:
            print(f"Error while transcribing '{file_path}': {exc}\n")
        status.finish_file()  # Update status after finishing transcription
        task_queue.task_done()  # Mark the task as done

        # Check if all files have been transcribed
        if status.finished_files == status.total_files_added:
            status.all_transcribed()


def transcribe_single_file(transcribe_function):
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_paths = filedialog.askopenfilenames(
        title="Select audio files",
        filetypes=[
            ("Audio Files", "*.mp3"),
            ("Audio Files", "*.wav"),
            ("Audio Files", "*.m4a"),
            ("Audio Files", "*.mp4"),
            ("All Files", "*.*")  # Optional: to show all files
        ]
    )
    
    if file_paths:
        status = TranscriptionStatus()
        task_queue = queue.Queue()
        
        # Enqueue all selected files
        for file_path in file_paths:
            print(f"Adding audio file to queue: {file_path}")
            task_queue.put(file_path)
            status.add_file()

        # Start a thread to process the queue
        threading.Thread(target=transcribe_from_queue, args=(transcribe_function, task_queue, status), daemon=True).start()
        
        # Wait until all tasks are done
        task_queue.join()
    else:
        print("No files selected.")


class FileEventHandler(watchdog.events.FileSystemEventHandler):
    def __init__(self, transcribe_function, task_queue, status):
        self.transcribe_function = transcribe_function
        self.task_queue = task_queue
        self.status = status

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(AUDIO_EXTENSIONS):
            print(f"Detected new file: {event.src_path}")
            print(f"Adding audio file to queue: {event.src_path}")
            self.task_queue.put(event.src_path)  # Add the file to the queue
            self.status.add_file()
        else:
            print("Ignored non-audio file or directory.\n")


def choose_local_model():
    print("Choose a local Whisper model:")
    print("1) ENG Small (English-only), model 'small.en', RAM ~2 GB")
    print("2) ENG Medium (English-only), model 'medium.en', RAM ~5 GB")
    print("3) Multilingual Small (incl. RU), model 'small', RAM ~2 GB")
    print("4) Multilingual Medium (incl. RU), model 'medium' [recommended], RAM ~5 GB")
    print("5) Multilingual Large-v3 (best quality, slower), model 'large-v3', RAM ~9-10 GB")
    print("RAM values are approximate and depend on backend/device (MLX/CUDA/CPU).")
    print("For Russian language, choose 3/4/5. Do NOT use '.en' models.")
    
    model_choice = input("Enter the number of your choice: ")
    model_mapping = {
        '1': 'small.en',
        '2': 'medium.en',
        '3': 'small',
        '4': 'medium',
        '5': 'large-v3',
    }
    
    return model_mapping.get(model_choice, 'medium')  # Default to 'medium' if invalid choice


def choose_source():
    print("Choose transcription source:")
    print("1) Local (offline, auto backend: macOS->mlx-whisper, CUDA->Whisper GPU)")
    print("2) OpenAI API (supports diarization model for speaker split)")
    source_choice = input("Enter the number of your choice: ")
    return source_choice if source_choice in {"1", "2"} else "1"


def choose_openai_model():
    print("Choose an OpenAI transcription model:")
    print("1) gpt-4o-transcribe-diarize (speaker labels + timestamps)")
    print("2) gpt-4o-transcribe (no speaker labels)")
    print("3) gpt-4o-mini-transcribe (faster/cheaper)")
    model_choice = input("Enter the number of your choice: ")
    model_mapping = {
        "1": "gpt-4o-transcribe-diarize",
        "2": "gpt-4o-transcribe",
        "3": "gpt-4o-mini-transcribe",
    }
    return model_mapping.get(model_choice, "gpt-4o-transcribe-diarize")


def choose_timestamps_setting():
    timestamps_choice = input("Include timestamps in output files? (y/n): ").strip().lower()
    return timestamps_choice in {"y", "yes", "1", "true"}


def has_module(module_name):
    importlib.invalidate_caches()
    return importlib.util.find_spec(module_name) is not None


def ask_yes_no(prompt_text, default_yes=True):
    raw = input(prompt_text).strip().lower()
    if not raw:
        return default_yes
    return raw in {"y", "yes", "1", "true"}


def install_python_package(package_name):
    print(f"Installing '{package_name}' with {sys.executable} -m pip ...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-U", package_name],
            check=True,
        )
        return True
    except Exception as exc:
        print(f"Failed to install '{package_name}': {exc}")
        return False


def ensure_mlx_whisper_for_macos():
    if sys.platform != "darwin":
        return False
    if has_module("mlx_whisper"):
        return True

    print("macOS detected: mlx-whisper is not installed.")
    should_install = ask_yes_no(
        "Install mlx-whisper now for faster local transcription on Mac? (Y/n): ",
        default_yes=True,
    )
    if not should_install:
        print("mlx-whisper installation skipped. Falling back to openai-whisper.")
        return False

    if not install_python_package("mlx-whisper"):
        print("Could not install mlx-whisper. Falling back to openai-whisper.")
        return False

    if has_module("mlx_whisper"):
        print("mlx-whisper installed successfully.")
        return True

    print("mlx-whisper install finished but module is still not available. Falling back to openai-whisper.")
    return False


def choose_best_device_for_local_whisper():
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def configure_cpu_threads():
    if torch is None:
        return
    env_value = os.getenv("WHISPER_CPU_THREADS")
    if env_value:
        try:
            threads = int(env_value)
            if threads > 0:
                torch.set_num_threads(threads)
                print(f"CPU threads configured from WHISPER_CPU_THREADS={threads}")
                return
        except ValueError:
            pass
    threads = os.cpu_count() or 1
    torch.set_num_threads(threads)
    print(f"CPU threads configured automatically: {threads}")


def configure_cuda_runtime():
    if torch is None or not torch.cuda.is_available():
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def resolve_mlx_repo(model_name):
    if os.getenv("MLX_WHISPER_REPO"):
        return os.getenv("MLX_WHISPER_REPO")

    model_mapping = {
        "tiny.en": "mlx-community/whisper-tiny.en-mlx",
        "tiny": "mlx-community/whisper-tiny-mlx",
        "base.en": "mlx-community/whisper-base.en-mlx",
        "base": "mlx-community/whisper-base-mlx",
        "small.en": "mlx-community/whisper-small.en-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "medium.en": "mlx-community/whisper-medium.en-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "large-v2": "mlx-community/whisper-large-v2-mlx",
        "large-v3": "mlx-community/whisper-large-v3-mlx",
        "large": "mlx-community/whisper-large-v3-mlx",
    }
    return model_mapping.get(model_name, f"mlx-community/whisper-{model_name}-mlx")


def create_local_transcribe_function(model_name, include_timestamps):
    if sys.platform == "darwin":
        if ensure_mlx_whisper_for_macos():
            from mlx_whisper import transcribe as mlx_transcribe

            mlx_repo = resolve_mlx_repo(model_name)
            print(f"Local backend selected: mlx-whisper (Metal), repo: {mlx_repo}")
            return lambda file_path: transcribe_audio_local_mlx(
                file_path,
                mlx_transcribe,
                mlx_repo,
                model_name,
                include_timestamps,
            )

    if not has_module("whisper"):
        raise RuntimeError(
            "Local backend is unavailable. Install either 'mlx-whisper' (macOS) or 'openai-whisper'."
        )

    try:
        import whisper
    except ImportError as import_error:
        raise RuntimeError(
            "Failed to import 'whisper'. Install dependencies with: pip install openai-whisper torch"
        ) from import_error

    device = choose_best_device_for_local_whisper()
    print(f"Local backend selected: openai-whisper, device: {device}")
    if device == "cpu":
        configure_cpu_threads()
    if device == "cuda":
        configure_cuda_runtime()
    model = whisper.load_model(model_name, device=device)
    backend_label = f"local-openai-whisper({device})"
    return lambda file_path: transcribe_audio_local_whisper(
        file_path,
        model,
        model_name,
        include_timestamps,
        backend_label,
    )


if __name__ == "__main__":
    # Ask the user what they want to do
    choice = input("Do you want to (1) transcribe a single audio file or (2) set up auto transcription for a folder? (Enter 1 or 2): ")
    include_timestamps = choose_timestamps_setting()

    source_choice = choose_source()
    if source_choice == "1":
        model_name = choose_local_model()
        transcribe_function = create_local_transcribe_function(model_name, include_timestamps)
    else:
        model_name = choose_openai_model()
        transcribe_function = lambda file_path: transcribe_audio_openai(file_path, model_name, include_timestamps)

    if choice == '1':
        transcribe_single_file(transcribe_function)
    elif choice == '2':
        # Ask the user to select a folder to watch
        folder_to_watch = select_folder()
        print(f"Watching folder: {folder_to_watch}")  # Confirm the folder path
        if not folder_to_watch:
            print("No folder selected. Exiting.")
            exit()

        task_queue = queue.Queue()  # Create a queue for transcription tasks
        status = TranscriptionStatus()

        event_handler = FileEventHandler(transcribe_function, task_queue, status)
        observer = PollingObserver()
        observer.schedule(event_handler, folder_to_watch, recursive=False)
        observer.start()

        # Start a thread to process the queue
        threading.Thread(target=transcribe_from_queue, args=(transcribe_function, task_queue, status), daemon=True).start()

        print(f"Watching folder: {folder_to_watch} for new audio files...\n")
        
        try:
            while True:
                time.sleep(1)  # Keep the script running without printing file lists
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
    else:
        print("Invalid choice. Exiting.")
