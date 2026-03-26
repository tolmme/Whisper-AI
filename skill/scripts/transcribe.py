#!/usr/bin/env python3
"""Local audio transcription CLI using mlx-whisper (Metal) / openai-whisper fallback.

Requires the Whisper AI project's dependencies (mlx-whisper or openai-whisper, pydub).
Run from the Whisper AI project root, or set WHISPER_PROJECT_DIR env var.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Resolve project root: env var or parent of skill/ dir
PROJECT_DIR = os.environ.get(
    "WHISPER_PROJECT_DIR",
    str(Path(__file__).resolve().parent.parent.parent),
)
sys.path.insert(0, PROJECT_DIR)

from main import (
    AUDIO_EXTENSIONS,
    ensure_audio_binaries,
    create_local_transcribe_function,
    format_hhmmss,
)

STATE_DIR = os.environ.get(
    "TRANSCRIBE_STATE_DIR",
    os.path.expanduser("~/.transcribe"),
)
STATE_FILE = os.path.join(STATE_DIR, "queue.json")


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"queue": [], "completed": [], "current": None, "started_at": None}


def save_state(state):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def clear_state():
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)


def show_progress():
    state = load_state()
    total = len(state["completed"]) + len(state["queue"])
    if state["current"]:
        total += 1

    if total == 0:
        print("No transcription in progress.")
        return

    done = len(state["completed"])
    print(f"Progress: {done}/{total} files completed")

    if state["current"]:
        name = os.path.basename(state["current"])
        started = state.get("started_at", "")
        print(f"  Currently processing: {name}")
        if started:
            elapsed = datetime.now() - datetime.fromisoformat(started)
            print(f"  Elapsed: {str(elapsed).split('.')[0]}")

    if state["queue"]:
        print(f"  Queued: {len(state['queue'])} files")
        for f in state["queue"]:
            print(f"    - {os.path.basename(f)}")

    if state["completed"]:
        print(f"  Completed: {done} files")
        for item in state["completed"]:
            name = os.path.basename(item["file"])
            duration = item.get("elapsed", "?")
            print(f"    + {name} ({duration})")


def add_to_queue(files):
    state = load_state()
    added = 0
    for f in files:
        f = os.path.abspath(f)
        if f not in state["queue"] and f != state.get("current"):
            state["queue"].append(f)
            added += 1
            print(f"  Added: {os.path.basename(f)}")
    save_state(state)
    print(f"Added {added} file(s) to queue. Total queued: {len(state['queue'])}")


def _move_outputs(input_file, output_dir):
    """Move transcription output files from input file's directory to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    input_dir = os.path.dirname(input_file)
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    for f in os.listdir(input_dir):
        if f.startswith(base_name) and f != os.path.basename(input_file):
            ext = os.path.splitext(f)[1]
            if ext in (".txt", ".json"):
                src = os.path.join(input_dir, f)
                dst = os.path.join(output_dir, f)
                os.rename(src, dst)
                print(f"  Moved: {f} -> {output_dir}/")


def transcribe_files(files, model_name="medium", timestamps=False, output_dir=None):
    """Transcribe a list of files with queue tracking."""
    ensure_audio_binaries()

    valid_files = []
    for f in [os.path.abspath(p) for p in files]:
        if not os.path.exists(f):
            print(f"WARNING: File not found, skipping: {f}")
            continue
        ext = os.path.splitext(f)[1].lower()
        if ext not in AUDIO_EXTENSIONS:
            print(f"WARNING: Not an audio file ({ext}), skipping: {f}")
            continue
        valid_files.append(f)

    if not valid_files:
        print("No valid audio files to transcribe.")
        return

    print(f"\nTranscription settings:")
    print(f"  Model: {model_name}")
    print(f"  Timestamps: {'yes' if timestamps else 'no'}")
    print(f"  Output dir: {output_dir or 'same as input'}")
    print(f"  Files: {len(valid_files)}")
    print()

    print("Loading model...")
    transcribe_fn = create_local_transcribe_function(
        model_name,
        timestamps,
        interactive_prompt=False,
        auto_install_mlx=True,
    )

    state = {
        "queue": list(valid_files[1:]) if len(valid_files) > 1 else [],
        "completed": [],
        "current": None,
        "started_at": None,
    }
    save_state(state)

    for i, file_path in enumerate(valid_files):
        file_name = os.path.basename(file_path)
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(valid_files)}] {file_name}")
        print(f"{'='*60}")

        state["current"] = file_path
        state["started_at"] = datetime.now().isoformat()
        if file_path in state["queue"]:
            state["queue"].remove(file_path)
        save_state(state)

        start_time = time.time()

        try:
            transcribe_fn(file_path)
            elapsed = time.time() - start_time

            if output_dir:
                _move_outputs(file_path, output_dir)

            state["completed"].append({
                "file": file_path,
                "elapsed": format_hhmmss(elapsed),
                "finished_at": datetime.now().isoformat(),
            })
        except Exception as e:
            print(f"ERROR transcribing {file_name}: {e}")
            state["completed"].append({
                "file": file_path,
                "elapsed": "error",
                "error": str(e),
                "finished_at": datetime.now().isoformat(),
            })

        state["current"] = None
        state["started_at"] = None
        save_state(state)

    print(f"\n{'='*60}")
    print("TRANSCRIPTION COMPLETE")
    print(f"{'='*60}")
    successful = [c for c in state["completed"] if c.get("elapsed") != "error"]
    failed = [c for c in state["completed"] if c.get("elapsed") == "error"]
    print(f"  Successful: {len(successful)}/{len(valid_files)}")
    if failed:
        print(f"  Failed: {len(failed)}")
    for item in successful:
        print(f"    + {os.path.basename(item['file'])} ({item['elapsed']})")

    clear_state()


def main():
    parser = argparse.ArgumentParser(
        description="Local audio transcription (mlx-whisper / openai-whisper)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s recording.m4a                     # transcribe single file
  %(prog)s recording.m4a --timestamps        # with timestamps (for video editing)
  %(prog)s *.m4a -o ~/transcripts            # batch + custom output dir
  %(prog)s --progress                        # check queue progress
  %(prog)s --add file1.m4a file2.mp3         # add to running queue

Environment variables:
  WHISPER_PROJECT_DIR   Path to Whisper AI project (auto-detected)
  TRANSCRIBE_STATE_DIR  Queue state directory (default: ~/.transcribe)
""",
    )
    parser.add_argument("files", nargs="*", help="Audio files to transcribe")
    parser.add_argument(
        "--model", "-m",
        default="medium",
        choices=["tiny", "small", "medium", "large-v3"],
        help="Whisper model (default: medium, best for Russian)",
    )
    parser.add_argument(
        "--timestamps", "-t",
        action="store_true",
        help="Include segment + word-level timestamps (off by default)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory (default: same as input file)",
    )
    parser.add_argument(
        "--progress", "-p",
        action="store_true",
        help="Show progress of current transcription queue",
    )
    parser.add_argument(
        "--add",
        nargs="+",
        metavar="FILE",
        help="Add files to the running transcription queue",
    )

    args = parser.parse_args()

    if args.progress:
        show_progress()
        return

    if args.add:
        add_to_queue(args.add)
        return

    if not args.files:
        parser.print_help()
        return

    transcribe_files(
        files=args.files,
        model_name=args.model,
        timestamps=args.timestamps,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
