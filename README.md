# Whisper AI Transcription (Local + OpenAI API)

Cross-platform transcription tool with:
- Local transcription (auto-optimized backend by platform)
- Optional timestamps (segment + word level)
- Optional speaker labels (OpenAI diarization model)
- Single-file and watched-folder processing

## What is implemented

- Local mode auto-selects backend:
  - macOS: `mlx-whisper` (Metal) if installed
  - other systems: `openai-whisper` on best device (`cuda` -> `mps` -> `cpu`)
- On macOS, if `mlx-whisper` is missing, script asks to install it automatically.
- OpenAI API mode supports:
  - `gpt-4o-transcribe-diarize` (speaker split)
  - `gpt-4o-transcribe`
  - `gpt-4o-mini-transcribe`
- Optional timestamps toggle (`y/n`) per run.
- Progress bar is shown in local mode (per file progress via Whisper/MLX internals).
- Russian supported via multilingual models (`small`, `medium`, `large-v3`).

## Requirements

- Python 3.11+
- `ffmpeg`
- macOS file dialog support (`tkinter`)

Python packages:
- Required: `pydub`, `watchdog`
- Local Whisper fallback/CUDA path: `openai-whisper`, `torch`
- macOS optimized path: `mlx-whisper` (optional, but recommended)
- OpenAI API mode: `openai` (optional)

## Installation

```bash
git clone https://github.com/tolmme/Whisper-AI.git
cd Whisper-AI
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install pydub watchdog openai-whisper torch
```

Install ffmpeg:
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- Windows: install ffmpeg and add it to `PATH`

Optional packages:

```bash
# macOS optimized local backend
pip install -U mlx-whisper

# OpenAI API mode
pip install -U openai
```

## Usage

```bash
source venv/bin/activate
python main.py
```

Then choose:
1. Single file mode or watched folder mode
2. Timestamps enabled/disabled
3. Source:
   - Local (offline, auto backend)
   - OpenAI API
4. Model

## Russian language models

For Russian, use multilingual models:
- `small`
- `medium` (recommended balance)
- `large-v3` (best quality, slower)

Do not use `.en` models for Russian (`small.en`, `medium.en` are English-only).

## Output files

Always saved:
- Main transcript text file (with duration/model/symbol/time in filename)

Saved only when timestamps are enabled:
- Segment timestamps: `*_timestamps_model-...txt`
- Word timestamps: `*_words_model-...txt` (if available)
- Full raw result JSON: `*_full_result_model-...json`

If OpenAI diarization model is used, speaker labels are included in timestamp files.

## Environment variables

- `OPENAI_API_KEY` - required for OpenAI API mode
- `WHISPER_CPU_THREADS` - optional CPU thread override for local CPU mode
- `MLX_WHISPER_REPO` - optional custom MLX model repo override

## Notes

- OpenAI API mode currently does not expose detailed per-segment server progress; local backends do show progress bars.
- Batch progress in queue mode is shown as `X of Y files`.

## License

MIT. See [LICENSE](LICENSE).
