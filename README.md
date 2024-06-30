# Whisper AI Transcription Project

This project utilizes OpenAI's Whisper to transcribe audio files. It now supports batch processing, allowing users to transcribe multiple files at once and providing real-time updates on the transcription progress.

## Features
- User-friendly interface for selecting multiple audio files for transcription.
- Transcription of audio files using various Whisper models.
- Real-time progress updates during transcription, indicating the number of files processed and remaining.
- Calculation of audio duration and transcription time.
- Saving transcriptions to text files with detailed filenames.

### Requirements
- Python 3.x
- `whisper`
- `pydub`
- `tkinter` (for file dialog on macOS)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/whisper-ai-transcription.git
   cd whisper-ai-transcription
   ```

2. Install the required Python packages:
   ```bash
   pip install whisper pydub
   ```

3. Install `ffmpeg`:
   - **macOS**:
     ```bash
     brew install ffmpeg
     ```

### Usage
1. Run the script:
   ```bash
   python transcribe.py
   ```

2. Follow the on-screen instructions to select an audio file and choose a Whisper model.

### Batch Transcription

- The application now supports selecting multiple audio files for simultaneous transcription.
- Users will receive progress updates after each file is processed, which helps in tracking the transcription status especially when dealing with large batches of audio files.

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
