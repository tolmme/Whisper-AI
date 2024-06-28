## Whisper AI Transcription Project

This project utilizes OpenAI's Whisper to transcribe audio files. It provides a simple interface for selecting audio files and choosing the desired Whisper model for transcription. The project calculates the audio duration and transcription time, and saves the transcriptions to text files with descriptive filenames.

### Features
- User-friendly file selection and model choice interface
- Transcription of audio files using OpenAI's Whisper models
- Calculation of audio duration and transcription time
- Saving transcriptions to text files with detailed filenames

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

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

You can include this description in your `README.md` file on GitHub to provide an overview of your project.
