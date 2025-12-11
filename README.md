# Authenticity Detection System


A multi-modal pipeline for detecting AI-assisted responses in audio recordings. This system combines acoustic analysis, speech recognition, and AI text detection to assess whether speech is spontaneous or read/scripted.


---


## Live Demo & Documentation


- **Live Demo:** [Try on Hugging Face Spaces](https://huggingface.co/spaces/ranamhamoud/Authenticity)

- **Project Website:** [Documentation](https://api-project-2025.netlify.app/)


---


## Project Structure


```

├── app.py # Gradio web interface for the detection system
├── pipeline.py # Main orchestration pipeline combining all modules
├── audio_classifier.py # CNN-based audio classification (read vs spontaneous)
├── speech_recognizer.py # Whisper-based speech-to-text and linguistic analysis
├── text_analyzer.py # AI text detection wrapper
├── plagiarism_detection.py # Desklib AI text detector implementation
│
├── model_checkpoints/ # Pre-trained CNN model weights
│ ├── spectrogram_cnn_3s_window.pth
│ ├── spectrogram_cnn_4s_window.pth
│
├── audio_classification_training/ # Training scripts for the CNN model
│ ├── model.py # ResNet-based training script
│ └── spectrogram.py # Audio-to-spectrogram conversion utility
│
├── ai_plagiarism_testing/ # AI text detection experiments and results
│ ├── ai_plagiarism_experiment/ # Experiment results (CSV files)
│ ├── ai_plagiarism_tuning_plots/ # Threshold tuning plots
│
├── examples/ # Sample audio files for demo
│ ├── read1.ogg # Example of read speech
│ └── spontaneous1.ogg # Example of spontaneous speech
│
├── index.html # Project documentation webpage
├── requirements.txt # Python dependencies
│
├── iemocap.zip # Audio dataset (sample from IEMOCAP)
└── samples_read_vs_spon.zip # Read vs Spontaneous speech samples

```


---


## Data Files


The project includes two ZIP archives containing audio data:


- [**`iemocap.zip`**](https://sail.usc.edu/iemocap/) — A sample subset of the larger IEMOCAP (Interactive Emotional Dyadic Motion Capture) dataset. This is used for training and testing the audio classification models.


- **`samples_read_vs_spon.zip`** — Audio samples labeled as either "read" or "spontaneous" speech, used for model validation.


> **Note:** These are sample datasets. The full IEMOCAP dataset is much larger and must be obtained separately from the official source.


---


## Installation


1. **Clone the repository and navigate to the project directory:**

```bash

cd /path/to/api

```


2. **Create a virtual environment (recommended):**

```bash

python -m venv venv

source venv/bin/activate # On Windows: venv\Scripts\activate

```


3. **Install dependencies:**

```bash

pip install -r requirements.txt

```


4. **Download the AI detection model (required for text analysis):**

The system uses the [Desklib AI Text Detector](https://huggingface.co/desklib/ai-text-detector-v1.01). It will be downloaded automatically on first run.


---


## Running the System


### Web Interface (Recommended)


Launch the Gradio web application:


```bash

python app.py

```


This starts a local server at `http://localhost:7860`. You can:

- Upload audio files (WAV, MP3, M4A, FLAC, OGG)

- Record audio directly via microphone

- View comprehensive analysis results across multiple tabs


Each module can be run independently:


```bash

# Audio classifier (CNN-based read/spontaneous detection)

python audio_classifier.py


# Speech recognizer (Whisper transcription + linguistic analysis)

python speech_recognizer.py


# Text analyzer (AI text detection)

python text_analyzer.py


# Run AI plagiarism experiments

python plagiarism_detection.py

```


---


## Training the Audio Classification Model


To train the CNN model on your own spectrogram data:


1. **Generate spectrograms from audio files:**

```bash

python audio_classification_training/spectrogram.py <source_audio_dir> <dest_spectrogram_dir> --start 0 --end 3

```


2. **Train the model:**

Update the `DATA_DIR` path in `audio_classification_training/model.py`, then run:

```bash

python audio_classification_training/model.py

```


The trained model will be saved as a `.pth` file in the current directory.

---


## Requirements


- Python 3.8+

- PyTorch 2.0+

- CUDA-compatible GPU (optional, but recommended for faster processing)

- ~4GB RAM minimum

- FFmpeg (for audio processing)


---


## Project Members


- Ahmad El Hage

- Alexander Scheerder

- Leon Monster

- Ranam Hamoud


---


## License


This project is for academic purposes as part of Leiden University's Audio Processing & Indexing course.
