#Dependencies:
#openai-whisper
#torch

import whisper
import sys
import os

def transcribe_audio(input_file, output_file=None):
    model = whisper.load_model("base")

    print("running whisper")

    result = model.transcribe(input_file)

    text = result["text"]

    # If no output file provided, auto-create one
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + ".txt"

    # Save text to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Transcription complete! Saved to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py your_audio_file.wav [optional_output.txt]")
        sys.exit(1)

    input_audio = sys.argv[1]
    output_text = sys.argv[2] if len(sys.argv) > 2 else None

    transcribe_audio(input_audio, output_text)
