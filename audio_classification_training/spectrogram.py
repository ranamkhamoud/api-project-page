#!/usr/bin/env python3
import os
import argparse
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import uuid

AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}

def generate_spectrogram(
    audio_path, 
    dest_dir, 
    start_time, 
    end_time, 
    randomize
):
    y, sr = librosa.load(audio_path, sr=16000)
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # Extract segment
    y_segment = y[start_sample:end_sample]
    if len(y_segment) == 0 or len(y_segment) < (end_sample - start_sample):
        raise ValueError("Audio segment is too short or empty.")

    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(y=y_segment, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Plot
    plt.figure(figsize=(4, 4))
    plt.axis('off')
    librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save figure
    if randomize:
        filename = f"{uuid.uuid4().hex}.png"
    else:
        basename = os.path.basename(audio_path)
        filename = os.path.splitext(basename)[0] + ".png"
        
    dest_path = dest_dir / filename
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(dest_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()


def process_directory(source_dir, dest_dir, start_time, end_time, max_workers, randomize):
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    
    if not os.path.exists(source_dir) or os.path.isdir(source_dir) is False:
        print(f"Source directory {source_dir} is not a directory or does not exist.")
        return
    
    if not os.path.exists(dest_dir) or os.path.isdir(dest_dir) is False:
        print(f"Destination directory {dest_dir} is not a directory or does not exist.")
        return
    
    total_audio_files = sum(
        1 for f in Path(source_dir).rglob('*')
        if f.suffix.lower() in AUDIO_EXTS
    )
    progress = 0
    failed = 0

    tasks = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for root, _, files in os.walk(source_dir):
            for file in files:
                if Path(file).suffix.lower() in AUDIO_EXTS:
                    src_path = Path(root) / file
                    tasks.append(executor.submit(
                        generate_spectrogram, src_path, dest_dir, start_time, end_time, randomize
                    ))

        # Wait for all tasks to finish
        for future in as_completed(tasks):
            try:
                future.result()
            except Exception as e:
                failed += 1
            progress += 1
            print(f"{progress}/{total_audio_files} files processed.")
    
    print(f"Processing complete. {progress} files proccessed, {failed} files failed.")
            
def main():
    parser = argparse.ArgumentParser(
        description="Generate mel spectrograms for all audio files in a directory."
    )
    parser.add_argument("source_dir", help="Path to the source directory containing audio files")
    parser.add_argument("dest_dir", help="Path to the destination directory for spectrogram images")
    parser.add_argument("--start", type=float, default=0.0, help="Start time (seconds)")
    parser.add_argument("--end", type=float, default=10.0, help="End time (seconds)")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers (default: all cores)")
    parser.add_argument("--randomize_name", action="store_true", help="Give each spectrogram a random name.")
    args = parser.parse_args()

    process_directory(args.source_dir, args.dest_dir, args.start, args.end, args.workers, args.randomize_name)

if __name__ == "__main__":
    main()
    