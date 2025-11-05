#!/usr/bin/env python3
import os
import argparse
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}

def generate_spectrogram(audio_path, dest_path, start_time, end_time):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # Extract segment
        y_segment = y[start_sample:end_sample]
        if len(y_segment) == 0:
            print(f"File {audio_path} is too small.")
            return

        # Compute mel spectrogram
        S = librosa.feature.melspectrogram(y=y_segment, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Plot
        plt.figure(figsize=(8, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'{Path(audio_path).stem} ({start_time}s-{end_time}s)')
        plt.tight_layout()

        # Save image
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(dest_path, dpi=150)
        plt.close()
        print(f"Saved: {dest_path}")
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")


def process_directory(source_dir, dest_dir, start_time, end_time, max_workers):
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)

    tasks = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for root, _, files in os.walk(source_dir):
            for file in files:
                if Path(file).suffix.lower() in AUDIO_EXTS:
                    src_path = Path(root) / file
                    rel_path = src_path.relative_to(source_dir)
                    dest_path = dest_dir / rel_path.with_suffix('.png')
                    tasks.append(executor.submit(
                        generate_spectrogram, src_path, dest_path, start_time, end_time
                    ))

        # Wait for all tasks to finish
        for i, future in enumerate(as_completed(tasks), 1):
            try:
                future.result()
            except Exception as e:
                print(f"Task {i} failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate mel spectrograms for all audio files in a directory."
    )
    parser.add_argument("source_dir", help="Path to the source directory containing audio files")
    parser.add_argument("dest_dir", help="Path to the destination directory for spectrogram images")
    parser.add_argument("--start", type=float, default=10.0, help="Start time (seconds)")
    parser.add_argument("--end", type=float, default=20.0, help="End time (seconds)")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers (default: all cores)")
    args = parser.parse_args()

    process_directory(args.source_dir, args.dest_dir, args.start, args.end, args.workers)

if __name__ == "__main__":
    main()
    