import argparse
import os
import random
import shutil
from pathlib import Path

def sample_files(
    source_dir, 
    dest_dir=None, 
    seed=None, 
    sample_size=10, 
    recursive=False,
    quiet=False
):
    source_dir = Path(source_dir)
    if not os.path.exists(source_dir) or os.path.isdir(source_dir) is False:
        print(f"Source directory {source_dir} is not a directory or does not exist.")
        return
    
    if dest_dir and (not os.path.exists(dest_dir) or os.path.isdir(dest_dir) is False):
        print(f"Destination directory {dest_dir} is not a directory or does not exist.")
        return
    
    if recursive:
        all_files = [f for f in source_dir.rglob('*') if f.is_file()]
    else:
        all_files = [f for f in source_dir.glob('*') if f.is_file()]
    
    if seed is not None:
        random.seed(seed)
    
    sampled_files = random.sample(all_files, min(sample_size, len(all_files)))
    
    for file_path in sampled_files:
        if not quiet:
            print(file_path)
        if dest_dir:
            dest_path = Path(dest_dir) / file_path.name
            shutil.copy(file_path, dest_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Randomly sample files from a directory."
    )
    parser.add_argument("source_dir", help="Path to the source directory containing files")
    parser.add_argument("--dest_dir", type=str, help="Files will be copied to this directory if specified")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    parser.add_argument("--sample_size", type=int, default=10, help="Number of files to sample (default: 10)")
    parser.add_argument("--recursive", action='store_true', help="Recursively search subdirectories")
    parser.add_argument("--quiet", action='store_true', help="Suppress output of sampled file paths")
    args = parser.parse_args()
    
    sample_files(
        args.source_dir,
        dest_dir=args.dest_dir,
        seed=args.seed,
        sample_size=args.sample_size,
        recursive=args.recursive,
        quiet=args.quiet
    )
    