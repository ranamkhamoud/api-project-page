# run this file to remove huggingfacemodels cache, it removes all saved huggingface models from your system
import shutil
from pathlib import Path

# Expand ~ to your home directory
hf_cache = Path.home() / ".cache" / "huggingface"

# Remove the directory if it exists
if hf_cache.exists() and hf_cache.is_dir():
    shutil.rmtree(hf_cache)
    print(f"Removed Hugging Face cache at: {hf_cache}")
else:
    print(f"No Hugging Face cache found at: {hf_cache}")