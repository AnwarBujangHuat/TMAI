import os
import random
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

RAW_DIR = "raw_audio"
OUT_DIR = "dataset"
CATEGORIES = ["positive"]
TARGET_SR = 16000
TARGET_LEN = TARGET_SR  # 1 second

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_audio(path):
    data, sr = sf.read(path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    return data, sr

def resample_audio(data, orig_sr, target_sr):
    if orig_sr == target_sr:
        return data
    return resample_poly(data, target_sr, orig_sr)

def pad_or_trim(data, length):
    if len(data) > length:
        # Keep the loudest 1s
        window = length
        max_energy = 0
        start = 0
        for i in range(0, len(data) - window + 1, int(TARGET_SR * 0.1)):
            energy = np.sum(np.abs(data[i:i+window]))
            if energy > max_energy:
                max_energy = energy
                start = i
        return data[start:start+window]
    elif len(data) < length:
        pad = np.zeros(length)
        pad[:len(data)] = data
        return pad
    else:
        return data

def process_and_save(src_path, dst_path):
    data, sr = load_audio(src_path)
    data = resample_audio(data, sr, TARGET_SR)
    data = pad_or_trim(data, TARGET_LEN)
    sf.write(dst_path, data, TARGET_SR)


def collect_files(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.wav')]

def main():
    # Prepare output folders
    for cat in CATEGORIES:
        ensure_dir(os.path.join(OUT_DIR, cat))

    # 1. Process positive samples
    pos_files = collect_files(os.path.join(RAW_DIR, "positive"))
    pos_out = []
    for idx, f in enumerate(pos_files):
        out_path = os.path.join(OUT_DIR, "positive", f"positive{idx+1}.wav")
        process_and_save(f, out_path)
        pos_out.append(out_path)
    n_pos = len(pos_out)

 
  
    # 4. Print summary
    print(f"Number of positive samples: {n_pos}")

    print("✅ Balanced dataset ready in: dataset/")

if __name__ == "__main__":
    main()