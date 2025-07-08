# precompute.py

import os
import librosa
import numpy as np
import soundata

# Where to save your precomputed specs
SAVE_ROOT = "./precomputed_specs"
os.makedirs(SAVE_ROOT, exist_ok=True)

# Load the UrbanSound8K dataset using soundata
ds = soundata.initialize('urbansound8k')

# One-time loop over all clips
for clip_id in ds.clip_ids:
    clip = ds.clip(clip_id)
    audio, sr = clip.audio

    # Resample if needed
    if sr != 22050:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)

    # Compute log-mel spectrogram
    melspec = librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128, n_fft=512)
    logmelspec = librosa.power_to_db(melspec, ref=np.max)

    # Save as .npy
    np.save(os.path.join(SAVE_ROOT, f"{clip_id}.npy"), logmelspec)

    print(f"Saved: {clip_id}.npy")

print("✅ All spectrograms precomputed!")