# dataset.py

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import soundata

class UrbanSoundPrecomputedDataset(Dataset):
    def __init__(self, folds=[1], sr=22050, n_mels=128, transform=None):
        """
        Loads precomputed log-mel spectrograms instead of computing them on the fly.
        """
        self.dataset = soundata.initialize('urbansound8k')
        self.save_root = "precomputed_specs"

        # Filter clips by folds
        self.clips = []
        for clip_id in self.dataset.clip_ids:
            clip = self.dataset.clip(clip_id)
            if int(clip.fold) in folds:
                self.clips.append(clip)

        self.n_mels = n_mels
        self.transform = transform
        self.max_frames = 348  # Use your verified max

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        # Load precomputed log-mel
        spec = np.load(os.path.join(self.save_root, f"{clip.clip_id}.npy"))
        log_melspec = torch.tensor(spec).float()

        # Pad or trim to max_frames
        num_frames = log_melspec.shape[-1]
        if num_frames < self.max_frames:
            pad_amount = self.max_frames - num_frames
            log_melspec = torch.nn.functional.pad(log_melspec, (0, pad_amount))
        elif num_frames > self.max_frames:
            log_melspec = log_melspec[..., :self.max_frames]

        if len(log_melspec.shape) == 2:
            log_melspec = log_melspec.unsqueeze(0)

        label = clip.class_id
        return log_melspec, label