import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import soundata

class UrbanSoundDataset(Dataset):
    def __init__(self, fold=1, sr=22050, n_mels=128, transform=None):
        self.dataset = soundata.initialize('urbansound8k')
        # self.dataset.download()  # Optional, skip if already done
        # self.dataset.validate()

        # Now use clip_ids + filter by fold
        self.clips = []
        for clip_id in self.dataset.clip_ids:
            clip = self.dataset.clip(clip_id)
            if int(clip.fold) == fold:
                self.clips.append(clip)

        self.sr = sr
        self.n_mels = n_mels
        self.transform = transform

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        audio, sr = clip.audio

        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
            sr = self.sr

        melspec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=self.n_mels)
        log_melspec = librosa.power_to_db(melspec, ref=np.max)
        log_melspec = torch.tensor(log_melspec).float()

        max_frames = 348 #Checked, it's the max
        num_frames = log_melspec.shape[-1]

        if num_frames < max_frames:
            pad_amount = max_frames - num_frames
            log_melspec = torch.nn.functional.pad(log_melspec, (0, pad_amount))
        elif num_frames > max_frames:
            log_melspec = log_melspec[..., :max_frames]

        if len(log_melspec.shape) == 2:
            log_melspec = log_melspec.unsqueeze(0)

        label = clip.class_id

        #print(clip.to_dict())
        return log_melspec, label