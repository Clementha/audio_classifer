import torch
import librosa
import numpy as np
from models.cnn import EnvSoundCNN  # or EnvSoundTransformer

# --- Load model ---
model = EnvSoundCNN(num_classes=10)
model.load_state_dict(torch.load("./artifacts/cnn_baseline_final.pth", map_location="cpu"))
model.eval()

# --- Preprocess clip ---
#audio_path = "./samples/dogbark.flac"
audio_path = "./samples/dogbark.flac"
sr = 22050
n_mels = 128
max_frames = 348

y, sr = librosa.load(audio_path, sr=sr)
melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=512)
log_melspec = librosa.power_to_db(melspec, ref=np.max)
log_melspec = torch.tensor(log_melspec).float()

num_frames = log_melspec.shape[-1]
if num_frames < max_frames:
    pad_amount = max_frames - num_frames
    log_melspec = torch.nn.functional.pad(log_melspec, (0, pad_amount))
elif num_frames > max_frames:
    log_melspec = log_melspec[..., :max_frames]

log_melspec = log_melspec.unsqueeze(0).unsqueeze(0)

# --- Predict ---
with torch.no_grad():
    output = model(log_melspec)
    predicted_class = output.argmax(dim=1).item()

ID_TO_LABEL = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music"
}

print("Predicted ID:", predicted_class)
print("Predicted label:", ID_TO_LABEL[predicted_class])