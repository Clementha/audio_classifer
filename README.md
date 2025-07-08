# UrbanSound8K Audio Classifier (CNN + Transformer)

This repo uses PyTorch to classify UrbanSound8K sounds.
- Uses log-mel spectrograms as images.
- Choose CNN or Transformer backend.
- Logs to Weights & Biases (W&B).

## How to use

1. Install Urban Sound dataset
   ```
   pip install soundata
   ```

2. Install modules:
   ```
   pip install -r requirements.txt
   ```

3. Edit `configs/config.yaml` to pick model (transformer or CNN)

4. Precompute all log-mel spectrograms, which are CPU bound, to speed up training 
   ```
   python precompute.py
   ```

5. Train:
   ```
   python train.py
   ```

6. Predict:
   ```
   python predict.py
   ```

Enjoy!
