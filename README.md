# UrbanSound8K Audio Classifier (CNN + Transformer)

This starter repo uses PyTorch to classify UrbanSound8K sounds.
- Uses log-mel spectrograms as images.
- Choose CNN or Transformer backend.
- Logs to Weights & Biases (W&B).

## How to use

1. ```
   pip install soundata
   ```

2. Install:
   ```
   pip install -r requirements.txt
   ```

3. Edit `configs/config.yaml` to pick model.

4. Run:
   ```
   export WANDB_API_KEY=YOUR_KEY
   python train.py
   ```

Enjoy!
