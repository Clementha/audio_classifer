import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from dataset import UrbanSoundPrecomputedDataset
from models.cnn import EnvSoundCNN
from models.transformer import EnvSoundTransformer
from utils import load_config

def train():
    config = load_config('configs/config.yaml')
    wandb.init(project=config['project_name'], name=config['run_name'], config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Train on folds 1–9
    train_dataset = UrbanSoundPrecomputedDataset(
        folds=[1,2,3,4,5,6,7,8,9],
        sr=config['sr'],
        n_mels=config['n_mels']
    )
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=8, pin_memory=True)

    # ✅ Validate on fold 10
    val_dataset = UrbanSoundPrecomputedDataset(
        folds=[10],
        sr=config['sr'],
        n_mels=config['n_mels']
    )
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=8, pin_memory=True)

    if config['model_type'] == 'cnn':
        model = EnvSoundCNN(num_classes=config['num_classes'])
    else:
        model = EnvSoundTransformer(num_classes=config['num_classes'])

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            if config['model_type'] == 'transformer':
                inputs = inputs.squeeze(1)

            # ✅ Debug prints — only do this for first batch or use if i == 0:
            if i == 0 and epoch == 0:
                print("Inputs:", inputs.device)
                print("Labels:", labels.device)
                print("Model:", next(model.parameters()).device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total

        # ✅ Do simple validation loop
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if config['model_type'] == 'transformer':
                    inputs = inputs.squeeze(1)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total

        wandb.log({
            "loss": running_loss / len(train_loader),
            "train_accuracy": train_acc,
            "val_accuracy": val_acc
        })
        print(f"Epoch [{epoch+1}/{config['epochs']}], "
              f"Loss: {running_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    torch.save(model.state_dict(), f"./artifacts/{config['run_name']}_final.pth")

if __name__ == "__main__":
    train()