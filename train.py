import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from dataset import UrbanSoundDataset
from models.cnn import EnvSoundCNN
from models.transformer import EnvSoundTransformer
from utils import load_config

def train():
    config = load_config('configs/config.yaml')
    wandb.init(project=config['project_name'], name=config['run_name'], config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = UrbanSoundDataset(
            fold=1,  # Choose a fold (1–10)
            sr=config['sr'],
            n_mels=config['n_mels']
        )
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

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

        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            if config['model_type'] == 'transformer':
                inputs = inputs.squeeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        wandb.log({"loss": running_loss / len(dataloader), "accuracy": acc})
        print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {running_loss:.4f}, Acc: {acc:.2f}%")

    torch.save(model.state_dict(), f"./artifacts/{config['run_name']}_final.pth")

if __name__ == "__main__":
    train()
