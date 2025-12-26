import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataloader.radioml_dataset import RadioMLDataset
from src.model.cnn_model import AMC_CNN

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    dataset = RadioMLDataset()
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    model = AMC_CNN(dataset.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    print(" Training started")

    for epoch in range(5):
        loss_sum = 0
        correct = total = 0

        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch+1} | Loss: {loss_sum:.2f} | Acc: {acc:.2f}%")

    print(" Training complete")

if __name__ == "__main__":
    main()
