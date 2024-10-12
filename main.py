import torch
from model.vit import VisionTransformer
from utils.data_loader import load_cifar10
import torch.optim as optim
import torch.nn as nn

def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model = model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

def evaluate_model(model, test_loader, device='cuda'):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    batch_size = 64
    num_epochs = 10
    train_loader, test_loader = load_cifar10(batch_size=batch_size)
    vit = VisionTransformer(img_size=224, patch_size=16, num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vit.parameters(), lr=3e-4)
    train_model(vit, train_loader, criterion, optimizer, num_epochs=num_epochs)
    evaluate_model(vit, test_loader)
