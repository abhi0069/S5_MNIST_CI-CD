import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.network import SimpleCNN
from datetime import datetime
import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def train():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model (only once)
    model = SimpleCNN().to(device)
    
    # Display parameter count
    total_params = count_parameters(model)
    print(f'\nTotal parameters in model: {total_params:,}')
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    
    # Train for exactly 1 epoch
    print("\nStarting training for 1 epoch...")
    model.train()
    total_batches = len(train_loader)
    
    # Training loop
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{total_batches}, Loss: {loss.item():.4f}')
    
    print("\nEvaluating accuracies...")
    
    # Calculate training accuracy
    train_accuracy = evaluate(model, train_loader, device)
    print(f'Training Accuracy: {train_accuracy:.2f}%')
    
    # Calculate test accuracy
    test_accuracy = evaluate(model, test_loader, device)
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    
    # Save model with timestamp and both accuracies
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), 
              f'models/model_{timestamp}_train{train_accuracy:.2f}_test{test_accuracy:.2f}.pth')
    
if __name__ == "__main__":
    train() 