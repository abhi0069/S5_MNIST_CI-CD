import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from model.network import SimpleCNN
from torchvision import datasets, transforms

def test_model_parameters():
    model = SimpleCNN()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 25000, f"Model has {total_params} parameters, should be less than 25000"

def test_input_output_dimensions():
    model = SimpleCNN()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"

def test_model_accuracy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    
    # Load the latest model
    import glob
    model_files = glob.glob('models/*.pth')
    latest_model = max(model_files, key=os.path.getctime)
    model.load_state_dict(torch.load(latest_model))
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy > 80, f"Model accuracy is {accuracy}%, should be > 80%"

def test_output_probabilities():
    """Test if model outputs valid probabilities after softmax"""
    model = SimpleCNN()
    test_input = torch.randn(1, 1, 28, 28)
    output = torch.nn.functional.softmax(model(test_input), dim=1)
    
    # Check if probabilities sum to 1 (with small tolerance for floating point)
    assert abs(output.sum().item() - 1.0) < 1e-6, "Output probabilities don't sum to 1"
    # Check if all probabilities are between 0 and 1
    assert torch.all((output >= 0) & (output <= 1)), "Output contains invalid probabilities"

def test_model_gradient_flow():
    """Test if gradients flow through the model properly"""
    model = SimpleCNN()
    test_input = torch.randn(1, 1, 28, 28)
    test_target = torch.tensor([5])  # Random target class
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Forward pass
    output = model(test_input)
    loss = criterion(output, test_target)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check if gradients exist and are not zero for all parameters
    has_gradients = all(param.grad is not None and torch.any(param.grad != 0) 
                       for param in model.parameters())
    assert has_gradients, "Some parameters have no gradients or zero gradients"

def test_model_batch_processing():
    """Test if model can handle different batch sizes"""
    model = SimpleCNN()
    batch_sizes = [1, 4, 16, 32]
    
    for batch_size in batch_sizes:
        test_input = torch.randn(batch_size, 1, 28, 28)
        output = model(test_input)
        assert output.shape == (batch_size, 10), f"Failed for batch size {batch_size}"