import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from model import SelfPruningNet

def train_and_evaluate(lambda_val, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    model = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"\n--- Running Experiment: Lambda = {lambda_val} ---")
    for epoch in range(epochs):
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels) + lambda_val * model.get_sparsity_loss()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} complete.")

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total, model.get_sparsity_level(), model.get_all_gates()

if __name__ == "__main__":
    # Create results folder if it doesn't exist
    if not os.path.exists('results'): os.makedirs('results')
    
    lambdas = [1e-6, 1e-4, 1e-3]
    results = []
    best_gates = None

    for l in lambdas:
        acc, sp, gates = train_and_evaluate(l)
        results.append((l, acc, sp))
        if l == 1e-4: best_gates = gates

    # Save distribution plot
    plt.figure(figsize=(10, 6))
    plt.hist(best_gates, bins=50, color='skyblue', edgecolor='black')
    plt.title("Gate Value Distribution (Lambda = 1e-04)")
    plt.savefig('results/gates_dist.png')
    
    # Print summary table
    print("\n" + "="*45)
    print(f"{'Lambda':<10} | {'Accuracy (%)':<15} | {'Sparsity (%)':<12}")
    print("-" * 45)
    for l, acc, sp in results:
        print(f"{l:<10} | {acc:<15.2f} | {sp:<12.2f}")
    print("="*45)