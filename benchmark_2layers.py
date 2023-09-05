import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
import torch.optim as optim
import numpy as np

# Set device (GPU if available, else CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Set a fixed seed for reproducibility
seed = 42
torch.manual_seed(seed)

# Hyperparameters
input_size = 3 * 32 * 32  # Input image size for CIFAR-10 (32x32 with 3 color channels)
hidden_size = 128
num_classes = 10

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define the neural network model
class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Training parameters
num_trials = 5
trial_epochs = 5

# Store trial metrics
trial_metrics = []

# Record the start time for the trials
start_time = time.time()

# Outer loop for trials
for trial in range(num_trials):
    print(f"Trial {trial+1}/{num_trials}")

    # Create the model
    net = TwoLayerNet(input_size, hidden_size, num_classes).to(device)
    print(net)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize optimizer
    optimizer = optim.Adam(net.parameters())
    
    # Store loss values for plotting
    train_losses = []
    test_losses = []  

    start_trial_time = time.time()  # Record start time of trial

    # Inner loop for trial epochs
    for epoch in range(trial_epochs):
        net.train()
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(inputs.size(0), -1)  # Flatten the input
            
            optimizer.zero_grad()  # Zero the gradients
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()  # Update parameters

            running_loss += loss.item()

        # Calculate and store the average training loss for the epoch
        avg_train_loss = running_loss / len(trainloader)
        train_losses.append(avg_train_loss)

        # Calculate test loss and accuracy on the test set
        net.eval()
        with torch.no_grad():
            test_loss = 0.0
            all_preds = []
            all_labels = []
            for data in testloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.view(inputs.size(0), -1)
                preds = net(inputs)
                loss = criterion(preds, labels)
                test_loss += loss.item()  # Accumulate test loss
                _, predicted = torch.max(preds, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            test_loss /= len(testloader)  # Calculate average test loss
            accuracy = accuracy_score(all_labels, all_preds)

            test_losses.append(test_loss)  # Append test loss to the list

        # Print epoch metrics
        print(f"Trial {trial+1}/{num_trials}, Epoch {epoch+1}/{trial_epochs} - Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}, Acc: {accuracy:.4f}")

    # Store trial metrics
    trial_metrics.append({
        "train_losses": train_losses,
        "test_losses": test_losses,
        "accuracy": accuracy
    })
    
# Calculate the total time taken for all trials
end_time = time.time()
total_time = end_time - start_time
print(f"Total time taken for all trials: {total_time:.2f} seconds")

# Calculate average training and testing losses and accuracy
avg_train_losses = [sum(metrics['train_losses']) / trial_epochs for metrics in trial_metrics]
avg_test_losses = [sum(metrics['test_losses']) / trial_epochs for metrics in trial_metrics]
avg_accuracies = [metrics['accuracy'] for metrics in trial_metrics]

# Calculate average standard deviations of training and testing losses
std_train_losses = [np.std(metrics['train_losses']) for metrics in trial_metrics]
std_test_losses = [np.std(metrics['test_losses']) for metrics in trial_metrics]

# Calculate and print average values
avg_train_loss = sum(avg_train_losses) / num_trials
avg_test_loss = sum(avg_test_losses) / num_trials
avg_accuracy = sum(avg_accuracies) / num_trials

# Calculate and print average standard deviations
avg_std_train_loss = sum(std_train_losses) / num_trials
avg_std_test_loss = sum(std_test_losses) / num_trials

# Concatenate the losses from all trials
all_train_losses = []
all_test_losses = []
for metrics in trial_metrics:
    all_train_losses.append(metrics['train_losses'])
    all_test_losses.append(metrics['test_losses'])

# Plot the metrics for all trials
plt.figure(figsize=(10, 6))
for trial in range(num_trials):
    plt.plot(range(1, trial_epochs + 1), all_train_losses[trial], label=f'Trial {trial+1} Train Loss')
    plt.plot(range(1, trial_epochs + 1), all_test_losses[trial], label=f'Trial {trial+1} Test Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Test Losses for All Trials')

# Add legend with average metrics
plt.legend() 

# Add average metrics annotations with adjusted x-coordinates
x_coord = 1.2  # Adjust the x-coordinate for better positioning
plt.text(x_coord, avg_train_loss + 0.05, f'Avg Train Loss: {avg_train_loss:.4f}', color='black')
plt.text(x_coord, avg_test_loss + 0.03, f'Avg Test Loss: {avg_test_loss:.4f}', color='gray')
plt.text(x_coord, avg_accuracy - 0.03, f'Avg Accuracy: {avg_accuracy:.4f}', color='green')
plt.text(x_coord, avg_std_train_loss - 0.05, f'Avg Std Train Loss: {avg_std_train_loss:.4f}', color='blue')
plt.text(x_coord, avg_std_test_loss - 0.07, f'Avg Std Test Loss: {avg_std_test_loss:.4f}', color='orange')
plt.tight_layout()  

# Save the plot as an image file with higher resolution
plt.savefig('metrics_plot.png', dpi=300)

plt.show()
