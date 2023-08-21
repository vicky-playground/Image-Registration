import torch
import torch.nn as nn
import torch.optim as optim
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18 # pre trained models for Image Classification
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import copy
print(f"device: {device}")


# set a fixed seed for reproducibility
seed = 42
torch.manual_seed(seed)

# load CIFAR-10 dataset and apply transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# define the neural network architecture
net = resnet18(pretrained=False, num_classes=10).to(device)

# define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# lists to store loss values
losses = []

# train the neural network with perturbation
num_epochs = 3000
input_weights = None

for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # calculate accuracy on training set
    net.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for data in trainloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            preds = net(inputs)
            _, predicted = torch.max(preds, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(trainloader)} - Accuracy: {accuracy:.4f}")

    # append current loss to the list
    losses.append(running_loss/len(trainloader))

    """
    Perturbation:
    1. Generate random noise proportional to weight changes.
    2. Apply constraints to noise to prevent extreme changes.
    """
    current_weights = copy.deepcopy(net.state_dict())
    if epoch > 0: # perturbation starts from the second epoch
        perturbation_scale = 0.1

        for name, weight in current_weights.items():
            distance = (current_weights[name].float() - input_weights[name].float()).float()

            # get the vector magnitude and then extract it
            distance_scalar = torch.norm(distance).item()

            # convert weight tensor to float for consistent data type
            weight_float = weight.to(torch.float)

            # create a random perturbation that is proportional to the distance between the current weights and initial weights
            perturbation = torch.randn_like(weight_float) * distance_scalar * perturbation_scale 

            # If any element of the perturbation tensor is less than this lower bound, it is replaced with -distance_scalar
            perturbation = torch.where(perturbation < -distance_scalar, -perturbation.new_tensor(distance_scalar), perturbation) # lower bound 
            # If any element of the perturbation tensor is greater than this upper bound, it is replaced with distance_scalar
            perturbation = torch.where(perturbation > distance_scalar, perturbation.new_tensor(distance_scalar), perturbation) # upper bound

            # add the perturbation tensor to the initial weights of previous epoch
            current_weights[name] = input_weights[name].float() + perturbation

        net.load_state_dict(current_weights)

    input_weights = copy.deepcopy(current_weights)


# plot the loss values
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

"""
When the train is finished, we have to save the model in order to use it for testing or actual usage.
"""
PATH = './cifar_noise.pth'
torch.save(net.state_dict(), PATH)

"""
Test:
1. show image first.
2. then load Models that we have saved during the training process.
"""
net = resnet18(num_classes=10).to(device)
net.load_state_dict(torch.load(PATH))

# check the accuracy of the model:
dataiter = iter(testloader) 
images, labels = next(dataiter)
images = images.to(device)  
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)  
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy:.4f}")

"""
# see pictures from training and testing:
import numpy as np
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
dataiter = iter(testloader) 
images, labels = next(dataiter)
# print images
imshow(torchvision.utils.make_grid(images))
"""

# print GROUND TRUTH (labels)
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
# print prediction results
outputs = net(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))