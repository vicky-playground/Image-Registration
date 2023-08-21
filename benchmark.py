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

"""
In the data manipulation process, there are 3 main parts.
1. transform is the part where we will define the transformation. Change the image as required, such as resize, rotate, but the important part is ToTensor and Normalize.
2. dataset or trainset, testset in the example if it's easy to explain. In this part, we take the data to transform as in step 1
3. dataloader is to change the dataset that we set in step 2 to change the arrangement format to be ready to be used in the neural network, which will have variables Batch size is related to batch size.
Batch size is the number of images that we will insert for the machine to learn in 1 learning cycle.
"""

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

# define the neural network architecture: the output of the previous layer must be equal to the input of the next layer.
net = resnet18(num_classes=10).to(device)
print(net) # Show the layer architecture of the ResNet model. See what’s inside those deep hidden layers.


# define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

num_epochs = 50
input_weights = None 
train_losses = [] # store loss values for plotting

# print the initial weights before training
initial_weights = copy.deepcopy(net.state_dict())
print("Initial Weights:")
for name, param in initial_weights.items():
    print(name, param)


"""
Train the neural network:
Load the batches of images and do the feed forward loop. Then calculate the loss function, and use the optimizer to apply gradient descent in back-propagation.
update the weights until the point where the loss hardly changes and stop or call Converged
"""
# loop over the dataset multiple times
for epoch in range(num_epochs): 
    net.train()
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        # data is a list of [inputs, labels] from torchvision
        inputs, labels = data 
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print result each step
        running_loss += loss.item()
        
        # calculate and store the average training loss for the epoch
        avg_train_loss = running_loss / len(trainloader)
        train_losses.append(avg_train_loss)

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
            all_preds.extend(predicted.cpu().numpy()) # convert tensors to NumPy arrays 
            all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(trainloader)} - Accuracy: {accuracy:.4f}")

# plot the change of loss metrics
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

"""
When the train is finished, we have to save the model in order to use it for testing or actual usage.
"""
PATH = './cifar.pth'
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
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
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