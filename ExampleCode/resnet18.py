import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torchvision.models import resnet18

from constants import dataset_path

"""
# Define transforms for the dataset
transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])"""

# Transformations for Tiny ImageNet
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(64, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Load the dataset
trainset = torchvision.datasets.ImageFolder(root=dataset_path+'re-tiny-imagenet-200/train', transform=transform)
testset = torchvision.datasets.ImageFolder(root=dataset_path+'re-tiny-imagenet-200/val', transform=transform)

# Define the dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Load the model
model = resnet18(pretrained=True)  # Change this line to load the pretrained ResNet-18 model

# Update the last layer for the number of classes in Tiny ImageNet
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 200)

# Set the model to training mode
model.train()

# Move the model to GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"device using is {device}")
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

from datetime import datetime
start_train_time = datetime.now()
# Train the model
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')
current_time = datetime.now()
print("Total training Time:", current_time-start_train_time)


# Set the model to evaluation mode
model.eval()

# Classify the images
with torch.no_grad():
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
print()