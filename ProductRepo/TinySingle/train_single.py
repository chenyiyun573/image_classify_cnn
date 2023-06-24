import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import time
import argparse
from constants import dataset_path, batch_size, get_model_and_device, epoch_c

# Add argument parsing
parser = argparse.ArgumentParser(description='Train a model on a specified GPU')
parser.add_argument('--gpu', type=int, required=True, help='GPU ID to use for training')
args = parser.parse_args()

model_name = str(args.gpu)
model_c, device_c = get_model_and_device(model_name)

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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Load the model
model = model_c

# Update the last layer for the number of classes in Tiny ImageNet
# Update the last layer for the number of classes in Tiny ImageNet
if isinstance(model_c, (torchvision.models.AlexNet, torchvision.models.SqueezeNet)):
    num_ftrs = model_c.classifier[6].in_features
    model_c.classifier[6] = nn.Linear(num_ftrs, 200)
elif isinstance(model_c, torchvision.models.Inception3):
    num_ftrs = model_c.fc.in_features
    model_c.fc = nn.Linear(num_ftrs, 200)
    model_c.AuxLogits.fc = nn.Linear(768, 200)
else:
    num_ftrs = model_c.fc.in_features
    model_c.fc = nn.Linear(num_ftrs, 200)


# Move the model to GPU
device = device_c
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model and evaluate on the test set
start_time = time.time()
for epoch in range(epoch_c):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_loss /= len(trainloader)
    train_accuracy = 100 * train_correct / train_total

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(testloader)
    test_accuracy = 100 * correct / total
    epoch_time = time.time() - start_time
    print(f'Epoch {epoch + 1}, Time: {epoch_time:.2f}s, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

total_time = time.time() - start_time
print(f'Total training time: {total_time:.2f}s')
