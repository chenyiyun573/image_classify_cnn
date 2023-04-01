import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torch.utils.data import DataLoader

# Define the number of GPUs to use
num_gpus = 2

# Define the batch size per GPU
batch_size = 32

# Define the transforms to be applied to the data
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load the ImageNet dataset
dataset = datasets.ImageNet('/path/to/imagenet/', split='val', transform=transform)

# Define the data loader
data_loader = DataLoader(dataset, batch_size=batch_size*num_gpus, shuffle=False, num_workers=4*num_gpus, pin_memory=True)

# Define the model and move it to the GPUs
model = models.resnet50(pretrained=True)
model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
model.cuda()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Set the model to evaluation mode
model.eval()

# Loop over the batches of data
with torch.no_grad():
    for i, (inputs, targets) in enumerate(data_loader):
        # Move the data to the GPUs
        inputs = inputs.cuda()
        targets = targets.cuda()

        # Compute the output
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, targets)

        # Compute the accuracy
        _, predicted = torch.max(outputs, 1)
        correct = predicted.eq(targets).sum().item()
        accuracy = correct / targets.size(0)

        # Print the batch number, loss, and accuracy
        print('Batch {} - Loss: {:.4f}, Accuracy: {:.2f}%'.format(i+1, loss.item(), accuracy*100))
