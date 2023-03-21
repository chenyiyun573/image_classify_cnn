"""
    In this code, we define a simple convolutional neural network and train it on the CIFAR-10 dataset. 
    We use the nn.DataParallel module to parallelize the training on multiple GPUs. 
    We also use the torch.utils.data.distributed.DistributedSampler to distribute the data across the GPUs.

    To run this code on multiple GPUs, you can simply use the CUDA_VISIBLE_DEVICES environment variable to specify which GPUs to use. 
    For example, to use the first two GPUs, you can run:
        $ CUDA_VISIBLE_DEVICES=0,1 python train.py
    This will run the training code on the first two GPUs. 
    If you have more GPUs, you can add them to the list separated by commas.
"""









import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data.distributed as distributed

# Set the number of GPUs to use
num_gpus = torch.cuda.device_count()

# Define the network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(-1, 64 * 4 * 4)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Load the data
train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_sampler = distributed.DistributedSampler(train_set)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, sampler=train_sampler)

# Use DataParallel to train the network on multiple GPUs
net = nn.DataParallel(net)

# Move the network and the data to the GPUs
net.cuda()
criterion.cuda()
train_sampler.set_epoch(0)

# Train the network
for epoch in range(10):
    train_sampler.set_epoch(epoch)
    for data, target in train_loader:
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
