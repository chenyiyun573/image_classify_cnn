import os
import time
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
from efficientnet_pytorch import EfficientNet

from constants import dataset_path
from torchvision.models import resnet18


# Hyperparameters
batch_size = 32
epochs = 10
learning_rate = 0.001
momentum = 0.9
num_classes = 200
num_workers = 16
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(rank, world_size):
    torch.manual_seed(0)
    device = rank
    # initialize distributed process group
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', rank=rank, world_size=world_size)

    # define data loading and augmentation for Tiny ImageNet dataset
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = datasets.ImageFolder(root=dataset_path+'re-tiny-imagenet-200/train', transform=train_transforms)
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_set = datasets.ImageFolder(root=dataset_path+'re-tiny-imagenet-200/val', transform=test_transforms)
    test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank)
    test_loader = DataLoader(test_set, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers, pin_memory=True)

    # define the ResNet-18 model
    #model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=False)
    model = resnet18(pretrained=True) 
    # model.fc = nn.Linear(model.fc.in_features, num_classes)
    #model.to(device)
    model = model.to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    print('ready to train')
    # train the model
    for epoch in range(epochs):
        # set the model to train mode
        model.train()

        # train for one epoch
        epoch_start_time = time.time()
        train_loss = 0.0
        train_acc = 0.0
        num_samples_train = 0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            num_samples_train += len(inputs)
            train_loss += loss.item() * len(inputs)
            train_acc += (outputs.max(1)[1] == targets).sum().item()


        train_loss /= num_samples_train
        train_acc /= num_samples_train

        # synchronize the model
        dist.barrier()

        # compute all reduce mean of the loss and accuracy
        train_loss_reduce = torch.tensor(train_loss).to(device)
        dist.all_reduce(train_loss_reduce)
        train_loss_reduce /= world_size

        train_acc_reduce = torch.tensor(train_acc).to(device)
        dist.all_reduce(train_acc_reduce)
        train_acc_reduce /= world_size

        # set the model to eval mode
        model.eval()

        # evaluate on the test set
        with torch.no_grad():
            test_loss = 0.0
            test_acc = 0.0
            num_samples_test = 0
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                num_samples_test += len(inputs)
                test_loss += loss.item() * len(inputs)
                test_acc += (outputs.max(1)[1] == targets).sum().item()

            test_loss /= num_samples_test
            test_acc /= num_samples_test

        # compute all reduce mean of the test loss and accuracy
        test_loss_reduce = torch.tensor(test_loss).to(device)
        dist.all_reduce(test_loss_reduce)
        test_loss_reduce /= world_size

        test_acc_reduce = torch.tensor(test_acc).to(device)
        dist.all_reduce(test_acc_reduce)
        test_acc_reduce /= world_size

        # print the loss and accuracy for each epoch
        if rank == 0:
            epoch_end_time = time.time()
            print(f"[{datetime.now()}]: [{epoch+1}/{epochs}] Train Loss: {train_loss_reduce:.5f}, Train Acc: {100*train_acc_reduce:.2f}%, Test Loss: {test_loss_reduce:.5f}, Test Acc: {100*test_acc_reduce:.2f}%, Time: {(epoch_end_time-epoch_start_time):.3f}s")

    # destroy the process group
    dist.destroy_process_group()

if __name__ == '__main__':
    mp.spawn(train, args=(6,), nprocs=6, join=True)

