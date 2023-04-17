import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
import json

# Load Tiny ImageNet dataset and create a mapping for each sub-network
# Load your hierarchy JSON file
with open('output.json', 'r') as f:
    sub_networks = json.load(f)

sub_networks.update({'Root': ['Animals', 'Objects', 'Others'],})

# Define a mapping for the 'Root' network
root_mapping = {'Others': 0, 'Objects': 1, 'Animals': 2 }
all_classes = [class_name for classes in sub_networks.values() for class_name in classes]

# Modify target_transform_factory for the 'Root' network
def root_target_transform_factory(root_mapping):
    def root_target_transform(target):
        search_value = target
        try:
            found_key = [key for key, value in sub_networks.items() if search_value in value][0]
        except IndexError:
            print(search_value)
            print('Not Found')
            exit()
        return root_mapping[found_key]
    return root_target_transform


# Transformations for Tiny ImageNet
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(64, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.ImageFolder(root='/home/superbench/v-yiyunchen/net/re-tiny-imagenet-200/train', transform=transform)
"""
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, target_transform=None):
        self.original_dataset = original_dataset
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.original_dataset[index]

        if self.target_transform is not None:
            target = self.original_dataset.dataset.classes[target]
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.original_dataset)
"""

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, target_transform=None):
        self.original_dataset = original_dataset
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.original_dataset[index]

        if self.target_transform is not None:
            target = self.original_dataset.classes[target]
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.original_dataset)


# for debugging
def print_labels(dataset):
    labels = set()
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.add(label)

    print(f"Number of labels: {len(labels)}")
    print("Labels: ", labels)

from torch.utils.data import Dataset

class FilteredDataset(Dataset):
    def __init__(self, original_dataset, filtered_indices, label_mapping):
        self.original_dataset = original_dataset
        self.filtered_indices = filtered_indices
        self.label_mapping = label_mapping

    def __getitem__(self, idx):
        data, target = self.original_dataset[self.filtered_indices[idx]]
        return data, self.label_mapping[self.original_dataset.classes[target]]

    def __len__(self):
        return len(self.filtered_indices)


# Create a trainloader for each sub-network
trainloaders = {}
for network, classes in sub_networks.items():
    if network == 'Root':
        root_dataset = CustomDataset(trainset, target_transform=root_target_transform_factory(root_mapping))
        trainloader = torch.utils.data.DataLoader(root_dataset, batch_size=100, shuffle=True, num_workers=2)
        print_labels(root_dataset)
    else:
        filtered_indices = [idx for idx, (_, target) in enumerate(trainset) if trainset.classes[target] in classes]

        # Create a dictionary to map the original labels to new labels
        label_mapping = {original_label: new_label for new_label, original_label in enumerate(classes)}
        print(label_mapping)
        filtered_trainset = FilteredDataset(trainset, filtered_indices, label_mapping)

        trainloader = torch.utils.data.DataLoader(filtered_trainset, batch_size=100, shuffle=True, num_workers=2)
        
        
        
        print_labels(filtered_trainset)
        unique_labels = set()

        for _, labels in trainloader:
            unique_labels.update(torch.unique(labels).tolist())

        num_labels = len(unique_labels)
        print(f"Number of labels: {num_labels}")

    trainloaders[network] = trainloader
    
import sys
# Function to create and train a model on a specific GPU
def train_model(network, trainloader, gpu_id):
    model = models.resnet18(pretrained=True)
    num_classes = len(sub_networks[network])
    print(f"number classes{num_classes} in {network}")
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[gpu id: {gpu_id}  epoch{epoch + 1}, {network}] loss: {running_loss / (i + 1)}")
        sys.stdout.flush()

    print(f"Finished training {network}")
    return model







import torch.multiprocessing as mp

def train_on_gpu(network, trainloader, gpu_id):
    trained_model = train_model(network, trainloader, gpu_id)
    return (network, trained_model)


# Train models on different GPUs
gpu_ids = [0, 1, 2, 3]
trained_models = {}
processes = []

"""
for gpu_id, (network, trainloader) in zip(gpu_ids, trainloaders.items()):
    process = mp.Process(target=train_on_gpu, args=(network, trainloader, gpu_id))
    processes.append(process)
    process.start()
    print(f' gpu_id {gpu_id} network {network} process created ')

for process in processes:
    process.join()
"""
import time

for gpu_id, (network, trainloader) in zip(gpu_ids, trainloaders.items()):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f' gpu_id {gpu_id} network {network} training started at current time {current_time}')
    network, trained_model = train_on_gpu(network, trainloader, gpu_id)
    trained_models[network] = trained_model
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f' gpu_id {gpu_id} network {network} training finished at current time {current_time}')

import os

model_save_path = './models'
os.makedirs(model_save_path, exist_ok=True)

for network, model in trained_models.items():
    torch.save(model.state_dict(), f"{model_save_path}/{network}_model.pth")

print('models saved.')


