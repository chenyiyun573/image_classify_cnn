import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from torch.utils.data import DataLoader


def train_network(network, train_loader, device, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {running_loss / (i + 1)}')

    return network


def create_network(classes, device):
    network = resnet50(pretrained=False)
    num_features = network.fc.in_features
    network.fc = nn.Linear(num_features, len(classes))
    network.to(device)
    return network


def train_hierarchy(hierarchy, train_data_path, device):
    trained_models = {}
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for category, subcategories in hierarchy.items():
        if isinstance(subcategories, dict):
            trained_models[category] = train_hierarchy(subcategories, train_data_path, device)
        elif isinstance(subcategories, list):
            # classes = [str(class_id) for class_id in subcategories]
            classes = subcategories
            print(classes)
            dataset = ImageFolder(train_data_path, transform=transform, target_transform=lambda x: classes.index(x))
            train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

            network = create_network(classes, device)
            trained_network = train_network(network, train_loader, device)
            trained_models[category] = {"model": trained_network, "classes": classes}

    return trained_models


# Load your hierarchy JSON file
with open('output_dict.json', 'r') as f:
    hierarchy = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data_path = '/home/superbench/v-yiyunchen/net/re-tiny-imagenet-200/train'
trained_models = train_hierarchy(hierarchy, train_data_path, device)
