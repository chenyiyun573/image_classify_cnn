import json
import multiprocessing as mp
import sys
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import Dataset

from NetworkInfoNode import NetworkInfoNode

if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn')

dataset_path = '/mnt/imagenet/ILSVRC2012_img_train_extracted'
with open('tree_structure.json', 'r') as f:
    labels_tree_dict1 = json.load(f)

root = NetworkInfoNode(labels_tree_dict1)

class FilteredImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, selected_ids=None, label_mapping=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        if selected_ids is not None and label_mapping is not None:
            # Get the selected_classes (indices) from the selected_ids list
            selected_classes = [self.class_to_idx[id] for id in selected_ids]
            
            # Create a mapping from general category names to indices
            general_category_indices = {category: i for i, category in enumerate(sorted(set(label_mapping.values())))}
            
            # Create a mapping from the original class index to the new root label index
            class_idx_mapping = {class_idx: general_category_indices[label_mapping[id]] for id, class_idx in zip(selected_ids, selected_classes)}

            # Filter the dataset samples based on the selected_classes list and update labels
            self.samples = [(s, class_idx_mapping[c]) for s, c in self.samples if c in class_idx_mapping]
            
            self.targets = [class_idx_mapping[c] for c in self.targets if c in class_idx_mapping]

            # Update the class_to_idx and classes attributes
            self.class_to_idx = {cls: class_idx_mapping[idx] for cls, idx in self.class_to_idx.items() if idx in class_idx_mapping}
            self.classes = list(general_category_indices.keys())
            



transform = transforms.Compose([
    transforms.RandomResizedCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def set_trainloader(Node: NetworkInfoNode):
    # Create the custom dataset and DataLoader
    if Node.trainloader != None:
        return
    filtered_dataset = FilteredImageFolder(dataset_path, transform=transform, selected_ids=Node.input_id_list, label_mapping=Node.label_mapping)
    
    Node.trainloader = torch.utils.data.DataLoader(filtered_dataset, batch_size=256, shuffle=True, num_workers=0)
    print(f"{Node.input_label} Trainloader set.")
    if Node.children != []:
        
        for child in Node.children:
            set_trainloader(child)



def train_model(node: NetworkInfoNode, trainloader, gpu_id):
    network = node.input_label

    model = models.resnet18(pretrained=True)
    num_classes = len(node.output_label_list)
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
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[gpu id: {gpu_id}  epoch{epoch + 1}, {network}] loss: {running_loss / (i + 1)} time: {datetime.now()}")
        sys.stdout.flush()

    print(f"Finished training {network}")
    return model


def start_processes(node, gpu_id=0):
    processes = []

    # Train the current node
    process = mp.Process(target=train_model, args=(node, node.trainloader, gpu_id))
    processes.append(process)
    process.start()
    print(f' gpu_id {gpu_id} network {node.input_label} process created ')

    # Train the children nodes
    for i, child in enumerate(node.children):
        next_gpu_id = (gpu_id + i + 1) % 8  # Get the next GPU ID, wrapping around at 8
        child_processes = start_processes(child, next_gpu_id)
        processes.extend(child_processes)

    return processes

if __name__ == '__main__':
    set_trainloader(root)
    
    processes = start_processes(root)
    for process in processes:
        process.join()
    
