import torch
import json
from datetime import datetime
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from NetworkInfoNode import NetworkInfoNode

with open('tree_structure.json', 'r') as f:
    labels_tree_dict1 = json.load(f)

root = NetworkInfoNode(labels_tree_dict1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(model_path, num_classes, device):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    return model

def load_models(node: NetworkInfoNode, device):
    num_classes = len(node.output_label_list)
    model_path = f"./models/{node.input_label}_model.pth"
    node.model = load_model(model_path, num_classes, device)
    print(f"Loaded {node.input_label} model from {model_path}")

    for child in node.children:
        load_models(child, device)

load_models(root, device)

transform = transforms.Compose([
    transforms.RandomResizedCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


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



def set_trainloader(Node: NetworkInfoNode):
    # Create the custom dataset and DataLoader
    if Node.trainloader != None:
        return
    filtered_dataset = FilteredImageFolder('/mnt/imagenet/ILSVRC2012_img_train_extracted', transform=transform, selected_ids=Node.input_id_list, label_mapping=Node.label_mapping)
    
    Node.trainloader = torch.utils.data.DataLoader(filtered_dataset, batch_size=256, shuffle=True, num_workers=0)
    print(f"{Node.input_label} Trainloader set.")
    if Node.children != []:
        
        for child in Node.children:
            set_trainloader(child)
            
            
set_trainloader(root)


def predict_label(node, image):
    with torch.no_grad():
        outputs = node.model(image)
        _, predicted = torch.max(outputs.data, 1)
        next_label = node.trainloader.dataset.classes[predicted.item()]
        if node.end_level:
            return next_label
        else:
            for child in node.children:
                if child.input_label == next_label:
                    return predict_label(child, image)
            raise LookupError("labels match error")



def get_total_network_accuracy(network, model, trainset, device, selected_ids, label_mapping, batch_size=32):
    correct = 0
    total = 0
    model.eval()
    imagenet_dataset = datasets.ImageFolder(root='/mnt/imagenet/ILSVRC2012_img_train_extracted', transform=transform)
    imagenet_loader = DataLoader(imagenet_dataset, batch_size=64, shuffle=True, num_workers=4)

    with torch.no_grad():
        for images, labels in imagenet_loader:
            images, labels = images.to(device), labels.to(device)
            total += labels.size(0)
            predict_labels = [predict_label(root, image.unsqueeze(0)) for image in images]
            label_strings = [imagenet_loader.dataset.classes[l.item()] for l in labels]  # Convert tensor labels to string class names
            # print(predict_labels)
            # print(label_strings)
            correct += sum(1 for p, n in zip(predict_labels, label_strings) if p == n)  # Compare and count matching labels
            if total % (batch_size * 100) == 0:  # Adjust the print frequency based on batch_size
                print(f"Network {network} total {total} correct {correct} accuracy {correct / total * 100:.2f} time {datetime.now()}")
    return correct / total


def evaluate_total_networks(node, trainset, device):
    accuracy = get_total_network_accuracy(node.input_label, node.model, trainset, device, node.input_id_list, node.label_mapping)
    print(f"{node.input_label} trainset accuracy: {accuracy * 100:.2f}%  time {datetime.now()}")

trainset = datasets.ImageFolder(root='/mnt/imagenet/ILSVRC2012_img_train_extracted', transform=transform)
evaluate_total_networks(root, trainset, device)
