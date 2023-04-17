import torch
import json

from torchvision import models, datasets, transforms

model_load_path = './models'
trained_models = {}

with open('output.json', 'r') as f:
    sub_networks = json.load(f)


root_mapping = {'Others': 0, 'Objects': 1, 'Animals': 2 }
# Transformations for Tiny ImageNet
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(64, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = datasets.ImageFolder(root='/home/superbench/v-yiyunchen/net/re-tiny-imagenet-200/train', transform=transform)


all_classes = [class_id for class_id_list in sub_networks.values() for class_id in class_id_list]
def filter_trainset(trainset, all_classes):
    filtered_indices = [idx for idx, (_, target) in enumerate(trainset) if trainset.classes[target] in all_classes]
    filtered_trainset = torch.utils.data.Subset(trainset, filtered_indices)
    return filtered_trainset
filtered_trainset = filter_trainset(trainset, all_classes)



sub_networks.update({'Root': ['Animals', 'Objects', 'Others'],})

for network in sub_networks.keys():
    model = models.resnet18(pretrained=True)
    num_classes = len(sub_networks[network])
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(torch.load(f"{model_load_path}/{network}_model.pth"))
    model.eval()
    trained_models[network] = model


def get_final_predicted_label(image, root_model, sub_models, device):
    image = image.unsqueeze(0).to(device)
    root_output = root_model(image)
    _, root_predicted = torch.max(root_output.data, 1)
    root_class = list(root_mapping.keys())[list(root_mapping.values()).index(root_predicted.item())]

    sub_model = sub_models[root_class]
    sub_output = sub_model(image)
    _, sub_predicted = torch.max(sub_output.data, 1)

    sub_classes = sub_networks[root_class]
    final_label = sub_classes[sub_predicted.item()]

    return final_label



def get_hierarchical_accuracy(root_model, sub_models, filtered_trainset, device):
    correct = 0
    total = 0
    for i in range(len(filtered_trainset)):
        image, target = filtered_trainset[i]
        target_label = trainset.classes[target]
        predicted_label = get_final_predicted_label(image, root_model, sub_models, device)

        total += 1
        if target_label == predicted_label:
            correct += 1
        print(f'correct {correct}  total {total}')

    return correct / total



root_model = trained_models['Root']
sub_models = {k: v for k, v in trained_models.items() if k != 'Root'}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root_model.to(device)
for sub_model in sub_models.values():
    sub_model.to(device)

hierarchical_accuracy = get_hierarchical_accuracy(root_model, sub_models, filtered_trainset, device)
print(f"Hierarchical trainset accuracy: {hierarchical_accuracy * 100:.2f}%")

