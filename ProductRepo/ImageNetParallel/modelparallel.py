import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import torch.optim as optim
from torchvision.models import resnet152

class ModelParallelResNet152(nn.Module):
    def __init__(self, device1, device2):
        super(ModelParallelResNet152, self).__init__()

        self.device1 = device1
        self.device2 = device2

        model = resnet152(pretrained=False)

        self.layer1 = model.layer1.to(device1)
        self.layer2 = model.layer2.to(device1)
        self.layer3 = model.layer3.to(device2)
        self.layer4 = model.layer4.to(device2)
        self.fc = model.fc.to(device2)

        self.conv1 = model.conv1.to(device1)
        self.bn1 = model.bn1.to(device1)
        self.relu = model.relu.to(device1)
        self.maxpool = model.maxpool.to(device1)

        self.avgpool = model.avgpool.to(device2)

    def forward(self, x):
        x = x.to(self.device1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = x.to(self.device2)

        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def main():
    device1 = torch.device('cuda:0')
    device2 = torch.device('cuda:1')

    model_parallel = ModelParallelResNet152(device1, device2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_parallel.parameters(), lr=0.04, momentum=0.9)

    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = ImageFolder(root='/mnt/imagenet/ILSVRC2012_img_train_extracted', transform=train_transforms)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=16, pin_memory=True)

    epochs = 10
    for epoch in range(epochs):
        running_loss = 0.0
        running_corrects = 0
        total = 0
        model_parallel.train()
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model_parallel(inputs)
            loss = criterion(outputs, labels.to(device2))
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.to(device2))
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total
        print(f"Epoch {epoch + 1}/{epochs} Loss: {epoch_loss:.5f} Acc: {100 * epoch_acc:.2f}%")

if __name__ == '__main__':
    main()
