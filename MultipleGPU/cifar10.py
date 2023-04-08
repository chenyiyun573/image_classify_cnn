import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.distributed as dist
import torch.multiprocessing as mp

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # input size: 3x32x32
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(rank, world_size):
    torch.manual_seed(0)

    # initialize distributed process group
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', rank=rank, world_size=world_size)

    # load and process the CIFAR10 dataset
    if rank == 0:
        if not os.path.exists('./data'):
            os.mkdir('./data')
        train_set = datasets.CIFAR10(root='./data', train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]
                                         )
                                     ]))
        test_set = datasets.CIFAR10(root='./data', train=False, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]
                                        )
                                    ]))
    else:
        train_set = datasets.CIFAR10(root='./data', train=True, download=False,
                                     transform=transforms.Compose([
                                         transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]
                                         )
                                     ]))
        test_set = datasets.CIFAR10(root='./data', train=False, download=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]
                                        )
                                    ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    train_loader = data.DataLoader(train_set, batch_size=128, sampler=train_sampler)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, num_replicas=world_size, rank=rank)
    test_loader = data.DataLoader(test_set, batch_size=256, sampler=test_sampler)

    # create the model and move it to the current device
    model = Net().to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # initialize the optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # train the model
    for epoch in range(100):
        # set the model to train mode
        model.train()

        for batch_idx, (data_batch, target_batch) in enumerate(train_loader):
            data_batch, target_batch = data_batch.to(rank), target_batch.to(rank)
            optimizer.zero_grad()
            output = model(data_batch)
            loss = nn.CrossEntropyLoss()(output, target_batch)
            loss.backward()
            optimizer.step()

        # synchronize the model
        dist.barrier()

        # compute all reduce mean of the loss
        loss_reduce = torch.tensor(loss.item()).to(rank)
        dist.all_reduce(loss_reduce)
        loss_reduce /= world_size

        # set the model to eval mode
        model.eval()

        # print the loss for each epoch
        if rank == 0:
            print(f"Epoch {epoch}, Loss {loss_reduce.item()}")

        # evaluate the model on the test set
        model.eval()
        num_correct = 0
        num_examples = 0
        for data_batch, target_batch in test_loader:
            data_batch, target_batch = data_batch.to(rank), target_batch.to(rank)
            output = model(data_batch)
            _, predictions = torch.max(output, 1)
            num_correct += (predictions == target_batch).sum()
            num_examples += predictions.size(0)

        # compute the accuracy
        accuracy = 100.0 * num_correct / num_examples
        dist.all_reduce(accuracy)
        accuracy_reduce = accuracy / world_size

        # print the accuracy
        if rank == 0:
            print(f"Epoch {epoch}, Accuracy {accuracy_reduce.item():.2f}%")

    # destroy the process group
    dist.destroy_process_group()

if __name__ == '__main__':
    mp.spawn(train, args=(2,), nprocs=2, join=True)

