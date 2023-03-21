# import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MyModel

node_rank = 0 # TODO node_rank is the id of worker, need to be modified for each worker.
num_nodes = 2 # TODO num_nodes is the total number of workers
num_epochs = 100
log_interval = 100

# set up distributed training environment
dist.init_process_group(backend='gloo', init_method='tcp://master_ip_address:port',
                        rank=node_rank, world_size=num_nodes)

# create the model and move it to the GPU
model = MyModel()
model = nn.parallel.DistributedDataParallel(model)

# define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# set up the data loader
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=num_nodes, rank=node_rank)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, sampler=train_sampler)

# training loop
for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# clean up distributed training environment
dist.destroy_process_group()
