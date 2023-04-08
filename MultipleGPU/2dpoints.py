import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

def train(rank, world_size):

    # initialize distributed process group
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', rank=rank, world_size=world_size)

    # create the model and move it to the current device
    model = MyModel().to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # initialize the optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # load the data and train for some number of epochs
    for epoch in range(10):
        data = torch.randn(100, 10).to(rank)
        target = torch.randint(0, 2, (100,)).to(rank)
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # print the loss for each epoch
        if rank == 0:
            print(f"Epoch {epoch}, Loss {loss.item()}")

    # destroy the process group
    dist.destroy_process_group()

if __name__ == '__main__':
    # spawn two processes, one for each GPU
    mp.spawn(train, args=(2,), nprocs=2, join=True)

