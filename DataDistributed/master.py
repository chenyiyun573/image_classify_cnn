import subprocess
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from model import MyModel

# set up distributed training environment
num_nodes = 4
num_epochs = 100



port = 1234
dist.init_process_group(backend='gloo', init_method=f'tcp://localhost:{port}', world_size=num_nodes)

# launch worker nodes
processes = []
for i in range(num_nodes):
    cmd = ['python', 'train.py', f'--node_rank={i}', f'--num_nodes={num_nodes}', f'--port={port}']
    process = subprocess.Popen(cmd)
    processes.append(process)

# synchronize model state across worker nodes
model = MyModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)
model = nn.parallel.DistributedDataParallel(model)

for param in model.parameters():
    dist.broadcast(param.data, src=0)


# training loop (master node can also participate in training)
for epoch in range(num_epochs):
    # do whatever you need to coordinate training
    # clean up distributed training environment
    dist.destroy_process_group()
