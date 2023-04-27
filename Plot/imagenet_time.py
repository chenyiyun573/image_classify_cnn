import re
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import cm

with open('train3.out', 'r') as file:
    log = file.read()

# Parse the log and store the loss values
loss_values = {}
time_values = {}
time_format = "%Y-%m-%d %H:%M:%S.%f"

for line in log.splitlines():
    match = re.match(r'\[gpu id: (\d)  epoch(\d+), (.+?)\] loss: (.+?) time: (.+)', line)
    if match:
        gpu_id, epoch, label, loss, timestamp = match.groups()
        key = f'{label}'
        if key not in loss_values:
            loss_values[key] = []
            time_values[key] = []
        loss_values[key].append((int(epoch), float(loss)))
        time_values[key].append(datetime.strptime(timestamp, time_format))

# Group data by network name
grouped_data = {}
for key, values in loss_values.items():
    label = key
    if label not in grouped_data:
        grouped_data[label] = []
    for epoch, loss in values:
        time = time_values[key][epoch - 1]
        grouped_data[label].append({'epoch': epoch, 'loss': loss, 'time': time})

# Create colormap
cmap = cm.get_cmap('tab20', len(grouped_data))

# Plot
fig, ax = plt.subplots()
for idx, (label, data) in enumerate(grouped_data.items()):
    time_data_sec = [(t - data[0]['time']).total_seconds() for t in [d['time'] for d in data]]
    ax.plot(time_data_sec, [d['loss'] for d in data], color=cmap(idx), linewidth=1, label=label)

ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Loss')
ax.legend()
plt.title('Loss vs. Time')

plt.show()
