import matplotlib.pyplot as plt
import datetime
import re
import numpy as np

data = '''
[2023-04-23 05:12:16.951050]: [1/30] Train Loss: 6.17805, Train Acc: 2.45%, Test Loss: 5.61739, Test Acc: 5.02%, Time: 187.794s
[2023-04-23 05:15:23.599218]: [2/30] Train Loss: 5.24589, Train Acc: 7.66%, Test Loss: 5.22023, Test Acc: 7.59%, Time: 186.647s
[2023-04-23 05:18:30.954190]: [3/30] Train Loss: 4.72892, Train Acc: 12.53%, Test Loss: 4.38267, Test Acc: 15.16%, Time: 187.354s
[2023-04-23 05:21:37.798395]: [4/30] Train Loss: 4.34417, Train Acc: 16.98%, Test Loss: 4.07669, Test Acc: 18.91%, Time: 186.844s
[2023-04-23 05:24:44.900654]: [5/30] Train Loss: 4.03934, Train Acc: 20.82%, Test Loss: 3.86763, Test Acc: 21.54%, Time: 187.102s
[2023-04-23 05:27:52.178538]: [6/30] Train Loss: 3.79621, Train Acc: 24.18%, Test Loss: 3.54493, Test Acc: 26.33%, Time: 187.277s
[2023-04-23 05:30:58.265616]: [7/30] Train Loss: 3.60012, Train Acc: 27.10%, Test Loss: 3.40213, Test Acc: 28.51%, Time: 186.086s
[2023-04-23 05:34:05.863846]: [8/30] Train Loss: 3.42837, Train Acc: 29.70%, Test Loss: 3.23980, Test Acc: 31.11%, Time: 187.598s
[2023-04-23 05:37:12.564239]: [9/30] Train Loss: 3.28588, Train Acc: 31.97%, Test Loss: 3.08829, Test Acc: 33.65%, Time: 186.700s
[2023-04-23 05:40:20.623256]: [10/30] Train Loss: 3.16967, Train Acc: 33.84%, Test Loss: 3.01928, Test Acc: 34.73%, Time: 188.058s
[2023-04-23 05:43:28.158304]: [11/30] Train Loss: 3.06155, Train Acc: 35.66%, Test Loss: 2.85226, Test Acc: 37.67%, Time: 187.534s
[2023-04-23 05:46:35.960578]: [12/30] Train Loss: 2.97165, Train Acc: 37.12%, Test Loss: 2.92995, Test Acc: 36.47%, Time: 187.801s
[2023-04-23 05:49:44.140095]: [13/30] Train Loss: 2.88971, Train Acc: 38.56%, Test Loss: 2.70343, Test Acc: 40.13%, Time: 188.179s
[2023-04-23 05:52:52.039956]: [14/30] Train Loss: 2.81545, Train Acc: 39.82%, Test Loss: 2.66961, Test Acc: 40.89%, Time: 187.899s
[2023-04-23 05:56:00.493306]: [15/30] Train Loss: 2.74941, Train Acc: 40.95%, Test Loss: 2.51316, Test Acc: 43.58%, Time: 188.452s
[2023-04-23 05:59:08.016249]: [16/30] Train Loss: 2.68921, Train Acc: 42.00%, Test Loss: 2.59405, Test Acc: 42.29%, Time: 187.522s
[2023-04-23 06:02:15.686668]: [17/30] Train Loss: 2.63507, Train Acc: 42.99%, Test Loss: 2.50619, Test Acc: 43.89%, Time: 187.670s
[2023-04-23 06:05:23.482275]: [18/30] Train Loss: 2.58410, Train Acc: 43.87%, Test Loss: 2.44303, Test Acc: 45.06%, Time: 187.795s
[2023-04-23 06:08:31.345692]: [19/30] Train Loss: 2.53512, Train Acc: 44.82%, Test Loss: 2.52084, Test Acc: 43.81%, Time: 187.863s
[2023-04-23 06:11:39.587805]: [20/30] Train Loss: 2.49482, Train Acc: 45.46%, Test Loss: 2.42361, Test Acc: 45.39%, Time: 188.241s
[2023-04-23 06:14:46.858311]: [21/30] Train Loss: 2.45634, Train Acc: 46.20%, Test Loss: 2.30003, Test Acc: 47.76%, Time: 187.269s
[2023-04-23 06:17:53.233237]: [22/30] Train Loss: 2.41591, Train Acc: 47.00%, Test Loss: 2.28702, Test Acc: 47.76%, Time: 186.374s
[2023-04-23 06:21:01.127629]: [23/30] Train Loss: 2.38309, Train Acc: 47.58%, Test Loss: 2.36571, Test Acc: 46.29%, Time: 187.894s
[2023-04-23 06:24:09.073164]: [24/30] Train Loss: 2.35384, Train Acc: 48.11%, Test Loss: 2.19876, Test Acc: 49.51%, Time: 187.945s
[2023-04-23 06:27:16.568589]: [25/30] Train Loss: 2.31991, Train Acc: 48.74%, Test Loss: 2.24877, Test Acc: 48.83%, Time: 187.495s
[2023-04-23 06:30:24.070229]: [26/30] Train Loss: 2.28825, Train Acc: 49.30%, Test Loss: 2.18719, Test Acc: 50.06%, Time: 187.501s
[2023-04-23 06:33:30.908551]: [27/30] Train Loss: 2.26461, Train Acc: 49.77%, Test Loss: 2.22832, Test Acc: 49.40%, Time: 186.837s
[2023-04-23 06:36:38.881429]: [28/30] Train Loss: 2.24268, Train Acc: 50.19%, Test Loss: 2.16459, Test Acc: 50.22%, Time: 187.972s
[2023-04-23 06:39:46.620478]: [29/30] Train Loss: 2.21247, Train Acc: 50.74%, Test Loss: 2.22659, Test Acc: 49.37%, Time: 187.738s
[2023-04-23 06:42:55.321431]: [30/30] Train Loss: 2.19408, Train Acc: 51.13%, Test Loss: 2.13114, Test Acc: 51.05%, Time: 188.700s
'''




data_lines = data.strip().split('\n')

timestamps = []
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

start_time = None

for line in data_lines:
    parts = re.findall(r'\[.*?\]|\d+\.\d+|\d+%', line)
    timestamp = datetime.datetime.strptime(parts[0][1:-1], "%Y-%m-%d %H:%M:%S.%f")
    
    if start_time is None:
        start_time = timestamp

    time_diff = (timestamp - start_time).total_seconds() / 3600  # convert time difference to hours

    train_loss = float(parts[2])
    train_accuracy = float(parts[3][:-1])
    test_loss = float(parts[4])
    test_accuracy = float(parts[5][:-1])

    timestamps.append(time_diff)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

time_elapsed = [result[-1] for result in zip(timestamps, train_losses, train_accuracies, test_losses, test_accuracies)]
relative_time = np.cumsum(time_elapsed) - time_elapsed[0]

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Loss', color=color)
ax1.plot(relative_time, train_losses, color=color, marker='o', label='Train Loss')
ax1.plot(relative_time, test_losses, color=color, linestyle='--', marker='o', label='Test Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Accuracy (%)', color=color)
ax2.plot(relative_time, train_accuracies, color=color, marker='o', label='Train Acc')
ax2.plot(relative_time, test_accuracies, color=color, linestyle='--', marker='o', label='Test Acc')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

fig.tight_layout()
plt.title('Training and Test Loss and Accuracy over Time')
plt.grid()
plt.show()
