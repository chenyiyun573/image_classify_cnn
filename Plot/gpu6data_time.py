import re
from datetime import datetime
import matplotlib.pyplot as plt

data = '''
[2023-04-17 07:14:56.757723]: [1/10] Train Loss: 5.51293, Train Acc: 4.21%, Test Loss: 4.39674, Test Acc: 10.93%, Time: 46.068s
[2023-04-17 07:15:41.455380]: [2/10] Train Loss: 4.20730, Train Acc: 13.49%, Test Loss: 3.44705, Test Acc: 23.51%, Time: 44.697s
[2023-04-17 07:16:27.020475]: [3/10] Train Loss: 3.59563, Train Acc: 22.20%, Test Loss: 2.83936, Test Acc: 33.61%, Time: 45.564s
[2023-04-17 07:17:12.060073]: [4/10] Train Loss: 3.20150, Train Acc: 28.72%, Test Loss: 2.46189, Test Acc: 41.27%, Time: 45.039s
[2023-04-17 07:17:57.048542]: [5/10] Train Loss: 2.92008, Train Acc: 33.89%, Test Loss: 2.20459, Test Acc: 46.65%, Time: 44.988s
[2023-04-17 07:18:41.848676]: [6/10] Train Loss: 2.71976, Train Acc: 37.66%, Test Loss: 2.03087, Test Acc: 49.79%, Time: 44.799s
[2023-04-17 07:19:26.702666]: [7/10] Train Loss: 2.57233, Train Acc: 40.66%, Test Loss: 1.90031, Test Acc: 53.12%, Time: 44.853s
[2023-04-17 07:20:11.678977]: [8/10] Train Loss: 2.45695, Train Acc: 43.01%, Test Loss: 1.80079, Test Acc: 55.41%, Time: 44.974s
[2023-04-17 07:20:56.479197]: [9/10] Train Loss: 2.34120, Train Acc: 45.19%, Test Loss: 1.72838, Test Acc: 56.68%, Time: 44.799s
[2023-04-17 07:21:41.151529]: [10/10] Train Loss: 2.25621, Train Acc: 47.08%, Test Loss: 1.65189, Test Acc: 58.46%, Time: 44.672s
'''

time_format = "%Y-%m-%d %H:%M:%S.%f"
pattern = r'\[(.+?)\]: \[(\d+)/\d+\] Train Loss: (.+?), Train Acc: (.+?), Test Loss: (.+?), Test Acc: (.+?), Time: (.+?)s'

# Parse data
timestamps, train_losses, train_accs, test_losses, test_accs, times = [], [], [], [], [], []
for line in data.splitlines():
    match = re.match(pattern, line)
    if match:
        timestamp, epoch, train_loss, train_acc, test_loss, test_acc, time = match.groups()
        timestamps.append(datetime.strptime(timestamp, time_format))
        train_losses.append(float(train_loss))
        train_accs.append(float(train_acc.strip('%')))
        test_losses.append(float(test_loss))
        test_accs.append(float(test_acc.strip('%')))
        times.append(float(time))

# Compute cumulative time
cumulative_time = [sum(times[:i+1]) for i in range(len(times))]

# Plot
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(cumulative_time, train_losses, 'r-', label='Train Loss')
ax1.plot(cumulative_time, test_losses, 'r--', label='Test Loss')
ax2.plot(cumulative_time, train_accs, 'b-', label='Train Acc')
ax2.plot(cumulative_time, test_accs, 'b--', label='Test Acc')

ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Loss')
ax2.set_ylabel('Accuracy (%)')

ax1.legend(loc='upper left')
ax2.legend(loc='lower right')

plt.title('Loss and Accuracy vs. Time')
plt.show()
