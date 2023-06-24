import re
import matplotlib.pyplot as plt

file_path = 'parallel2.out'

with open(file_path, 'r') as file:
    epoch_data = file.read()

epoch_pattern = re.compile(r'\[(.*?)\]: \[(\d+)/(\d+)\] Train Loss: (.*?), Train Acc: (.*?), Test Loss: (.*?), Test Acc: (.*?), Time: (.*?)s')

times = []
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for match in epoch_pattern.finditer(epoch_data):
    times.append(float(match.group(8)))
    train_losses.append(float(match.group(4)))
    train_accuracies.append(float(match.group(5)[:-1]))
    test_losses.append(float(match.group(6)))
    test_accuracies.append(float(match.group(7)[:-1]))

time_axis = [sum(times[:i+1]) for i in range(len(times))]

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Loss', color=color)
ax1.plot(time_axis, train_losses, color=color, label="Train Loss")
ax1.plot(time_axis, test_losses, color=color, linestyle='dashed', label="Test Loss")
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Accuracy (%)', color=color)
ax2.plot(time_axis, train_accuracies, color=color, label="Train Accuracy")
ax2.plot(time_axis, test_accuracies, color=color, linestyle='dashed', label="Test Accuracy")
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
fig.legend(loc="upper right", bbox_to_anchor=(1.15, 1))

plt.savefig('parallel2.png')
