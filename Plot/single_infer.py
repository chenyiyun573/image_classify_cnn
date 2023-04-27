import re
import datetime
import matplotlib.pyplot as plt
import numpy as np

filename = 'single_infer.out'

# Regular expressions to extract data
network_pattern = re.compile(r'(Network (?:Root|Network \d+(?:-\d+)?))')
accuracy_pattern = re.compile(r'accuracy ([\d.]+)')
time_pattern = re.compile(r'time (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6})')
total_pattern = re.compile(r'total (\d+)')

network_info = {}

with open(filename, 'r') as file:
    lines = file.readlines()

    for line in lines:
        network_match = network_pattern.search(line)
        accuracy_match = accuracy_pattern.search(line)
        time_match = time_pattern.search(line)
        total_match = total_pattern.search(line)

        if network_match and accuracy_match and time_match and total_match:
            network = network_match.group(1)
            accuracy = float(accuracy_match.group(1))
            time = datetime.datetime.strptime(time_match.group(1), "%Y-%m-%d %H:%M:%S.%f")
            total = int(total_match.group(1))

            if network not in network_info:
                network_info[network] = {'start_time': time, 'end_time': time, 'accuracy': accuracy, 'total': total}
            else:
                network_info[network]['end_time'] = time
                network_info[network]['accuracy'] = accuracy
                network_info[network]['total'] = total

network_names = []
image_numbers = []
times = []
accuracies = []

for network, info in network_info.items():
    inference_time = (info['end_time'] - info['start_time']).total_seconds()
    network_names.append(network)
    image_numbers.append(info['total'])
    times.append(inference_time)
    accuracies.append(info['accuracy'])

# Plotting
bar_width = 0.25
spacing = 0.1
index = np.arange(len(network_names))

fig, ax1 = plt.subplots()

bar1 = ax1.bar(index - bar_width - spacing / 2, image_numbers, bar_width, label='Image numbers', color='b')
ax1.set_xlabel('Networks')
ax1.set_ylabel('Image numbers', color='b')
ax1.tick_params(axis='y', labelcolor='b')

ax2 = ax1.twinx()
bar2 = ax2.bar(index, times, bar_width, label='Inference time (s)', color='g')
ax2.set_ylabel('Inference time (s)', color='g')
ax2.tick_params(axis='y', labelcolor='g')

ax3 = ax1.twinx()
bar3 = ax3.bar(index + bar_width + spacing / 2, accuracies, bar_width, label='Accuracy (%)', color='r')
ax3.set_ylabel('Accuracy (%)', color='r')
ax3.tick_params(axis='y', labelcolor='r')

# Move the last y-axis to the right
ax3.spines['right'].set_position(('axes', 1.15))

ax1.set_xticks(index)
ax1.set_xticklabels(network_names, rotation=45)

fig.legend([bar1, bar2, bar3], ['Image numbers', 'Inference time (s)', 'Accuracy (%)'], loc='upper center', ncol=3)

plt.tight_layout()
plt.show()
