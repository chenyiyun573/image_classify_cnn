import os
import re
import matplotlib.pyplot as plt

log_files1 = {
    'alexnet': './logs/alexnet_nopre.log',
    'resnet18': './logs/resnet18_nopre.log',
    'resnet152': './logs/resnet152_nopre.log',
    #'googlenet': './logs/googlenet.log'
}

log_files = {
    'alexnet': './logs/alexnet.log',
    'resnet18': './logs/resnet18.log',
    'resnet152': './logs/resnet152.log',
    #'googlenet': './logs/googlenet.log'
}

def parse_log_file(log_file):
    train_accuracies = []
    test_accuracies = []
    times = []
    with open(log_file, 'r') as f:
        for line in f:
            if "Time" in line:
                time = float(re.findall(r"Time: (\d+\.\d+)s", line)[0])
                times.append(time)
            if "Train Accuracy" in line:
                train_acc = float(re.findall(r"Train Accuracy: (\d+\.\d+)%", line)[0])
                train_accuracies.append(train_acc)
            if "Test Accuracy" in line:
                test_acc = float(re.findall(r"Test Accuracy: (\d+\.\d+)%", line)[0])
                test_accuracies.append(test_acc)
    return times, train_accuracies, test_accuracies

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Training and Test Accuracy')

for model, log_file in log_files.items():
    times, train_acc, test_acc = parse_log_file(log_file)
    ax1.plot(times, train_acc, label=model)
    ax2.plot(times, test_acc, label=model)

ax1.set_title('Training Accuracy')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.set_title('Test Accuracy')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.savefig('training_test_accuracy.png', bbox_inches='tight')
