import matplotlib.pyplot as plt
import numpy as np

# Results (timestamp, train_loss, train_acc, test_loss, test_acc, time)
results = [
    ("2023-04-21 06:31:33.069476", 1.32086, 68.90, 1.25169, 69.28, 187.379),
    ("2023-04-21 06:34:40.119891", 1.30678, 69.16, 1.24863, 69.39, 187.050),
    ("2023-04-21 06:37:45.850284", 1.30098, 69.30, 1.24585, 69.40, 185.730),
    ("2023-04-21 06:40:52.480210", 1.29448, 69.43, 1.24430, 69.43, 186.629),
    ("2023-04-21 06:44:00.705285", 1.29364, 69.42, 1.24340, 69.40, 188.224),
    ("2023-04-21 06:47:07.511050", 1.29116, 69.50, 1.24344, 69.42, 186.805),
    ("2023-04-21 06:50:14.216963", 1.29405, 69.43, 1.24232, 69.41, 186.705),
    ("2023-04-21 06:53:22.106469", 1.28915, 69.59, 1.24249, 69.45, 187.889),
    ("2023-04-21 06:56:30.308222", 1.28885, 69.56, 1.24110, 69.47, 188.201),
    ("2023-04-21 06:59:37.987657", 1.28531, 69.62, 1.23785, 69.55, 187.679)
]

time_elapsed = [result[-1] for result in results]
relative_time = np.cumsum(time_elapsed) - time_elapsed[0]

train_acc = [result[2] for result in results]
test_acc = [result[4] for result in results]

plt.plot(relative_time, train_acc, label='Train Acc', marker='o')
plt.plot(relative_time, test_acc, label='Test Acc', marker='o')
plt.xlabel('Time (s)')
plt.ylabel('Accuracy (%)')
plt.title('Training and Test Accuracy over Time')
plt.legend()
plt.grid()
plt.show()
