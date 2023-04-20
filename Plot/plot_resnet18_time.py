import matplotlib.pyplot as plt

# Epoch data
epoch_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Loss data
loss_data = [53.344, 42.437, 37.967, 34.643, 31.907, 29.465, 27.112, 24.895, 22.819, 20.862]

# Total training time in seconds
total_training_time = 10 * 60 + 50.915716

# Calculate the time elapsed for each epoch
time_elapsed_per_epoch = total_training_time / len(epoch_data)
time_data_sec = [i * time_elapsed_per_epoch for i in range(len(epoch_data))]

# Create a figure and axis
fig, ax = plt.subplots()

# Plot time and loss
ax.plot(time_data_sec, loss_data, 'r-')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Loss')
ax.tick_params('x', colors='r')

# Set the title
plt.title('Loss vs. Time')

# Display the plot
plt.show()

