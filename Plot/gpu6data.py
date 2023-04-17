"""
Code from /MultipleGPU/resnet18.py, using 6 GPU do data parallel training.

(ytorch-venv) superbench@a100-dev-000018:~/v-yiyunchen/net/image_classify_cnn/MultipleGPU$ python3 resnet18.py
ready to train
ready to train
ready to train
ready to train
ready to train
ready to train
/home/superbench/v-yiyunchen/ytorch-venv/lib/python3.6/site-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.)
  return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
/home/superbench/v-yiyunchen/ytorch-venv/lib/python3.6/site-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.)
  return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
/home/superbench/v-yiyunchen/ytorch-venv/lib/python3.6/site-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.)
  return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
/home/superbench/v-yiyunchen/ytorch-venv/lib/python3.6/site-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.)
  return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
/home/superbench/v-yiyunchen/ytorch-venv/lib/python3.6/site-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.)
  return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
/home/superbench/v-yiyunchen/ytorch-venv/lib/python3.6/site-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.)
  return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
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

According to the log provided, the total training time was 10 minutes and 50.915716 seconds.


"""


import matplotlib.pyplot as plt

# Define the loss and accuracy values
train_losses = [5.51293, 4.20730, 3.59563, 3.20150, 2.92008, 2.71976, 2.57233, 2.45695, 2.34120, 2.25621]
train_accs = [4.21, 13.49, 22.20, 28.72, 33.89, 37.66, 40.66, 43.01, 45.19, 47.08]
test_losses = [4.39674, 3.44705, 2.83936, 2.46189, 2.20459, 2.03087, 1.90031, 1.80079, 1.72838, 1.65189]
test_accs = [10.93, 23.51, 33.61, 41.27, 46.65, 49.79, 53.12, 55.41, 56.68, 58.46]

# Plot the loss and accuracy curves
fig, axs = plt.subplots(2, figsize=(10, 8))
axs[0].plot(train_losses, label='train')
axs[0].plot(test_losses, label='test')
axs[0].set_title('Loss Curve')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend()
axs[1].plot(train_accs, label='train')
axs[1].plot(test_accs, label='test')
axs[1].set_title('Accuracy Curve')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy (%)')
axs[1].legend()
plt.show()
