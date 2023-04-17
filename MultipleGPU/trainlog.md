(ytorch-venv) superbench@a100-dev-000018:~/v-yiyunchen/net/image_classify_cnn/MultipleGPU$ python3 tinyimagenet.py
Loaded pretrained weights for efficientnet-b0
Loaded pretrained weights for efficientnet-b0
ready to train
ready to train
[2023-04-08 22:46:37.993506]: [1/10] Train Loss: 5.05572, Train Acc: 8.25%, Test Loss: 4.40450, Test Acc: 32.16%, Time: 131.239s
[2023-04-08 22:48:48.938963]: [2/10] Train Loss: 3.90393, Train Acc: 27.59%, Test Loss: 2.53901, Test Acc: 50.96%, Time: 130.944s
[2023-04-08 22:51:00.699374]: [3/10] Train Loss: 2.90500, Train Acc: 39.31%, Test Loss: 1.71800, Test Acc: 62.42%, Time: 131.759s
[2023-04-08 22:53:13.497321]: [4/10] Train Loss: 2.42606, Train Acc: 46.41%, Test Loss: 1.37213, Test Acc: 68.00%, Time: 132.797s
[2023-04-08 22:55:26.372396]: [5/10] Train Loss: 2.19147, Train Acc: 49.96%, Test Loss: 1.18157, Test Acc: 71.22%, Time: 132.874s
[2023-04-08 22:57:38.709312]: [6/10] Train Loss: 2.03754, Train Acc: 52.80%, Test Loss: 1.08672, Test Acc: 72.47%, Time: 132.336s
[2023-04-08 22:59:51.409074]: [7/10] Train Loss: 1.92936, Train Acc: 54.76%, Test Loss: 1.02201, Test Acc: 73.75%, Time: 132.699s
[2023-04-08 23:02:04.010574]: [8/10] Train Loss: 1.84845, Train Acc: 56.41%, Test Loss: 0.96657, Test Acc: 74.95%, Time: 132.600s
[2023-04-08 23:04:15.279293]: [9/10] Train Loss: 1.78292, Train Acc: 57.65%, Test Loss: 0.91604, Test Acc: 76.31%, Time: 131.268s
[2023-04-08 23:06:25.833949]: [10/10] Train Loss: 1.73460, Train Acc: 58.57%, Test Loss: 0.89032, Test Acc: 76.70%, Time: 130.553s
So, we can see that for the result above I use two A100 GPUs

(ytorch-venv) superbench@a100-dev-000018:~/v-yiyunchen/net/image_classify_cnn/MultipleGPU$ python3 tinyimagenet.py
Loaded pretrained weights for efficientnet-b0
Loaded pretrained weights for efficientnet-b0
Loaded pretrained weights for efficientnet-b0
Loaded pretrained weights for efficientnet-b0
ready to train
ready to train
ready to train
ready to train
[2023-04-09 08:40:01.902032]: [1/10] Train Loss: 5.21937, Train Acc: 2.68%, Test Loss: 5.04582, Test Acc: 13.68%, Time: 82.290s
[2023-04-09 08:41:23.927289]: [2/10] Train Loss: 4.88305, Train Acc: 14.43%, Test Loss: 4.38720, Test Acc: 33.75%, Time: 82.024s
[2023-04-09 08:42:44.989187]: [3/10] Train Loss: 4.23936, Train Acc: 24.46%, Test Loss: 3.28395, Test Acc: 43.28%, Time: 81.061s
[2023-04-09 08:44:06.354584]: [4/10] Train Loss: 3.53853, Train Acc: 31.59%, Test Loss: 2.47974, Test Acc: 51.71%, Time: 81.364s
[2023-04-09 08:45:27.680884]: [5/10] Train Loss: 3.04064, Train Acc: 37.88%, Test Loss: 1.98981, Test Acc: 59.45%, Time: 81.325s
[2023-04-09 08:46:49.192426]: [6/10] Train Loss: 2.70680, Train Acc: 42.50%, Test Loss: 1.67718, Test Acc: 63.30%, Time: 81.510s
[2023-04-09 08:48:11.358235]: [7/10] Train Loss: 2.48578, Train Acc: 45.56%, Test Loss: 1.48762, Test Acc: 65.99%, Time: 82.165s
[2023-04-09 08:49:32.746154]: [8/10] Train Loss: 2.32585, Train Acc: 48.01%, Test Loss: 1.35429, Test Acc: 67.73%, Time: 81.386s
[2023-04-09 08:50:53.695363]: [9/10] Train Loss: 2.21539, Train Acc: 49.63%, Test Loss: 1.25517, Test Acc: 69.74%, Time: 80.948s
[2023-04-09 08:52:14.972836]: [10/10] Train Loss: 2.12778, Train Acc: 51.25%, Test Loss: 1.18912, Test Acc: 70.65%, Time: 81.276s

(ytorch-venv) superbench@a100-dev-000018:~/v-yiyunchen/net/image_classify_cnn/MultipleGPU$ python3 tinyimagenet.py
Loaded pretrained weights for efficientnet-b0
Loaded pretrained weights for efficientnet-b0
Loaded pretrained weights for efficientnet-b0
Loaded pretrained weights for efficientnet-b0
Loaded pretrained weights for efficientnet-b0
Loaded pretrained weights for efficientnet-b0
ready to train
ready to train
ready to train
ready to train
ready to train
ready to train
[2023-04-09 08:58:25.470374]: [1/10] Train Loss: 5.25606, Train Acc: 1.43%, Test Loss: 5.15314, Test Acc: 6.22%, Time: 66.949s
[2023-04-09 08:59:32.048925]: [2/10] Train Loss: 5.08179, Train Acc: 7.92%, Test Loss: 4.87575, Test Acc: 22.11%, Time: 66.577s
[2023-04-09 09:00:38.255661]: [3/10] Train Loss: 4.78498, Train Acc: 16.90%, Test Loss: 4.35689, Test Acc: 33.99%, Time: 66.206s
[2023-04-09 09:01:44.914819]: [4/10] Train Loss: 4.33790, Train Acc: 23.40%, Test Loss: 3.65393, Test Acc: 40.45%, Time: 66.658s
[2023-04-09 09:02:51.875995]: [5/10] Train Loss: 3.85062, Train Acc: 28.52%, Test Loss: 2.97475, Test Acc: 46.40%, Time: 66.960s
[2023-04-09 09:03:58.881403]: [6/10] Train Loss: 3.42261, Train Acc: 33.11%, Test Loss: 2.47844, Test Acc: 51.94%, Time: 67.004s
[2023-04-09 09:05:05.448598]: [7/10] Train Loss: 3.10563, Train Acc: 36.79%, Test Loss: 2.13607, Test Acc: 56.60%, Time: 66.566s
[2023-04-09 09:06:11.894606]: [8/10] Train Loss: 2.84873, Train Acc: 40.60%, Test Loss: 1.87495, Test Acc: 60.55%, Time: 66.444s
[2023-04-09 09:07:18.776963]: [9/10] Train Loss: 2.66385, Train Acc: 43.09%, Test Loss: 1.68900, Test Acc: 62.79%, Time: 66.881s
[2023-04-09 09:08:25.805589]: [10/10] Train Loss: 2.50976, Train Acc: 45.33%, Test Loss: 1.54467, Test Acc: 65.10%, Time: 67.027s


I have confusion about lower convergence speed, the following is answers by chatGPT - YiyunChen.
It is possible that your idea is reasonable. When using data parallelism to train a neural network across multiple GPUs, the gradients calculated by each GPU need to be combined in order to update the model parameters. This can be done using various techniques such as averaging, summation, or gradient accumulation.

However, combining gradients from multiple GPUs can introduce additional noise and variance, which can slow down the convergence of the optimization process. Additionally, if the model architecture or hyperparameters are not well-suited for parallel training, then the benefits of data parallelism may be limited.

It is also worth noting that improving training speed does not necessarily guarantee faster convergence to a certain test accuracy. The test accuracy depends on many factors such as the complexity of the task, quality and quantity of data, and the model architecture and hyperparameters.

Therefore, it is important to carefully monitor the training process and experiment with different model architectures, hyperparameters, and parallel training techniques to determine the most effective approach for your specific problem.

