
"""
result from ExampleCode/resnet182.py


20230417 14:53
epoch [1] loss: 53.344
epoch [2] loss: 42.437
epoch [3] loss: 37.967
epoch [4] loss: 34.643
epoch [5] loss: 31.907
epoch [6] loss: 29.465
epoch [7] loss: 27.112
epoch [8] loss: 24.895
epoch [9] loss: 22.819
epoch [10] loss: 20.862
Finished Training
Total training Time: 0:10:50.915716
Accuracy of the network on the test images: 47 %

20230417 
20230417 14:34 result of resnet18
[10,  1200] loss: 1.359
[10,  1300] loss: 1.370
[10,  1400] loss: 1.395
[10,  1500] loss: 1.383
Finished Training
Total training Time: 0:10:28.642483
Accuracy of the network on the test images: 47 %


"""


import matplotlib.pyplot as plt

# Define the loss values
losses = [53.344, 42.437, 37.967, 34.643, 31.907, 29.465, 27.112, 24.895, 22.819, 20.862]

# Plot the loss descent
plt.plot(losses)
plt.title('Loss Descent')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
