20230417 14:34 result of resnet18
[10,  1200] loss: 1.359
[10,  1300] loss: 1.370
[10,  1400] loss: 1.395
[10,  1500] loss: 1.383
Finished Training
Total training Time: 0:10:28.642483
Accuracy of the network on the test images: 47 %





202304131627 I do not know why model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=200) is trained so slow. 
So, I replace it with models 
```
model = resnet18(pretrained=True)  
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 200)
```
However, it was the same, also very slow compared to hierarchy training. 
For the reason, the main difference is the dataset, in hierarchy training, deeper levels have much less images to iterate 
Compared to the root level.
The following is result from resnet18.py
```
Current Time: 2023-04-13 09:10:24.239934
/home/superbench/v-yiyunchen/ytorch-venv/lib/python3.6/site-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.)
  return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
[1,   100] loss: 4.545
[1,   200] loss: 3.667
......
[10,  1300] loss: 0.508
[10,  1400] loss: 0.528
[10,  1500] loss: 0.515
Finished Training
Current Time: 2023-04-13 09:59:58.090693
Accuracy of the network on the test images: 57 %
```




For tiny.py, it use a simple net work to classify images in dataset tiny-imagenet-200.
Before training:
wget ...(url of tiny-imagenet-200)
unzip filename
python3 reformat.py (the folder tree of tiny-imagenet-200 is different with the datasets.ImageFolder's requirements)


tinyimagenet.py is the same like tiny.py, but it use a pre-trained network.
The code of tinyimagenet.py works well.
[10,  1400] loss: 0.431
[10,  1500] loss: 0.439
Finished Training
Accuracy of the network on the test images: 67 %  - 2023.4.8 Yiyun
