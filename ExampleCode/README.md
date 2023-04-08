For tiny.py, it use a simple net work to classify images in dataset tiny-imagenet-200.
Before training:
wget ...(url of tiny-imagenet-200)
unzip filename
python3 reformat.py (the folder tree of tiny-imagenet-200 is different with the datasets.ImageFolder's requirements)




tinyimagenet.py is the same like tiny.py, but it use a pre-trained network.

The code of tinyimagenet.py works well.i
[10,  1400] loss: 0.431
[10,  1500] loss: 0.439
Finished Training
Accuracy of the network on the test images: 67 %  - 2023.4.8 Yiyun
