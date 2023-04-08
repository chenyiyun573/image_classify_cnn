## Usage
run train.py. It will check and download MINIST dataset to folder tmp

The code of train.py works well. - 2023.3.28 Yiyun

Another train_tensorboard.py is the same code with train.py but with tensorboard to record and analyze.

One error of tensorboard may happen like AttributeError: module 'distutils' has no attribute 'version', solved by:
https://github.com/pytorch/pytorch/issues/69894
I did it by pip3 install setuptools==58.2.0

The code of train_tensorboard.py works. The Function of summaryWriter is not tested. - 2023.4.8 Yiyun

