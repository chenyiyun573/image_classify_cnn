This folder is code to train network on one single GPU.




Command to train here, here is 8 GPU on one VM:
```
mkdir -p logs

nohup python3 train_single.py --gpu 0 > ./logs/alexnet.log 2>&1 &
nohup python3 train_single.py --gpu 1 > ./logs/vgg19.log 2>&1 &
nohup python3 train_single.py --gpu 2 > ./logs/resnet18.log 2>&1 &
nohup python3 train_single.py --gpu 3 > ./logs/resnet152.log 2>&1 &
nohup python3 train_single.py --gpu 4 > ./logs/densenet201.log 2>&1 &
nohup python3 train_single.py --gpu 5 > ./logs/inception_v3.log 2>&1 &
nohup python3 train_single.py --gpu 6 > ./logs/googlenet.log 2>&1 &
nohup python3 train_single.py --gpu 7 > ./logs/squeezenet1_1.log 2>&1 &

```


Memory Storage:
3: 10616MiB / 40960MiB 

|    0   N/A  N/A     38884      C   python3                          3374MiB |
|    2   N/A  N/A     41004      C   python3                          3376MiB |
|    3   N/A  N/A     43326      C   python3                         10614MiB |

After trying, only alexnet, resnet, googlenet have no errors happened, train the non-pretrained results:
nohup python3 train_single.py --gpu 0 > ./logs/alexnet_nopre.log 2>&1 &
nohup python3 train_single.py --gpu 2 > ./logs/resnet18_nopre.log 2>&1 &
nohup python3 train_single.py --gpu 3 > ./logs/resnet152_nopre.log 2>&1 &
nohup python3 train_single.py --gpu 6 > ./logs/googlenet_nopre.log 2>&1 &
