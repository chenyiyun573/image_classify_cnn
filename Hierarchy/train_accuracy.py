import torch
import json
from datetime import datetime
from torchvision import models, datasets, transforms

model_load_path = './models'
trained_models = {}

with open('output.json', 'r') as f:
    sub_networks = json.load(f)


root_mapping = {'Others': 0, 'Objects': 1, 'Animals': 2 }
# Transformations for Tiny ImageNet
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(64, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = datasets.ImageFolder(root='/home/superbench/v-yiyunchen/net/re-tiny-imagenet-200/train', transform=transform)


all_classes = [class_id for class_id_list in sub_networks.values() for class_id in class_id_list]
def filter_trainset(trainset, all_classes):
    filtered_indices = [idx for idx, (_, target) in enumerate(trainset) if trainset.classes[target] in all_classes]
    filtered_trainset = torch.utils.data.Subset(trainset, filtered_indices)
    return filtered_trainset
filtered_trainset = filter_trainset(trainset, all_classes)



sub_networks.update({'Root': ['Animals', 'Objects', 'Others'],})

for network in sub_networks.keys():
    model = models.resnet18(pretrained=True)
    num_classes = len(sub_networks[network])
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(torch.load(f"{model_load_path}/{network}_model.pth"))
    model.eval()
    trained_models[network] = model


def get_final_predicted_label(image, root_model, sub_models, device):
    image = image.unsqueeze(0).to(device)
    root_output = root_model(image)
    _, root_predicted = torch.max(root_output.data, 1)
    root_class = list(root_mapping.keys())[list(root_mapping.values()).index(root_predicted.item())]

    sub_model = sub_models[root_class]
    sub_output = sub_model(image)
    _, sub_predicted = torch.max(sub_output.data, 1)

    sub_classes = sub_networks[root_class]
    final_label = sub_classes[sub_predicted.item()]

    return final_label



def get_hierarchical_accuracy(root_model, sub_models, filtered_trainset, device):
    correct = 0
    total = 0
    for i in range(len(filtered_trainset)):
        image, target = filtered_trainset[i]
        target_label = trainset.classes[target]
        predicted_label = get_final_predicted_label(image, root_model, sub_models, device)

        total += 1
        if target_label == predicted_label:
            correct += 1
        if total%1000 == 0:
            print(f"total {total} correct {correct} accuracy {correct/total* 100:.2f} time {datetime.now()}")

    return correct / total



root_model = trained_models['Root']
sub_models = {k: v for k, v in trained_models.items() if k != 'Root'}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root_model.to(device)
for sub_model in sub_models.values():
    sub_model.to(device)



hierarchical_accuracy = get_hierarchical_accuracy(root_model, sub_models, filtered_trainset, device)
print(f"Hierarchical trainset accuracy: {hierarchical_accuracy * 100:.2f}%  time {datetime.now()}")



"""
following data is using models trained by train4.py:
The time it takes to inference all images is not explicitly mentioned in the given log. 
However, we can see that the total number of images processed is 100,000, and the time stamp at the end of the log 
is 2023-04-17 09:24:22.173204. 
This indicates that the total time taken to process all images is approximately 10 minutes and 10 seconds 
(assuming that the processing started immediately after the first timestamp in the log).




  return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
total 1000 correct 870 accuracy 87.00 time 2023-04-17 09:14:12.397153
total 2000 correct 1609 accuracy 80.45 time 2023-04-17 09:14:18.565430
total 3000 correct 2316 accuracy 77.20 time 2023-04-17 09:14:24.734808
total 4000 correct 3109 accuracy 77.72 time 2023-04-17 09:14:30.897848
total 5000 correct 3912 accuracy 78.24 time 2023-04-17 09:14:37.063689
total 6000 correct 4683 accuracy 78.05 time 2023-04-17 09:14:43.229299
total 7000 correct 5557 accuracy 79.39 time 2023-04-17 09:14:49.396703
total 8000 correct 6367 accuracy 79.59 time 2023-04-17 09:14:55.567909
total 9000 correct 7090 accuracy 78.78 time 2023-04-17 09:15:01.738590
total 10000 correct 7811 accuracy 78.11 time 2023-04-17 09:15:07.910698
total 11000 correct 8671 accuracy 78.83 time 2023-04-17 09:15:14.076413
total 12000 correct 9551 accuracy 79.59 time 2023-04-17 09:15:20.234644
total 13000 correct 10286 accuracy 79.12 time 2023-04-17 09:15:26.398964
total 14000 correct 11004 accuracy 78.60 time 2023-04-17 09:15:32.560079
total 15000 correct 11743 accuracy 78.29 time 2023-04-17 09:15:38.724580
total 16000 correct 12514 accuracy 78.21 time 2023-04-17 09:15:44.886830
total 17000 correct 13175 accuracy 77.50 time 2023-04-17 09:15:51.048610
total 18000 correct 13976 accuracy 77.64 time 2023-04-17 09:15:57.215107
total 19000 correct 14770 accuracy 77.74 time 2023-04-17 09:16:03.374272
total 20000 correct 15558 accuracy 77.79 time 2023-04-17 09:16:09.536857
total 21000 correct 16304 accuracy 77.64 time 2023-04-17 09:16:15.697902
total 22000 correct 17022 accuracy 77.37 time 2023-04-17 09:16:21.859509
total 23000 correct 17955 accuracy 78.07 time 2023-04-17 09:16:28.024488
total 24000 correct 18725 accuracy 78.02 time 2023-04-17 09:16:34.187673
total 25000 correct 19370 accuracy 77.48 time 2023-04-17 09:16:40.351753
total 26000 correct 20161 accuracy 77.54 time 2023-04-17 09:16:46.547186
total 27000 correct 21008 accuracy 77.81 time 2023-04-17 09:16:52.715800
total 28000 correct 21826 accuracy 77.95 time 2023-04-17 09:16:58.884665
total 29000 correct 22627 accuracy 78.02 time 2023-04-17 09:17:05.054045
total 30000 correct 23437 accuracy 78.12 time 2023-04-17 09:17:11.224353
total 31000 correct 24210 accuracy 78.10 time 2023-04-17 09:17:17.391327
total 32000 correct 24884 accuracy 77.76 time 2023-04-17 09:17:23.558666
total 33000 correct 25481 accuracy 77.22 time 2023-04-17 09:17:29.722956
total 34000 correct 26219 accuracy 77.11 time 2023-04-17 09:17:35.887003
total 35000 correct 26937 accuracy 76.96 time 2023-04-17 09:17:42.050611
total 36000 correct 27707 accuracy 76.96 time 2023-04-17 09:17:48.211251
total 37000 correct 28413 accuracy 76.79 time 2023-04-17 09:17:54.370497
total 38000 correct 29106 accuracy 76.59 time 2023-04-17 09:18:00.525994
total 39000 correct 29716 accuracy 76.19 time 2023-04-17 09:18:06.677368
total 40000 correct 30421 accuracy 76.05 time 2023-04-17 09:18:12.834843
total 41000 correct 31180 accuracy 76.05 time 2023-04-17 09:18:18.987684
total 42000 correct 31917 accuracy 75.99 time 2023-04-17 09:18:25.142706
total 43000 correct 32596 accuracy 75.80 time 2023-04-17 09:18:31.297310
total 44000 correct 33318 accuracy 75.72 time 2023-04-17 09:18:37.449173
total 45000 correct 34021 accuracy 75.60 time 2023-04-17 09:18:43.600620
total 46000 correct 34838 accuracy 75.73 time 2023-04-17 09:18:49.758084
total 47000 correct 35587 accuracy 75.72 time 2023-04-17 09:18:55.916715
total 48000 correct 36299 accuracy 75.62 time 2023-04-17 09:19:02.068284
total 49000 correct 37031 accuracy 75.57 time 2023-04-17 09:19:08.222744
total 50000 correct 37669 accuracy 75.34 time 2023-04-17 09:19:14.373865
total 51000 correct 38330 accuracy 75.16 time 2023-04-17 09:19:20.524219
total 52000 correct 39074 accuracy 75.14 time 2023-04-17 09:19:26.676127
total 53000 correct 39707 accuracy 74.92 time 2023-04-17 09:19:32.831007
total 54000 correct 40386 accuracy 74.79 time 2023-04-17 09:19:38.987342
total 55000 correct 41167 accuracy 74.85 time 2023-04-17 09:19:45.138874
total 56000 correct 41902 accuracy 74.83 time 2023-04-17 09:19:51.293282
total 57000 correct 42611 accuracy 74.76 time 2023-04-17 09:19:57.444871
total 58000 correct 43394 accuracy 74.82 time 2023-04-17 09:20:03.598534
total 59000 correct 44143 accuracy 74.82 time 2023-04-17 09:20:09.750041
total 60000 correct 44844 accuracy 74.74 time 2023-04-17 09:20:15.906571
total 61000 correct 45487 accuracy 74.57 time 2023-04-17 09:20:22.064543
total 62000 correct 46091 accuracy 74.34 time 2023-04-17 09:20:28.217915
total 63000 correct 46863 accuracy 74.39 time 2023-04-17 09:20:34.368043
total 64000 correct 47591 accuracy 74.36 time 2023-04-17 09:20:40.524702
total 65000 correct 48309 accuracy 74.32 time 2023-04-17 09:20:46.680035
total 66000 correct 48865 accuracy 74.04 time 2023-04-17 09:20:52.830996
total 67000 correct 49506 accuracy 73.89 time 2023-04-17 09:20:58.986028
total 68000 correct 50196 accuracy 73.82 time 2023-04-17 09:21:05.143057
total 69000 correct 50856 accuracy 73.70 time 2023-04-17 09:21:11.294770
total 70000 correct 51463 accuracy 73.52 time 2023-04-17 09:21:17.446443
total 71000 correct 52215 accuracy 73.54 time 2023-04-17 09:21:23.598183
total 72000 correct 52978 accuracy 73.58 time 2023-04-17 09:21:29.753941
total 73000 correct 53745 accuracy 73.62 time 2023-04-17 09:21:35.909180
total 74000 correct 54449 accuracy 73.58 time 2023-04-17 09:21:42.064506
total 75000 correct 55185 accuracy 73.58 time 2023-04-17 09:21:48.217199
total 76000 correct 55875 accuracy 73.52 time 2023-04-17 09:21:54.373752
total 77000 correct 56556 accuracy 73.45 time 2023-04-17 09:22:00.529187
total 78000 correct 57332 accuracy 73.50 time 2023-04-17 09:22:06.684052
total 79000 correct 57933 accuracy 73.33 time 2023-04-17 09:22:12.834505
total 80000 correct 58490 accuracy 73.11 time 2023-04-17 09:22:18.983163
total 81000 correct 59213 accuracy 73.10 time 2023-04-17 09:22:25.147376
total 82000 correct 59943 accuracy 73.10 time 2023-04-17 09:22:31.310843
total 83000 correct 60709 accuracy 73.14 time 2023-04-17 09:22:37.476450
total 84000 correct 61470 accuracy 73.18 time 2023-04-17 09:22:43.637354
total 85000 correct 62045 accuracy 72.99 time 2023-04-17 09:22:49.797214
total 86000 correct 62791 accuracy 73.01 time 2023-04-17 09:22:55.956534
total 87000 correct 63459 accuracy 72.94 time 2023-04-17 09:23:02.118632
total 88000 correct 64033 accuracy 72.76 time 2023-04-17 09:23:08.275421
total 89000 correct 64696 accuracy 72.69 time 2023-04-17 09:23:14.440370
total 90000 correct 65400 accuracy 72.67 time 2023-04-17 09:23:20.600057
total 91000 correct 66066 accuracy 72.60 time 2023-04-17 09:23:26.758473
total 92000 correct 66843 accuracy 72.66 time 2023-04-17 09:23:32.916469
total 93000 correct 67631 accuracy 72.72 time 2023-04-17 09:23:39.076499
total 94000 correct 68499 accuracy 72.87 time 2023-04-17 09:23:45.228042
total 95000 correct 69353 accuracy 73.00 time 2023-04-17 09:23:51.383592
total 96000 correct 70151 accuracy 73.07 time 2023-04-17 09:23:57.546301
total 97000 correct 71018 accuracy 73.21 time 2023-04-17 09:24:03.702490
total 98000 correct 71774 accuracy 73.24 time 2023-04-17 09:24:09.859420
total 99000 correct 72420 accuracy 73.15 time 2023-04-17 09:24:16.017847
total 100000 correct 73135 accuracy 73.13 time 2023-04-17 09:24:22.173119
Hierarchical trainset accuracy: 73.13%  time 2023-04-17 09:24:22.173204


"""