
20230420 17:39 parallel.py change batch size to 64, use 8 GPUs
[2023-04-20 09:44:36.188133]: [1/10] Train Loss: 1.36981, Train Acc: 67.87%, Test Loss: 16.47146, Test Acc: 0.10%, Time: 186.852s
[2023-04-20 09:47:43.410603]: [2/10] Train Loss: 1.35788, Train Acc: 68.11%, Test Loss: 16.60953, Test Acc: 0.10%, Time: 187.222s
[2023-04-20 09:50:50.576854]: [3/10] Train Loss: 1.35353, Train Acc: 68.22%, Test Loss: 16.66453, Test Acc: 0.10%, Time: 187.166s
[2023-04-20 09:53:57.105793]: [4/10] Train Loss: 1.34545, Train Acc: 68.33%, Test Loss: 16.61565, Test Acc: 0.10%, Time: 186.528s
[2023-04-20 09:57:04.098655]: [5/10] Train Loss: 1.34322, Train Acc: 68.36%, Test Loss: 16.73233, Test Acc: 0.10%, Time: 186.992s
[2023-04-20 10:00:11.527071]: [6/10] Train Loss: 1.33849, Train Acc: 68.48%, Test Loss: 16.78511, Test Acc: 0.10%, Time: 187.428s
[2023-04-20 10:03:18.917718]: [7/10] Train Loss: 1.33878, Train Acc: 68.49%, Test Loss: 16.80750, Test Acc: 0.10%, Time: 187.390s
[2023-04-20 10:06:25.652137]: [8/10] Train Loss: 1.33734, Train Acc: 68.47%, Test Loss: 16.90336, Test Acc: 0.11%, Time: 186.734s
[2023-04-20 10:09:32.704432]: [9/10] Train Loss: 1.32972, Train Acc: 68.64%, Test Loss: 16.89867, Test Acc: 0.11%, Time: 187.052s
[2023-04-20 10:12:38.577040]: [10/10] Train Loss: 1.33007, Train Acc: 68.65%, Test Loss: 16.94833, Test Acc: 0.10%, Time: 185.872s
(ytorch-venv) superbench@a100-dev-000018:~/v-yiyunchen$ nvidia-smi
Thu Apr 20 09:43:56 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 520.61.05    Driver Version: 520.61.05    CUDA Version: 11.8     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM...  On   | 00000001:00:00.0 Off |                    0 |
| N/A   52C    P0   145W / 400W |   4819MiB / 40960MiB |     80%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA A100-SXM...  On   | 00000002:00:00.0 Off |                    0 |
| N/A   46C    P0   270W / 400W |   4963MiB / 40960MiB |     89%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   2  NVIDIA A100-SXM...  On   | 00000003:00:00.0 Off |                    0 |
| N/A   51C    P0   264W / 400W |   4963MiB / 40960MiB |     80%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   3  NVIDIA A100-SXM...  On   | 00000004:00:00.0 Off |                    0 |
| N/A   48C    P0   161W / 400W |   4963MiB / 40960MiB |     67%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   4  NVIDIA A100-SXM...  On   | 0000000B:00:00.0 Off |                    0 |
| N/A   48C    P0   238W / 400W |   4963MiB / 40960MiB |     97%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   5  NVIDIA A100-SXM...  On   | 0000000C:00:00.0 Off |                    0 |
| N/A   45C    P0   193W / 400W |   4963MiB / 40960MiB |     84%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   6  NVIDIA A100-SXM...  On   | 0000000D:00:00.0 Off |                    0 |
| N/A   50C    P0   284W / 400W |   4963MiB / 40960MiB |     79%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   7  NVIDIA A100-SXM...  On   | 0000000E:00:00.0 Off |                    0 |
| N/A   47C    P0   283W / 400W |   4819MiB / 40960MiB |     95%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     56984      C   ...n/ytorch-venv/bin/python3     4814MiB |
|    1   N/A  N/A     56985      C   ...n/ytorch-venv/bin/python3     4958MiB |
|    2   N/A  N/A     56986      C   ...n/ytorch-venv/bin/python3     4958MiB |
|    3   N/A  N/A     56998      C   ...n/ytorch-venv/bin/python3     4958MiB |
|    4   N/A  N/A     56999      C   ...n/ytorch-venv/bin/python3     4958MiB |
|    5   N/A  N/A     57000      C   ...n/ytorch-venv/bin/python3     4958MiB |
|    6   N/A  N/A     57001      C   ...n/ytorch-venv/bin/python3     4958MiB |
|    7   N/A  N/A     57002      C   ...n/ytorch-venv/bin/python3     4814MiB |
+-----------------------------------------------------------------------------+



20230420 17:37 parallel.py batch size 512, use 6 GPUs
[2023-04-20 09:10:38.656772]: [1/10] Train Loss: 1.31866, Train Acc: 68.92%, Test Loss: 16.52490, Test Acc: 0.11%, Time: 186.634s
[2023-04-20 09:13:41.357226]: [2/10] Train Loss: 1.30438, Train Acc: 69.21%, Test Loss: 16.46877, Test Acc: 0.10%, Time: 182.700s
[2023-04-20 09:16:43.724247]: [3/10] Train Loss: 1.30033, Train Acc: 69.32%, Test Loss: 16.52245, Test Acc: 0.11%, Time: 182.366s
[2023-04-20 09:19:45.988627]: [4/10] Train Loss: 1.29365, Train Acc: 69.41%, Test Loss: 16.53922, Test Acc: 0.11%, Time: 182.262s
[2023-04-20 09:22:49.202136]: [5/10] Train Loss: 1.29167, Train Acc: 69.44%, Test Loss: 16.53406, Test Acc: 0.11%, Time: 183.213s
[2023-04-20 09:25:51.897742]: [6/10] Train Loss: 1.29176, Train Acc: 69.49%, Test Loss: 16.58399, Test Acc: 0.11%, Time: 182.695s
[2023-04-20 09:28:53.572527]: [7/10] Train Loss: 1.29052, Train Acc: 69.53%, Test Loss: 16.52749, Test Acc: 0.11%, Time: 181.674s
[2023-04-20 09:31:56.687705]: [8/10] Train Loss: 1.28758, Train Acc: 69.58%, Test Loss: 16.60185, Test Acc: 0.11%, Time: 183.114s
[2023-04-20 09:34:59.473789]: [9/10] Train Loss: 1.28393, Train Acc: 69.59%, Test Loss: 16.57589, Test Acc: 0.11%, Time: 182.785s
[2023-04-20 09:38:02.226685]: [10/10] Train Loss: 1.28319, Train Acc: 69.63%, Test Loss: 16.58939, Test Acc: 0.11%, Time: 182.752s
(ytorch-venv) superbench@a100-dev-000018:~/v-yiyunchen$ nvidia-smi
Thu Apr 20 09:11:43 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 520.61.05    Driver Version: 520.61.05    CUDA Version: 11.8     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM...  On   | 00000001:00:00.0 Off |                    0 |
| N/A   57C    P0   273W / 400W |  21487MiB / 40960MiB |     99%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA A100-SXM...  On   | 00000002:00:00.0 Off |                    0 |
| N/A   49C    P0   324W / 400W |  21629MiB / 40960MiB |     99%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   2  NVIDIA A100-SXM...  On   | 00000003:00:00.0 Off |                    0 |
| N/A   51C    P0   184W / 400W |  21629MiB / 40960MiB |     91%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   3  NVIDIA A100-SXM...  On   | 00000004:00:00.0 Off |                    0 |
| N/A   50C    P0   105W / 400W |  21629MiB / 40960MiB |     99%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   4  NVIDIA A100-SXM...  On   | 0000000B:00:00.0 Off |                    0 |
| N/A   51C    P0   395W / 400W |  21629MiB / 40960MiB |     99%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   5  NVIDIA A100-SXM...  On   | 0000000C:00:00.0 Off |                    0 |
| N/A   49C    P0   329W / 400W |  21485MiB / 40960MiB |     92%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   6  NVIDIA A100-SXM...  On   | 0000000D:00:00.0 Off |                    0 |
| N/A   33C    P0    58W / 400W |      3MiB / 40960MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   7  NVIDIA A100-SXM...  On   | 0000000E:00:00.0 Off |                    0 |
| N/A   32C    P0    54W / 400W |      3MiB / 40960MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     65550      C   ...n/ytorch-venv/bin/python3    21470MiB |
|    1   N/A  N/A     65551      C   ...n/ytorch-venv/bin/python3    21614MiB |
|    2   N/A  N/A     65600      C   ...n/ytorch-venv/bin/python3    21614MiB |
|    3   N/A  N/A     65602      C   ...n/ytorch-venv/bin/python3    21614MiB |
|    4   N/A  N/A     65608      C   ...n/ytorch-venv/bin/python3    21614MiB |
|    5   N/A  N/A     65609      C   ...n/ytorch-venv/bin/python3    21470MiB |
+-----------------------------------------------------------------------------+


20230420 adjust the batch size to 512 in resnet18_2.py (copied from resnet18.py) and modify the print places. still slow.


20230420 16:06 too long when waiting for finishing resnet18.py's first epoch, The time for one epoch I guess >> 1.5 hour. which is unacceptable for me. the following is the usage of GPU.
|   0  NVIDIA A100-SXM...  On   | 00000001:00:00.0 Off |                    0 |
| N/A   34C    P0    66W / 400W |   4329MiB / 40960MiB |     18%      Default |
|                               |                      |             Disabled |
