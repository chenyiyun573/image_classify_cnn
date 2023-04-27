



20230422 13:07 change parallel2.py's lr to 0.04, and epochs to 30
  return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
[2023-04-23 05:12:16.951050]: [1/30] Train Loss: 6.17805, Train Acc: 2.45%, Test Loss: 5.61739, Test Acc: 5.02%, Time: 187.794s
[2023-04-23 05:15:23.599218]: [2/30] Train Loss: 5.24589, Train Acc: 7.66%, Test Loss: 5.22023, Test Acc: 7.59%, Time: 186.647s
[2023-04-23 05:18:30.954190]: [3/30] Train Loss: 4.72892, Train Acc: 12.53%, Test Loss: 4.38267, Test Acc: 15.16%, Time: 187.354s
[2023-04-23 05:21:37.798395]: [4/30] Train Loss: 4.34417, Train Acc: 16.98%, Test Loss: 4.07669, Test Acc: 18.91%, Time: 186.844s
[2023-04-23 05:24:44.900654]: [5/30] Train Loss: 4.03934, Train Acc: 20.82%, Test Loss: 3.86763, Test Acc: 21.54%, Time: 187.102s
[2023-04-23 05:27:52.178538]: [6/30] Train Loss: 3.79621, Train Acc: 24.18%, Test Loss: 3.54493, Test Acc: 26.33%, Time: 187.277s
[2023-04-23 05:30:58.265616]: [7/30] Train Loss: 3.60012, Train Acc: 27.10%, Test Loss: 3.40213, Test Acc: 28.51%, Time: 186.086s
[2023-04-23 05:34:05.863846]: [8/30] Train Loss: 3.42837, Train Acc: 29.70%, Test Loss: 3.23980, Test Acc: 31.11%, Time: 187.598s
[2023-04-23 05:37:12.564239]: [9/30] Train Loss: 3.28588, Train Acc: 31.97%, Test Loss: 3.08829, Test Acc: 33.65%, Time: 186.700s
[2023-04-23 05:40:20.623256]: [10/30] Train Loss: 3.16967, Train Acc: 33.84%, Test Loss: 3.01928, Test Acc: 34.73%, Time: 188.058s
[2023-04-23 05:43:28.158304]: [11/30] Train Loss: 3.06155, Train Acc: 35.66%, Test Loss: 2.85226, Test Acc: 37.67%, Time: 187.534s
[2023-04-23 05:46:35.960578]: [12/30] Train Loss: 2.97165, Train Acc: 37.12%, Test Loss: 2.92995, Test Acc: 36.47%, Time: 187.801s
[2023-04-23 05:49:44.140095]: [13/30] Train Loss: 2.88971, Train Acc: 38.56%, Test Loss: 2.70343, Test Acc: 40.13%, Time: 188.179s
[2023-04-23 05:52:52.039956]: [14/30] Train Loss: 2.81545, Train Acc: 39.82%, Test Loss: 2.66961, Test Acc: 40.89%, Time: 187.899s
[2023-04-23 05:56:00.493306]: [15/30] Train Loss: 2.74941, Train Acc: 40.95%, Test Loss: 2.51316, Test Acc: 43.58%, Time: 188.452s
[2023-04-23 05:59:08.016249]: [16/30] Train Loss: 2.68921, Train Acc: 42.00%, Test Loss: 2.59405, Test Acc: 42.29%, Time: 187.522s
[2023-04-23 06:02:15.686668]: [17/30] Train Loss: 2.63507, Train Acc: 42.99%, Test Loss: 2.50619, Test Acc: 43.89%, Time: 187.670s
[2023-04-23 06:05:23.482275]: [18/30] Train Loss: 2.58410, Train Acc: 43.87%, Test Loss: 2.44303, Test Acc: 45.06%, Time: 187.795s
[2023-04-23 06:08:31.345692]: [19/30] Train Loss: 2.53512, Train Acc: 44.82%, Test Loss: 2.52084, Test Acc: 43.81%, Time: 187.863s
[2023-04-23 06:11:39.587805]: [20/30] Train Loss: 2.49482, Train Acc: 45.46%, Test Loss: 2.42361, Test Acc: 45.39%, Time: 188.241s
[2023-04-23 06:14:46.858311]: [21/30] Train Loss: 2.45634, Train Acc: 46.20%, Test Loss: 2.30003, Test Acc: 47.76%, Time: 187.269s
[2023-04-23 06:17:53.233237]: [22/30] Train Loss: 2.41591, Train Acc: 47.00%, Test Loss: 2.28702, Test Acc: 47.76%, Time: 186.374s
[2023-04-23 06:21:01.127629]: [23/30] Train Loss: 2.38309, Train Acc: 47.58%, Test Loss: 2.36571, Test Acc: 46.29%, Time: 187.894s
[2023-04-23 06:24:09.073164]: [24/30] Train Loss: 2.35384, Train Acc: 48.11%, Test Loss: 2.19876, Test Acc: 49.51%, Time: 187.945s
[2023-04-23 06:27:16.568589]: [25/30] Train Loss: 2.31991, Train Acc: 48.74%, Test Loss: 2.24877, Test Acc: 48.83%, Time: 187.495s
[2023-04-23 06:30:24.070229]: [26/30] Train Loss: 2.28825, Train Acc: 49.30%, Test Loss: 2.18719, Test Acc: 50.06%, Time: 187.501s
[2023-04-23 06:33:30.908551]: [27/30] Train Loss: 2.26461, Train Acc: 49.77%, Test Loss: 2.22832, Test Acc: 49.40%, Time: 186.837s
[2023-04-23 06:36:38.881429]: [28/30] Train Loss: 2.24268, Train Acc: 50.19%, Test Loss: 2.16459, Test Acc: 50.22%, Time: 187.972s
[2023-04-23 06:39:46.620478]: [29/30] Train Loss: 2.21247, Train Acc: 50.74%, Test Loss: 2.22659, Test Acc: 49.37%, Time: 187.738s
[2023-04-23 06:42:55.321431]: [30/30] Train Loss: 2.19408, Train Acc: 51.13%, Test Loss: 2.13114, Test Acc: 51.05%, Time: 188.700s


20230422 20:40 the result of parallel2.py I changed the lr to 0.02 and epochs to 30 
[2023-04-22 16:29:53.915588]: [1/30] Train Loss: 6.41797, Train Acc: 1.57%, Test Loss: 5.95073, Test Acc: 3.05%, Time: 187.977s
[2023-04-22 16:32:59.486238]: [2/30] Train Loss: 5.65312, Train Acc: 4.74%, Test Loss: 5.39242, Test Acc: 6.34%, Time: 185.570s
[2023-04-22 16:36:05.456937]: [3/30] Train Loss: 5.23818, Train Acc: 7.69%, Test Loss: 5.01511, Test Acc: 9.05%, Time: 185.970s
[2023-04-22 16:39:12.097215]: [4/30] Train Loss: 4.92347, Train Acc: 10.54%, Test Loss: 4.79753, Test Acc: 10.92%, Time: 186.640s
[2023-04-22 16:42:19.486269]: [5/30] Train Loss: 4.65647, Train Acc: 13.29%, Test Loss: 4.52399, Test Acc: 13.67%, Time: 187.388s
[2023-04-22 16:45:26.306158]: [6/30] Train Loss: 4.43393, Train Acc: 15.87%, Test Loss: 4.29228, Test Acc: 16.25%, Time: 186.819s
[2023-04-22 16:48:33.343347]: [7/30] Train Loss: 4.24292, Train Acc: 18.21%, Test Loss: 3.96240, Test Acc: 20.58%, Time: 187.037s
[2023-04-22 16:51:40.772254]: [8/30] Train Loss: 4.06999, Train Acc: 20.41%, Test Loss: 3.82231, Test Acc: 22.41%, Time: 187.428s
[2023-04-22 16:54:48.639132]: [9/30] Train Loss: 3.92579, Train Acc: 22.37%, Test Loss: 3.73929, Test Acc: 23.74%, Time: 187.866s
[2023-04-22 16:57:57.236266]: [10/30] Train Loss: 3.78916, Train Acc: 24.34%, Test Loss: 3.54900, Test Acc: 26.40%, Time: 188.596s
[2023-04-22 17:01:04.197785]: [11/30] Train Loss: 3.66839, Train Acc: 26.07%, Test Loss: 3.45615, Test Acc: 27.73%, Time: 186.960s
[2023-04-22 17:04:12.044698]: [12/30] Train Loss: 3.56229, Train Acc: 27.64%, Test Loss: 3.46748, Test Acc: 27.57%, Time: 187.846s
[2023-04-22 17:07:19.958336]: [13/30] Train Loss: 3.46302, Train Acc: 29.21%, Test Loss: 3.21663, Test Acc: 31.50%, Time: 187.913s
[2023-04-22 17:10:27.268780]: [14/30] Train Loss: 3.37671, Train Acc: 30.51%, Test Loss: 3.19453, Test Acc: 31.85%, Time: 187.310s
[2023-04-22 17:13:36.177809]: [15/30] Train Loss: 3.29551, Train Acc: 31.76%, Test Loss: 3.15294, Test Acc: 32.56%, Time: 188.908s
[2023-04-22 17:16:43.608230]: [16/30] Train Loss: 3.22252, Train Acc: 32.95%, Test Loss: 3.04731, Test Acc: 34.41%, Time: 187.429s
[2023-04-22 17:19:50.944082]: [17/30] Train Loss: 3.15354, Train Acc: 34.06%, Test Loss: 2.99975, Test Acc: 35.25%, Time: 187.335s
[2023-04-22 17:22:58.670925]: [18/30] Train Loss: 3.09047, Train Acc: 35.11%, Test Loss: 3.05139, Test Acc: 34.70%, Time: 187.726s
[2023-04-22 17:26:05.824831]: [19/30] Train Loss: 3.03076, Train Acc: 36.16%, Test Loss: 2.88384, Test Acc: 37.18%, Time: 187.153s
[2023-04-22 17:29:13.740507]: [20/30] Train Loss: 2.97633, Train Acc: 37.05%, Test Loss: 2.82188, Test Acc: 38.09%, Time: 187.915s
[2023-04-22 17:32:21.032465]: [21/30] Train Loss: 2.92479, Train Acc: 37.93%, Test Loss: 2.68118, Test Acc: 40.97%, Time: 187.290s
[2023-04-22 17:35:28.044317]: [22/30] Train Loss: 2.87794, Train Acc: 38.76%, Test Loss: 2.66617, Test Acc: 41.13%, Time: 187.011s
[2023-04-22 17:38:34.850047]: [23/30] Train Loss: 2.83504, Train Acc: 39.48%, Test Loss: 2.66479, Test Acc: 41.10%, Time: 186.805s
[2023-04-22 17:41:42.597564]: [24/30] Train Loss: 2.79373, Train Acc: 40.23%, Test Loss: 2.58051, Test Acc: 42.61%, Time: 187.747s
[2023-04-22 17:44:49.880821]: [25/30] Train Loss: 2.75535, Train Acc: 40.90%, Test Loss: 2.59503, Test Acc: 42.22%, Time: 187.283s
[2023-04-22 17:47:57.878356]: [26/30] Train Loss: 2.71324, Train Acc: 41.64%, Test Loss: 2.50631, Test Acc: 44.10%, Time: 187.997s
[2023-04-22 17:51:05.447678]: [27/30] Train Loss: 2.68125, Train Acc: 42.25%, Test Loss: 2.51698, Test Acc: 44.04%, Time: 187.569s
[2023-04-22 17:54:13.093371]: [28/30] Train Loss: 2.64865, Train Acc: 42.77%, Test Loss: 2.48313, Test Acc: 44.45%, Time: 187.645s
[2023-04-22 17:57:20.343958]: [29/30] Train Loss: 2.61217, Train Acc: 43.38%, Test Loss: 2.45908, Test Acc: 44.73%, Time: 187.250s
[2023-04-22 18:00:27.523887]: [30/30] Train Loss: 2.58329, Train Acc: 43.98%, Test Loss: 2.44259, Test Acc: 45.40%, Time: 187.179s




20230422 23:31 the result of parallel2.py I changed the lr from 0.001 to 0.01
[2023-04-22 15:56:52.457022]: [1/10] Train Loss: 6.64516, Train Acc: 0.98%, Test Loss: 6.29571, Test Acc: 1.77%, Time: 187.781s
[2023-04-22 16:00:01.221142]: [2/10] Train Loss: 6.03132, Train Acc: 2.85%, Test Loss: 5.83401, Test Acc: 3.72%, Time: 188.763s
[2023-04-22 16:03:09.200925]: [3/10] Train Loss: 5.69252, Train Acc: 4.55%, Test Loss: 5.63916, Test Acc: 4.93%, Time: 187.979s
[2023-04-22 16:06:15.668587]: [4/10] Train Loss: 5.43412, Train Acc: 6.25%, Test Loss: 5.15971, Test Acc: 7.89%, Time: 186.467s
[2023-04-22 16:09:23.049941]: [5/10] Train Loss: 5.21680, Train Acc: 7.88%, Test Loss: 5.02523, Test Acc: 8.89%, Time: 187.381s
[2023-04-22 16:12:29.564235]: [6/10] Train Loss: 5.03278, Train Acc: 9.48%, Test Loss: 4.83375, Test Acc: 10.74%, Time: 186.514s
[2023-04-22 16:15:36.491492]: [7/10] Train Loss: 4.86764, Train Acc: 11.09%, Test Loss: 4.63401, Test Acc: 12.45%, Time: 186.927s
[2023-04-22 16:18:43.862898]: [8/10] Train Loss: 4.71686, Train Acc: 12.67%, Test Loss: 4.62841, Test Acc: 12.91%, Time: 187.370s
[2023-04-22 16:21:49.600663]: [9/10] Train Loss: 4.58318, Train Acc: 14.16%, Test Loss: 4.33105, Test Acc: 16.12%, Time: 185.736s
[2023-04-22 16:24:57.163739]: [10/10] Train Loss: 4.45948, Train Acc: 15.54%, Test Loss: 4.22837, Test Acc: 17.07%, Time: 187.562s


20230422 23:28 the result of parallel2.py without pretrained network: 
[2023-04-22 14:55:07.967288]: [1/10] Train Loss: 6.93532, Train Acc: 0.15%, Test Loss: 6.88683, Test Acc: 0.22%, Time: 186.959s
[2023-04-22 14:58:12.828062]: [2/10] Train Loss: 6.84264, Train Acc: 0.50%, Test Loss: 6.80398, Test Acc: 0.63%, Time: 184.860s
[2023-04-22 15:01:19.333319]: [3/10] Train Loss: 6.77260, Train Acc: 0.82%, Test Loss: 6.72143, Test Acc: 0.88%, Time: 186.504s
[2023-04-22 15:04:25.428355]: [4/10] Train Loss: 6.68829, Train Acc: 1.04%, Test Loss: 6.61888, Test Acc: 1.13%, Time: 186.094s
[2023-04-22 15:07:31.489606]: [5/10] Train Loss: 6.58842, Train Acc: 1.23%, Test Loss: 6.51464, Test Acc: 1.39%, Time: 186.061s
[2023-04-22 15:10:37.622672]: [6/10] Train Loss: 6.48740, Train Acc: 1.45%, Test Loss: 6.40988, Test Acc: 1.58%, Time: 186.133s
[2023-04-22 15:13:44.229694]: [7/10] Train Loss: 6.38947, Train Acc: 1.72%, Test Loss: 6.33344, Test Acc: 1.75%, Time: 186.606s
[2023-04-22 15:16:51.707997]: [8/10] Train Loss: 6.29882, Train Acc: 1.93%, Test Loss: 6.21680, Test Acc: 2.14%, Time: 187.478s
[2023-04-22 15:19:58.898027]: [9/10] Train Loss: 6.21840, Train Acc: 2.19%, Test Loss: 6.13493, Test Acc: 2.43%, Time: 187.189s
[2023-04-22 15:23:05.195703]: [10/10] Train Loss: 6.14534, Train Acc: 2.44%, Test Loss: 6.06622, Test Acc: 2.78%, Time: 186.297s



20230422 20:38 train1.py has a problem is that the set_trainloader is executed many times for each network it need. I ask GPT to solve it by modify the trainloader calling in train_model. create train2.py.


20230421 15:07 the test accuracy keep the same is caused by the error of val dataset's mislabeled folder name.
Then the following result use batch size 512, lr = 0.001
  return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
[2023-04-21 06:31:33.069476]: [1/10] Train Loss: 1.32086, Train Acc: 68.90%, Test Loss: 1.25169, Test Acc: 69.28%, Time: 187.379s
[2023-04-21 06:34:40.119891]: [2/10] Train Loss: 1.30678, Train Acc: 69.16%, Test Loss: 1.24863, Test Acc: 69.39%, Time: 187.050s
[2023-04-21 06:37:45.850284]: [3/10] Train Loss: 1.30098, Train Acc: 69.30%, Test Loss: 1.24585, Test Acc: 69.40%, Time: 185.730s
[2023-04-21 06:40:52.480210]: [4/10] Train Loss: 1.29448, Train Acc: 69.43%, Test Loss: 1.24430, Test Acc: 69.43%, Time: 186.629s
[2023-04-21 06:44:00.705285]: [5/10] Train Loss: 1.29364, Train Acc: 69.42%, Test Loss: 1.24340, Test Acc: 69.40%, Time: 188.224s
[2023-04-21 06:47:07.511050]: [6/10] Train Loss: 1.29116, Train Acc: 69.50%, Test Loss: 1.24344, Test Acc: 69.42%, Time: 186.805s
[2023-04-21 06:50:14.216963]: [7/10] Train Loss: 1.29405, Train Acc: 69.43%, Test Loss: 1.24232, Test Acc: 69.41%, Time: 186.705s
[2023-04-21 06:53:22.106469]: [8/10] Train Loss: 1.28915, Train Acc: 69.59%, Test Loss: 1.24249, Test Acc: 69.45%, Time: 187.889s
[2023-04-21 06:56:30.308222]: [9/10] Train Loss: 1.28885, Train Acc: 69.56%, Test Loss: 1.24110, Test Acc: 69.47%, Time: 188.201s
[2023-04-21 06:59:37.987657]: [10/10] Train Loss: 1.28531, Train Acc: 69.62%, Test Loss: 1.23785, Test Acc: 69.55%, Time: 187.679s







20230420 13:45 parallel.py change lr to 0.01, batch size to 256,  use 8 GPUs
[2023-04-21 05:50:03.875075]: [1/10] Train Loss: 1.34990, Train Acc: 68.18%, Test Loss: 16.43129, Test Acc: 0.10%, Time: 183.637s
[2023-04-21 05:53:07.787825]: [2/10] Train Loss: 1.34121, Train Acc: 68.36%, Test Loss: 16.40718, Test Acc: 0.11%, Time: 183.912s
[2023-04-21 05:56:20.523822]: [3/10] Train Loss: 1.33560, Train Acc: 68.44%, Test Loss: 16.61668, Test Acc: 0.11%, Time: 192.735s
[2023-04-21 05:59:25.236531]: [4/10] Train Loss: 1.32429, Train Acc: 68.66%, Test Loss: 16.53988, Test Acc: 0.11%, Time: 184.712s
[2023-04-21 06:02:29.535448]: [5/10] Train Loss: 1.32297, Train Acc: 68.72%, Test Loss: 16.73369, Test Acc: 0.11%, Time: 184.298s



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
