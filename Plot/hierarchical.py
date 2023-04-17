"""
result comes from Hierarchy/train5.py. 
"""



import re
import matplotlib.pyplot as plt

log = '''
# (Paste the log content here)
[gpu id: 1  epoch1, Animals] loss: 2.2920089508214896 time: 2023-04-17 07:46:34.762170
[gpu id: 2  epoch1, Objects] loss: 2.556488418159273 time: 2023-04-17 07:46:39.503369
[gpu id: 3  epoch1, Others] loss: 2.2085513904359604 time: 2023-04-17 07:46:41.093717
[gpu id: 1  epoch2, Animals] loss: 1.5134724261574528 time: 2023-04-17 07:46:51.413978
[gpu id: 2  epoch2, Objects] loss: 1.7574482054705969 time: 2023-04-17 07:46:58.227751
[gpu id: 3  epoch2, Others] loss: 1.4737271073659262 time: 2023-04-17 07:47:00.530604
[gpu id: 0  epoch1, Root] loss: 0.7638679411315918 time: 2023-04-17 07:47:00.693155
[gpu id: 1  epoch3, Animals] loss: 1.2747711094496308 time: 2023-04-17 07:47:07.752480
[gpu id: 2  epoch3, Objects] loss: 1.5109922901914559 time: 2023-04-17 07:47:17.443097
[gpu id: 3  epoch3, Others] loss: 1.255272408299976 time: 2023-04-17 07:47:19.341044
[gpu id: 1  epoch4, Animals] loss: 1.1128343370563296 time: 2023-04-17 07:47:24.218769
[gpu id: 2  epoch4, Objects] loss: 1.3349075530887866 time: 2023-04-17 07:47:36.732851
[gpu id: 3  epoch4, Others] loss: 1.102343098560969 time: 2023-04-17 07:47:38.150049
[gpu id: 1  epoch5, Animals] loss: 1.0017921277150157 time: 2023-04-17 07:47:40.435799
[gpu id: 0  epoch2, Root] loss: 0.6527864016723632 time: 2023-04-17 07:47:44.448801
[gpu id: 2  epoch5, Objects] loss: 1.1936301454777405 time: 2023-04-17 07:47:55.263973
[gpu id: 3  epoch5, Others] loss: 0.9778386397096845 time: 2023-04-17 07:47:56.767911
[gpu id: 1  epoch6, Animals] loss: 0.8891250757811644 time: 2023-04-17 07:47:57.203571
[gpu id: 1  epoch7, Animals] loss: 0.8046801591902131 time: 2023-04-17 07:48:13.744685
[gpu id: 2  epoch6, Objects] loss: 1.0710653940541972 time: 2023-04-17 07:48:14.149297
[gpu id: 3  epoch6, Others] loss: 0.8738458369970321 time: 2023-04-17 07:48:15.342096
[gpu id: 0  epoch3, Root] loss: 0.5965826799583435 time: 2023-04-17 07:48:28.116044
[gpu id: 1  epoch8, Animals] loss: 0.7135360777345758 time: 2023-04-17 07:48:30.294046
[gpu id: 2  epoch7, Objects] loss: 0.9716044485403278 time: 2023-04-17 07:48:32.868720
[gpu id: 3  epoch7, Others] loss: 0.7740334381130006 time: 2023-04-17 07:48:34.197529
[gpu id: 1  epoch9, Animals] loss: 0.6358123534801466 time: 2023-04-17 07:48:47.070247
[gpu id: 2  epoch8, Objects] loss: 0.8626451532359031 time: 2023-04-17 07:48:51.283947
[gpu id: 3  epoch8, Others] loss: 0.7017997831900914 time: 2023-04-17 07:48:53.209089
[gpu id: 1  epoch10, Animals] loss: 0.5817067299902827 time: 2023-04-17 07:49:03.953156
Finished training Animals
[gpu id: 2  epoch9, Objects] loss: 0.7899939149783871 time: 2023-04-17 07:49:10.107375
[gpu id: 3  epoch9, Others] loss: 0.6261862656672795 time: 2023-04-17 07:49:11.710028
[gpu id: 0  epoch4, Root] loss: 0.5553670062446594 time: 2023-04-17 07:49:12.369045
[gpu id: 2  epoch10, Objects] loss: 0.7121110076259086 time: 2023-04-17 07:49:28.671041
Finished training Objects
[gpu id: 3  epoch10, Others] loss: 0.5661421606871817 time: 2023-04-17 07:49:30.692555
Finished training Others
[gpu id: 0  epoch5, Root] loss: 0.5256321972990036 time: 2023-04-17 07:49:56.635615
[gpu id: 0  epoch6, Root] loss: 0.4972295149421692 time: 2023-04-17 07:50:40.506731
[gpu id: 0  epoch7, Root] loss: 0.4708647178554535 time: 2023-04-17 07:51:24.609404
[gpu id: 0  epoch8, Root] loss: 0.4453670884895325 time: 2023-04-17 07:52:08.731619
[gpu id: 0  epoch9, Root] loss: 0.427056620016098 time: 2023-04-17 07:52:52.484020
[gpu id: 0  epoch10, Root] loss: 0.4091861226129532 time: 2023-04-17 07:53:36.219451
Finished training Root
'''

# Parse the log and store the loss values
loss_values = {}
for line in log.splitlines():
    match = re.match(r'\[gpu id: (\d)  epoch(\d+), (.+?)\] loss: (.+?) time:', line)
    if match:
        gpu_id, epoch, label, loss = match.groups()
        key = f'{gpu_id}_{label}'
        if key not in loss_values:
            loss_values[key] = []
        loss_values[key].append((int(epoch), float(loss)))

# Plot the loss curve
plt.figure()
for key, values in loss_values.items():
    epochs, losses = zip(*values)
    plt.plot(epochs, losses, label=key)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')
plt.show()
