import matplotlib.pyplot as plt
import json
import numpy as np


root = "/home/wangbowen/PycharmProjects/mmpose/work_dirs/hrnet_w32_coco_tiny_256x192/None.log.json"
df = [json.loads(line) for line in open(root, 'r', encoding='utf-8')]


loss_epoch = []
epoch_record = []
for i in range(len(df)):
    current_dict = df[i]
    kk = list(current_dict.keys())
    if "mode" not in kk:
        continue

    current_epoch = current_dict["epoch"]
    if current_epoch not in epoch_record:
        epoch_record.append(current_epoch)
        loss_epoch.append(current_dict["loss"])
    else:
        continue

plt.figure(figsize=(15, 8), dpi=80)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.linewidth'] = 3

font = 30
x_index = np.arange(1, len(loss_epoch) + 1, 1)
plt.plot(x_index, np.array(loss_epoch), color="blue", linewidth=3)
plt.tick_params(labelsize=24)

plt.xlabel('Epochs', fontsize=font-4)
plt.ylabel('Training Loss', fontsize=font - 4)
plt.title("")

plt.savefig("loss_plot.png")
plt.tight_layout()
plt.show()
