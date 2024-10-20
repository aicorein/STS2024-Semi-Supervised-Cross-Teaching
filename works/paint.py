import json

import matplotlib.pyplot as plt
import numpy as np

with open(
    r"/home/melodyecho/miccai/SSL4MIS/output/Validation_0010_Mask.json",
    encoding="utf-8",
) as fp:
    obj = json.load(fp)

data = []
labels = []
for shape in obj["shapes"]:
    for px, py in shape["points"]:
        data.append((px, py))
        labels.append(shape["label"])

plt.figure(figsize=(32, 18))
plt.xlim((0, 2000))
plt.ylim((0, 942))
data = np.array(data)
plt.gca().invert_yaxis()
for label in set(labels):
    x = [data[i][0] for i in range(len(data)) if labels[i] == label]
    y = [data[i][1] for i in range(len(data)) if labels[i] == label]
    plt.scatter(x, y, label=label)
plt.legend()
plt.savefig("./output.png")
