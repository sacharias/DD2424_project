import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data as D

from customdataset import TinyImageNet, imshow_combine

dist = np.zeros((22,22))

path = 'data/canary-1/'
tiny_imagenet = TinyImageNet(path, build_dist=True)

stop = 0

for i, data in enumerate(D.DataLoader(tiny_imagenet, batch_size=1, shuffle=False), 0):
    stop += 1
    if stop > 5:
        break
    a_channel = data[0, 0, :, :].numpy()
    b_channel = data[0, 1, :, :].numpy()
    X, Y = a_channel.shape
    print('i', i)
    for x in range(X):
        for y in range(Y):
            a = a_channel[x, y]
            a = np.floor_divide(np.round(a + 110), 10)
            a = int(a)

            b = b_channel[x, y]
            b = np.floor_divide(np.round(b + 110), 10)
            b = int(b)

            dist[a,b] = dist[a,b] + 1

print(np.where(dist > 0))
