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

path = 'data/canary-1/'
tiny_imagenet = TinyImageNet(path, build_dist=True)

for i, data in enumerate(D.DataLoader(tiny_imagenet, batch_size=1, shuffle=False), 0):
    a_channel = data[0, 0, :, :].numpy()
    b_channel = data[0, 1, :, :].numpy()
    X, Y = a_channel.shape
    print('start', X, Y)
    for x in range(X):
        for y in range(Y):
            print(x, y)


# inputs, labels = dataiter.next()
# print(inputs.size())
# print(labels.size())
