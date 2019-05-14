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

# path = 'data/canary-1/'
path = 'data/tiny-imagenet-200/train'
tiny_imagenet = TinyImageNet(path, build_dist=True)

batch_size = 51
distribution = np.zeros((22,22))

edges = np.arange(-110,110,10)
# print(edges)

a_high = 0
a_low = 100

for i, data in enumerate(D.DataLoader(tiny_imagenet, batch_size=batch_size, shuffle=False), 0):
    print('hej')
    a = data[:,1,:,:].reshape([1,-1]).numpy()[0]
    b = data[:,2,:,:].reshape([1,-1]).numpy()[0]
    #print(a.shape)
    #print(b.shape)

    #a = a + 110
    #b = b + 110

    #a = np.floor_divide(np.round(a + 110), 10)
    #a = a.astype(int)
    #b = np.floor_divide(np.round(b + 110), 10)
    #b = b.astype(int)

    #a = a[0:500]
    #b = b[0:500]

    print(a, b)
    print('argmax', np.argmax(a), np.argmax(b))

    H, xedges, yedges = np.histogram2d(a, b, bins=edges)
    print(H.shape)
    print(np.where(H > 0))


    plt.imshow(H)
    plt.show()
    #distribution[a, b] 

    #print(a)
    #print(b)

    #print(distribution[a,b])

    #print(np.amin(a))
    #print(np.amax(a))
    
    #print(distribution)


    # stop += 1
    # if stop > 5:
    #     break
    # a_channel = data[0, 0, :, :].numpy()
    # b_channel = data[0, 1, :, :].numpy()
    # X, Y = a_channel.shape
    # print('i', i)
    # for x in range(X):
    #     for y in range(Y):
    #         a = a_channel[x, y]
    #         a = np.floor_divide(np.round(a + 110), 10)
    #         a = int(a)

    #         b = b_channel[x, y]
    #         b = np.floor_divide(np.round(b + 110), 10)
    #         b = int(b)

    #         dist[a,b] = dist[a,b] + 1

# print(np.where(dist > 0))
