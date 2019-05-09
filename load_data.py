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

import os
import glob
import os.path as osp
from PIL import Image

from skimage.color import rgb2lab, lab2rgb

path = 'data/tiny-imagenet-200/train/n01443537/images/'

class TinyImageNet(D.Dataset):
    def __init__(self, root):
        self.filenames = []
        self.root = root
        self.transform = transforms.ToTensor()
        filenames = glob.glob(osp.join(path, '*.JPEG'))
        for fn in filenames:
            self.filenames.append(fn)
        self.len = len(self.filenames)
    
    def __getitem__(self, index):
        # return label also
        image = Image.open(self.filenames[index])
        image = rgb2lab(image)
        return self.transform(image)
    
    def __len__(self):
        return self.len

def imshow(img):
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1,2,0))
    npimg = lab2rgb(npimg)
    plt.imshow(npimg)
    plt.show()

imagenet = TinyImageNet(path)
print(imagenet.len)

loader = D.DataLoader(imagenet, batch_size=1, shuffle=False)
dataiter = iter(loader)
images = dataiter.next()

image = images[0]

# make image black and white
print(image.size())
print(image)

#image[1,:,:] = 0
#image[2,:,:] = 0
# image[0,:,:] = 50


imshow(image)

#plt.figure(figsize=(16,8))
#imshow(torchvision.utils.make_grid(images))

