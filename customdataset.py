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

class TinyImageNet(D.Dataset):
    def __init__(self, root, build_dist=False):
        self.filenames = []
        self.root = root
        if build_dist:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                # transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.ToTensor()

        filenames = glob.glob(osp.join(root, '*.jpg'))
        for fn in filenames:
            self.filenames.append(fn)
        self.len = len(self.filenames)
        self.build_dist = build_dist
    
    def __getitem__(self, index):
        # returns features, labels
        img = Image.open(self.filenames[index])
        img = self.transform(img)
        img = rgb2lab(img)

        if (self.build_dist):
            return transforms.ToTensor()(img)

        return 0
        #labels = img_tensor[[1,2],:,:]
        #return img_tensor, labels.view(1, -1)
        # return img_tensor[0,:,:], img_tensor[[1,2],:,:].view(1, -1) 
    
    def __len__(self):
        return self.len

def imshow(img):
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1,2,0))
    npimg = lab2rgb(npimg)
    plt.imshow(npimg)
    plt.show()

def imshow_combine(feature, labels):
    image = torch.zeros([3, 64, 64])
    image[0] = feature
    image[1] = labels[0]
    image[2] = labels[1]
    imshow(image)

# imagenet = TinyImageNet(path)
# loader = D.DataLoader(imagenet, batch_size=1, shuffle=False)
# dataiter = iter(loader)
# features, labels = dataiter.next()

# feat = features[0]
# lab = labels[0]
# print(lab.size())

# imshow_combine(feat, lab)
