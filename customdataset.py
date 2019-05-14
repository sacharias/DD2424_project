import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils import data as D
import glob
import os.path as osp
from skimage.color import rgb2lab, lab2rgb

from PIL import Image

NO_CLASSES = 484
classes = np.arange(NO_CLASSES)
classes = np.reshape(classes, (22, -1))

class TreeDataset(D.Dataset):
    def __init__(self, build_dist=False, class_vector=False):
        self.filenames = []
        self.root = 'data/tree-training-1/'
        self.build_dist = build_dist
        self.class_vector = class_vector

        for fn in glob.glob(osp.join(self.root, '*.jpg')):
            self.filenames.append(fn)

        self.len = len(self.filenames)

        self.transform_before = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
        self.transform_after = transforms.ToTensor()

    def __getitem__(self, index):
        img = Image.open(self.filenames[index])
        img = self.transform_before(img)
        img = rgb2lab(img)

        img_tensor = self.transform_after(img)

        if (self.build_dist):
            return img_tensor

        elif self.class_vector:
            colors = img_tensor[1:, :, :]
            one_hot = self.channels_to_vector(colors)
            return img_tensor[0,:,:], one_hot

        # [batch_size, 224, 244] and [batch_size, 2, 224, 224]
        return img_tensor[0, :, :], img_tensor[1:, :, :]
    
    def channels_to_vector(self, colors):
        a = colors[0, :, :]
        b = colors[1, :, :]
        vect_chan = np.vectorize(channel_to_class)
        res = vect_chan(a, b)

        one_hot = np.zeros((res.shape[0], res.shape[1], NO_CLASSES))
        layer_idx = np.arange(res.shape[0]).reshape(res.shape[0], 1)
        component_idx = np.tile(np.arange(res.shape[1]), (res.shape[0], 1))
        one_hot[layer_idx, component_idx, res] = 1

        return one_hot

    def __len__(self):
        return self.len

def channel_to_class(a, b):
    a = (a + 110) // 10
    b = (b + 110) // 10
    c = classes[int(a), int(b)]
    return c

def one_hot(x):
    return 0

def imshow(img_tensor, label_tensor=None):
    img = img_tensor
    if label_tensor is not None:
        img = torch.zeros([3, 224, 224])
        img[0] = img_tensor
        img[1] = label_tensor[0]
        img[2] = label_tensor[1]

    npimg = np.transpose(img.numpy(), (1, 2, 0))
    npimg = lab2rgb(npimg)
    plt.imshow(npimg)
    plt.show()

def testDataset():
    tree_d = TreeDataset()
    loader = D.DataLoader(tree_d, batch_size=1, shuffle=False)
    data_iter = iter(loader)
    data = data_iter.next()
    #feat, label = data_iter.next()
    #imshow(feat[0], label[0])

def sakiTest():
    mx = np.arange(484)
    mx = np.reshape(mx, (22, -1))

    print(mx.shape)
    print(mx[21,21])

# sakiTest()
# testDataset()