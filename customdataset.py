import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils import data as D
import glob
import os.path as osp
from skimage.color import rgb2lab, lab2rgb

from PIL import Image


class TreeDataset(D.Dataset):
    def __init__(self, build_dist=False):
        self.filenames = []
        self.root = 'data/tree-training-1/'
        self.build_dist = build_dist

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

        # [batch_size, 224, 244] and [batch_size, 2, 224, 224]
        return img_tensor[0, :, :], img_tensor[1:, :, :]

    def __len__(self):
        return self.len


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


tree_d = TreeDataset()
loader = D.DataLoader(tree_d, batch_size=1, shuffle=False)
data_iter = iter(loader)
feat, label = data_iter.next()
imshow(feat[0], label[0])
