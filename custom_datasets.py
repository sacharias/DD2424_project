import torch
from torch.utils import data as D
import glob
import os.path as osp

import matplotlib.pyplot as plt
import torch.nn.functional as F

from show_data import imshow

class TreeDataset(D.Dataset):
    def __init__(self, root='data/tree-training-1-tensor'):
        self.filenames = []
        self.root = root
        for fn in glob.glob(osp.join(self.root, '*.pt')):
            self.filenames.append(fn)
        
        self.len = len(self.filenames)
    
    def __getitem__(self, index):
        img_tensor = torch.load(self.filenames[index])
        labels = F.max_pool2d(img_tensor[1:3], 4)
        return img_tensor[0].unsqueeze(0), labels
    
    def __len__(self):
        return self.len

class TreeClassDataset(D.Dataset):
    def __init__(self, root='data/tree-training-1-classtensor'):
        self.filenames = []
        self.root = root
        for fn in glob.glob(osp.join(self.root, '*.pt')):
            self.filenames.append(fn)   

        self.len = len(self.filenames)
    
    def __getitem__(self, index):
        img_tensor = torch.load(self.filenames[index])
        return img_tensor[0].unsqueeze(0), img_tensor[1]
    
    def __len__(self):
        return self.len

# tree_d = TreeDataset()
# loader = D.DataLoader(tree_d, batch_size=1, shuffle=True)
# for i, data in enumerate(loader, 0):
#     inputs, label = data
#     img_tensor = torch.zeros([3,224,224])
#     print(img_tensor.size())
#     img_tensor[0] = inputs[0]
#     img_tensor[1] = label[0][0]
#     img_tensor[2] = label[0][1]

#     imshow(img_tensor)
#     print(img_tensor)
#     print(inputs.size())
#     print(label.size())
#     break
