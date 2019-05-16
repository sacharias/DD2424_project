import torch
from torch.utils import data as D
import glob
import os.path as osp

import matplotlib.pyplot as plt

class TreeDataset(D.Dataset):
    def __init__(self):
        self.filenames = []
        self.root = 'data/tree-training-1-tensor'
        for fn in glob.glob(osp.join(self.root, '*.pt')):
            self.filenames.append(fn)
        
        self.len = len(self.filenames)
    
    def __getitem__(self, index):
        img_tensor = torch.load(self.filenames[index])
        return img_tensor[0], img_tensor[1:3]
    
    def __len__(self):
        return self.len

class TreeClassDataset(D.Dataset):
    def __init__(self):
        self.filenames = []
        self.root = 'data/tree-training-1-classtensor'
        for fn in glob.glob(osp.join(self.root, '*.pt')):
            self.filenames.append(fn)   

        self.len = len(self.filenames)
    
    def __getitem__(self, index):
        img_tensor = torch.load(self.filenames[index])
        return img_tensor[0], img_tensor[1]
    
    def __len__(self):
        return self.len

# tree_d = TreeDataset()
# loader = D.DataLoader(tree_d, batch_size=16, shuffle=False)
# for i, data in enumerate(loader, 0):
#     inputs, label = data
#     print(inputs.size())
#     print(label.size())
#     break
