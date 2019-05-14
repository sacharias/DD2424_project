import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from torch.utils import data as D
from tqdm import tqdm

from customdataset import TreeDataset

class DistributionBuilder():
    def __init__(self):
        self.batch_size = 50
        self.distribution = np.zeros((22,22))
        self.classes = np.zeros((22,22))
        self.images = TreeDataset(build_dist=True)
    
    def build(self):
        edges = np.arange(-110, 120, 10)
        total = math.ceil(len(self.images) // self.batch_size)

        with tqdm(total=total) as pbar:
            for i, data in enumerate(D.DataLoader(self.images, batch_size=self.batch_size, shuffle=False), 0):
                a = data[:,1,:,:].reshape([1,-1]).numpy()[0]
                b = data[:,2,:,:].reshape([1,-1]).numpy()[0]

                H, xedges, yedges = np.histogram2d(a, b, bins=edges)
                H = np.divide(H, a.shape[0])
                self.distribution = self.distribution + H
                pbar.update()
            
            self.distribution = self.distribution / total
            self.classes = np.where(self.distribution > 0, 1, 0)
    
    def plot(self):
        plt.imshow(self.distribution)
        plt.show()

        plt.imshow(self.classes)
        plt.show()
    
    def save(self):
        np.save('distribution.npy', self.distribution)
        np.save('classes.npy', self.classes)
    
    def load(self):
        self.distribution = np.load('distribution.npy')
        self.classes = np.load('classes.npy')

    def print_data(self):
        print('count_classes:', np.count_nonzero(self.classes))


db = DistributionBuilder()
db.build()
#db.save()
#db.load()
#db.plot()
#db.print_data()