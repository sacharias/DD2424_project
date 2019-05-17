import torch
import numpy as np
from skimage.color import lab2rgb
import matplotlib.pyplot as plt

def show_tensor():
    filename = 'data/tree-training-1-tensor/0a0a2d70c6eae4d2.pt'
    img_tensor = torch.load(filename)
    print(img_tensor.size())
    imshow(img_tensor)

def imshow(img_tensor):
    img_tensor = np.transpose(img_tensor.numpy(), (1,2,0))
    img = lab2rgb(img_tensor)
    plt.imshow(img)
    plt.show()

# show_tensor()
