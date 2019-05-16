import numpy as np
import torch
import torchvision.transforms as transforms
import glob
import os
import os.path as osp
from skimage.color import rgb2lab, lab2rgb
from PIL import Image
from tqdm import tqdm

NO_CLASSES = 484
classes = np.arange(NO_CLASSES)
classes = np.reshape(classes, (22, -1))

# HOW TO:
# Transform images to tensor and class tensor
# 1. make new folder '*-square' and run: find . -name '*.jpg' -execdir mogrify -resize 224x224^ -gravity Center -extent 224x224 {} \;
# 2. run transform_data() that makes folder '*-tensor' (tensor of size 224 x 224 x 3 in CIELAB)
# 3. run tensor_to_classtensor() that makes folder '*-classtensor' (tensor of size 224 x 224 x 3 in CIELAB)

def transform_data():
    """
    Transforms image to tensor of size 224 x 224 x 3
    """
    path = 'data/tree-training-1-square/'
    result_path = 'data/tree-training-1-tensor/'
    filenames = []
    transform = transforms.ToTensor()

    for fn in glob.glob(osp.join(path, '*.jpg')):
        filenames.append(fn)
    
    errors = 0

    with tqdm(total=len(filenames)) as pbar:
        for filename in filenames:
            basename = os.path.basename(filename)
            name = os.path.splitext(basename)[0]
            end_name = '{}{}.pt'.format(result_path, name)

            img = Image.open(filename)

            try: 
                img = rgb2lab(img)
                img_tensor = transform(img)
                torch.save(img_tensor, end_name)
            except ValueError:
                errors += 1

            pbar.update()
    print('total errors', errors)

def channel_to_class(a, b):
    a = (a + 110) // 10
    b = (b + 110) // 10
    c = classes[int(a), int(b)]
    return c

def tensor_to_classtensor():
    path = 'data/tree-training-1-tensor/'
    end_path = 'data/tree-training-1-classtensor/'
    filenames = []
    for fn in glob.glob(osp.join(path, '*.pt')):
        filenames.append(fn)
    
    with tqdm(total=len(filenames)) as pbar:
        for filename in filenames:
            basename = os.path.basename(filename)
            name = os.path.splitext(basename)[0]
            end_name = '{}{}_c.pt'.format(end_path, name)

            img_tensor = torch.load(filename)
            img_tensor = img_tensor.numpy()
            feat = img_tensor[0,:,:]
            v_channel_to_class = np.vectorize(channel_to_class)
            class_lab = v_channel_to_class(img_tensor[1,:,:], img_tensor[2,:,:])

            res = np.zeros((2, feat.shape[0], feat.shape[1]))
            res[0] = feat
            res[1] = class_lab

            torch.save(res, end_name)
            pbar.update()

# def to_one_hot(m):
#     one_hot = np.zeros((m.shape[0], m.shape[1], NO_CLASSES))
#     layer_idx = np.arange(m.shape[0]).reshape(m.shape[0], 1)
#     component_idx = np.tile(np.arange(m.shape[1]), (m.shape[0], 1))
#     one_hot[layer_idx, component_idx, m] = 1    
#     return one_hot

# transform_data()
tensor_to_classtensor()
