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

# Transform images to pytorch
# find . -name '*.jpg' -execdir mogrify -resize 224x224^ -gravity Center -extent 224x224 {} \;
# image -> 224 x 224 x 3 in CIE LAB
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

def tensor_to_classtensor():
    path = 'data/tree-training-1-tensor/'
    feat_path = 'data/tree-training-1-c-feat/'
    label_path = 'data/tree-training-1-c-label/'
    filenames = []

    for fn in glob.glob(osp.join(path, '*.pt')):
        filenames.append(fn)
    
    with tqdm(total=len(filenames)) as pbar:
        for filename in filenames:
            basename = os.path.basename(filename)
            name = os.path.splitext(basename)[0]
            feat_name = '{}{}_feat.pt'.format(feat_path, name)      
            label_name = '{}{}_label.pt'.format(label_path, name)

            img_tensor = torch.load(filename)
            img_tensor = img_tensor.numpy()
            features = img_tensor[0, :, :]
            labels = img_tensor[1:, :, :]

            # transforms labels to one-hot
            vect_channel_to_class = np.vectorize(channel_to_class)
            label = vect_channel_to_class(labels[0, :, :], labels[1, :, :])

            label = to_one_hot(label) # shape 224 x 224 x 484

            # save it as .pt
            features = torch.from_numpy(features)
            label = torch.from_numpy(label)

            torch.save(features, feat_name)
            torch.save(label, label_name)
              

def to_one_hot(m):
    one_hot = np.zeros((m.shape[0], m.shape[1], NO_CLASSES))
    layer_idx = np.arange(m.shape[0]).reshape(m.shape[0], 1)
    component_idx = np.tile(np.arange(m.shape[1]), (m.shape[0], 1))
    one_hot[layer_idx, component_idx, m] = 1    
    return one_hot

def channel_to_class(a, b):
    a = (a + 110) // 10
    b = (b + 110) // 10
    c = classes[int(a), int(b)]
    return c

# transform_data()
tensor_to_classtensor()
