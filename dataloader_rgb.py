import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import ipdb
import time
from itertools import cycle
import sys
from torch.utils.data.sampler import RandomSampler
from skimage import  io
import skimage
from skimage import util
from skimage import img_as_float
from torchvision import transforms
import random
random.seed(123)
# DATA_DIR = '../data/sub_vol_outputs_slices/'
COLOR_DIR = '/color_slices/'
SCAN_DIR = '/scan_slices/'


transform = transforms.Compose([transforms.ToTensor()])
def process_images(DATA_DIR ,image, OUT_TYPE_DIR, color=True):
    """Processes images in color volumes and converts to LAB color space
    :param: image_names : list of names of images
    :param: vols : list of volumes 
    :returns : images list containing the slices arrays corresponing to volumes
    """

    base_dir = DATA_DIR + OUT_TYPE_DIR
    
    if color:
        # image = cv2.imread(base_dir + '/' + image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = img_as_float(io.imread(base_dir + '/' + image))

        # print('color:', image.dtype)
        # image = image.astype(np.int8)
        # image = image[:, :, 1:]  # keeping all the channels dropping the luminosity
    else:
        # image = cv2.imread(base_dir + '/' + image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = io.imread(base_dir + '/' + image)
        image = img_as_float(skimage.color.rgb2gray(image))
        # image = util.invert(image)
        # print('gray : ', image.dtype)


    
    
    return image

def generate_train_test_split(DATA_DIR):
    color_images = np.array(os.listdir(DATA_DIR + COLOR_DIR))
    scan_images = np.array(os.listdir(DATA_DIR + SCAN_DIR))

    c=[]
    s= []
    indices = list(range(len(color_images)))
    random.shuffle(indices)
    for i in range(80000):
        c.append(color_images[indices[i]])
        s.append(scan_images[indices[i]])

    X_train, X_test, y_train, y_test = train_test_split(s, c, test_size=0.2, random_state=123)
    return X_train, X_test, y_train, y_test



class EfficientImageDataSet(Dataset):
    def __init__(self, X, y, DATA_DIR):

        self.X = X
        self.y = y
        self.DATA_DIR = DATA_DIR


    def __getitem__(self, index):
        img_name = self.X[index]
        
        x_processed = torch.from_numpy(process_images(self.DATA_DIR, img_name, SCAN_DIR, False)).float()
        
        x_processed = x_processed.unsqueeze(0).permute(1, 2, 0).numpy()
        
        y_processed = torch.from_numpy(process_images(self.DATA_DIR, img_name, COLOR_DIR)).float()
        # y_processed = y_processed.unsqueeze(0)
        
        y_processed = y_processed.numpy()

        return img_name, transform(x_processed), transform(y_processed)

    
    def __len__(self):
        return len(self.X)

class EfficientImageDataTestSet(Dataset):
    def __init__(self, X, y, DATA_DIR):

        self.X = X
        self.y = y
        self.DATA_DIR = DATA_DIR


    def __getitem__(self, index):
        img_name = self.X[index]
        
        x_processed = torch.from_numpy(process_images(self.DATA_DIR, img_name, SCAN_DIR, False)).float()
        x_processed = x_processed.unsqueeze(0).permute(1, 2, 0)
        
        
        y_processed = torch.from_numpy(process_images(self.DATA_DIR, img_name, COLOR_DIR)).float()
        
        y_processed = y_processed.numpy()
        x_processed = x_processed.numpy()
        
        return img_name, transform(x_processed), transform(y_processed)

    
    def __len__(self):
        return len(self.X)



def create_dataloader(DATA_DIR, X, y, batch_size=1, shuffle=True):
    
    dset = EfficientImageDataSet(X, y, DATA_DIR)
    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    return dataloader


def create_testdataloader(DATA_DIR, X, y, batch_size=1, shuffle=False):
    
    dset = EfficientImageDataTestSet(X, y, DATA_DIR)
    random_sampler =  RandomSampler(dset)
    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=2, sampler=random_sampler)
    return dataloader





if __name__ == '__main__':
    data_dir = sys.argv[1]
    X_train, X_test, y_train, y_test = generate_train_test_split(data_dir)
    print('X train:', X_train.shape)
    print('y  train:', y_train.shape)
    print('X test:', X_test.shape)
    print('y  test :', y_test.shape)

    try_dataloader = create_dataloader(data_dir, X_train, y_train,  2)
    for x, y in cycle(try_dataloader):
        print(x.shape)
        print(y.shape)
        # print(y[1].shape)
        # print('L channel ', y[:,0,:,:].unsqueeze(1).shape)
        # print('AB channel', y[:,1:,:,:].shape)
        break
    try_dataloader = create_dataloader(data_dir, X_test, y_test)
    for x, y in try_dataloader:
        print(x.shape)
        print(y.shape)
        # print(y[1].shape)
        break
    try_dataloader = create_testdataloader(data_dir, X_test, y_test, 15)
    for i, (name, x, y) in enumerate(try_dataloader):
        print(i)
        print(name)
        print(x.shape)
        # print(y[0].shape)
        print(y.shape)

        if i % 2 == 0 and i != 0:
            break