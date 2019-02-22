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

# DATA_DIR = '../data/sub_vol_outputs_slices/'
COLOR_DIR = '/color_slices/'
SCAN_DIR = '/scan_slices/'
LABEL_DIR = '/label_slices/'

transform = transforms.Compose([transforms.ToTensor()])

def process_images(DATA_DIR ,image, OUT_TYPE_DIR, color=True, random_state=-1):

    """Processes images in color volumes and converts to LAB color space
    :param: image_names : list of names of images
    :param: vols : list of volumes 
    :returns : images list containing the slices arrays corresponing to volumes
    """

    base_dir = DATA_DIR + OUT_TYPE_DIR
    
    if color:

        image = img_as_float(io.imread(base_dir + '/' + image))
        image_gray = skimage.color.rgb2gray(image)

    else:
        image = io.imread(base_dir + '/' + image)
        image = img_as_float(skimage.color.rgb2gray(image))
    if not color:
        if random_state == -1:
            pass
        elif random_state == 0:
            image = np.fliplr(image).copy()
        elif random_state == 1:
            image = np.flipud(image).copy()
        return image
    elif color:

        if random_state == -1:
            pass
        elif random_state == 0:
            image = np.fliplr(image).copy()
            image_gray = np.fliplr(image_gray).copy() 
        elif random_state == 1:
            image = np.flipud(image).copy()
            image_gray = np.flipud(image_gray).copy()
        return (image, image_gray)


def generate_train_test_split(DATA_DIR):
    color_images = np.array(sorted(os.listdir(DATA_DIR + COLOR_DIR)))
    scan_images = np.array(sorted(os.listdir(DATA_DIR + SCAN_DIR)))

    color_train, color_test = train_test_split(color_images, random_state=123)
    scan_train, scan_test = train_test_split(scan_images, random_state=123)
    return color_train, scan_train, color_test, scan_test



class EfficientImageDataSet(Dataset):
    def __init__(self, color_image_names, scan_image_names, DATA_DIR):

        self.color_image_names = color_image_names
        self.scan_image_names = scan_image_names
        
        self.DATA_DIR = DATA_DIR


    def __getitem__(self, index):
        colorimg_name = self.color_image_names[index]
        scanimg_name = self.scan_image_names[index]

        random_state = random.randint(-1, 1)

        scan_processed = torch.from_numpy(process_images(self.DATA_DIR, scanimg_name, SCAN_DIR, False, random_state)).float()
        scan_processed = scan_processed.unsqueeze(0).numpy()
        
        color, color_gray = process_images(self.DATA_DIR, colorimg_name, COLOR_DIR, True, random_state)
        color_processed = torch.from_numpy(color).float()
        
        color_processed = color_processed.numpy()

        color_gray_processed = torch.from_numpy(color_gray).float().unsqueeze(-1).numpy()
        
        return (colorimg_name, scanimg_name), torch.FloatTensor(scan_processed), transform(color_processed), transform(color_gray_processed)

    
    def __len__(self):
        return len(self.color_image_names)


class EfficientImageDataTestSet(Dataset):
    def __init__(self, color_image_names, scan_image_names, DATA_DIR):

        self.color_image_names = color_image_names
        self.scan_image_names = scan_image_names
        
        self.DATA_DIR = DATA_DIR


    def __getitem__(self, index):
        colorimg_name = self.color_image_names[index]
        scanimg_name = self.scan_image_names[index]

        random_state = random.randint(-1, 1)

        scan_processed = torch.from_numpy(process_images(self.DATA_DIR, scanimg_name, SCAN_DIR, False, random_state)).float()
        scan_processed = scan_processed.unsqueeze(0).numpy()
        
        color, color_gray = process_images(self.DATA_DIR, colorimg_name, COLOR_DIR, True, random_state)
        color_processed = torch.from_numpy(color).float()
        
        color_processed = color_processed.numpy()

        color_gray_processed = torch.from_numpy(color_gray).float().unsqueeze(-1).numpy()
        
        return (colorimg_name, scanimg_name), torch.FloatTensor(scan_processed), transform(color_processed), transform(color_gray_processed)


    def __len__(self):
        return len(self.color_image_names)



def create_dataloader(DATA_DIR, color, scan, batch_size=1, shuffle=True):
    
    dset = EfficientImageDataSet(color, scan, DATA_DIR)
    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
    return dataloader


def create_testdataloader(DATA_DIR, color, scan, batch_size=1, shuffle=False):
    
    dset = EfficientImageDataTestSet(color, scan, DATA_DIR)
    random_sampler =  RandomSampler(dset)
    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=1, sampler=random_sampler)
    return dataloader





if __name__ == '__main__':
    data_dir = sys.argv[1]
    color_train, scan_train, color_test, scan_test  = generate_train_test_split(data_dir)
    # print(color_train[:2])
    # print(scan_train[:2])
    # print(label_train[:2])
    # print(color_test[:2])
    # print(scan_test[:2])
    # print(label_test[:2])

    # print('X train:', X_train.shape)
    # print('y  train:', y_train.shape)
    # print('X test:', X_test.shape)
    # print('y  test :', y_test.shape)
    print('Train Dataloader')
    try_dataloader = create_dataloader(data_dir, color_train, scan_train, 2)
    for names, scan, color, cgray in cycle(try_dataloader):
        print(names)
        print('color:', color.shape)
        print('scan:', scan.shape)
        print('colored gray:', cgray.shape)
        import matplotlib.pyplot as plt
        plt.imshow(color[0].permute(1,2,0).cpu().numpy()); plt.show()
        plt.imshow(scan[0].permute(1,2,0).repeat(1,1,3).cpu().numpy()); plt.show()
        print(cgray[0].shape)
        plt.imshow(cgray[0].permute(1,2,0).repeat(1,1,3).cpu().numpy()); plt.show()
        break
    print('Test Dataloader')
    try_dataloader = create_testdataloader(data_dir, color_test, scan_test, 2)
    for names, scan, color, cgray in cycle(try_dataloader):
        print(names)
        print(color.shape)
        print(scan.shape)
        print(cgray.shape)

        import matplotlib.pyplot as plt
        plt.imshow(color[0].permute(1,2,0).cpu().numpy()); plt.show()
        plt.imshow(scan[0].permute(1,2,0).repeat(1,1,3).cpu().numpy()); plt.show()
        print(cgray[0].shape)
        plt.imshow(cgray[0].permute(1,2,0).repeat(1,1,3).cpu().numpy()); plt.show()
        # print(cgray.unique())
        break
    # for x, y in cycle(try_dataloader):
    #     print(x.shape)
    #     print(y.shape)
    #     # print(y[1].shape)
    #     # print('L channel ', y[:,0,:,:].unsqueeze(1).shape)
    #     # print('AB channel', y[:,1:,:,:].shape)
    #     break
    # try_dataloader = create_dataloader(data_dir, X_test, y_test)
    # for x, y in try_dataloader:
    #     print(x.shape)
    #     print(y.shape)
    #     # print(y[1].shape)
    #     break
    # try_dataloader = create_testdataloader(data_dir, X_test, y_test, 15)
    # for i, (name, x, y) in enumerate(try_dataloader):
    #     print(i)
    #     print(name)
    #     print(x.shape)
    #     # print(y[0].shape)
    #     print(y.shape)

    #     if i % 2 == 0 and i != 0:
    #         break