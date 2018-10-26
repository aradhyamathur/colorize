import numpy as np
from skimage import io
import os
import random
from tqdm import tqdm
from skimage.transform import warp, AffineTransform
import math
from skimage.transform import rescale

PREV_DATASET_DIR = '../../datasets/128dim_slices/slices/'
AUGMENTED_DIR = '../../datasets/128dim_slices_augmented_affine/slices/'
COLOR_DIR = '/color_slices/'
SCAN_DIR = '/scan_slices/'

scan_file_names = os.listdir(PREV_DATASET_DIR + SCAN_DIR)
color_file_names = os.listdir(PREV_DATASET_DIR + COLOR_DIR)

class RandomAffineTransform(object):
    def __init__(self,
                 scale_range,
                 rotation_range,
                 shear_range,
                 translation_range
                 ):
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.shear_range = shear_range
        self.translation_range = translation_range

    def modify(self, img_g, img_rgb):
        img_g = img_g[32:96, 32:96]
        img_g = rescale(img_g, 2.0)
            
        img_rgb = img_rgb[32:96, 32:96]
        img_rgb = rescale(img_rgb, 2.0)
        # flip upside down
        choice = random.randint(0,2)
        if choice == 0:
            pass
        elif choice == 1:
            img_rgb = np.flipud(img_rgb)
            img_g = np.flipud(img_g)
        elif choice == 2:
            img_rgb = np.fliplr(img_rgb)
            img_g = np.fliplr(img_g)


        # print(img_g.shape)
        # print(img_rgb.shape)

        # exit()

        img_data = np.array(img_rgb)
        h, w, n_chan = img_data.shape
        scale_x = np.random.uniform(*self.scale_range)
        scale_y = np.random.uniform(*self.scale_range)
        scale = (scale_x, scale_y)

        rotation = np.random.uniform(*self.rotation_range)
        shear = np.random.uniform(*self.shear_range)
        translation = (
            np.random.uniform(*self.translation_range) * w,
            np.random.uniform(*self.translation_range) * h
        )
        # print('Scale ', scale, 'rotation', rotation, 'translation', translation)
        af = AffineTransform(scale=scale, rotation=rotation, translation=translation, shear=shear)
        img_scan = warp(img_g, af.inverse)
        img_color = warp(img_rgb, af.inverse)
        return (img_scan*255.0).astype(np.uint8), (img_color*255).astype(np.uint8)


if len(scan_file_names)!= len(color_file_names):
    raise Exception("Inconsistent file numbers")

transformer = RandomAffineTransform(scale_range=(1.5, 0.5), rotation_range=(math.pi/6, 0), shear_range=(math.pi/6, -math.pi/6), translation_range=(0.2, 0.5))
for i, (scan, color) in tqdm(enumerate(zip(scan_file_names, color_file_names))):
    if scan != color:
        raise Exception('Inconsistent file names')
    print('########## Creating for {} #########'.format(i))
    scan_img = io.imread(PREV_DATASET_DIR+SCAN_DIR+scan)
    color_img = io.imread(PREV_DATASET_DIR+COLOR_DIR+color)
    io.imsave(AUGMENTED_DIR + SCAN_DIR + str(i) +'_orig.png', scan_img)
    io.imsave(AUGMENTED_DIR + COLOR_DIR + str(i) + '_orig.png', color_img)

    for k in range(4):
        s,c = transformer.modify(scan_img, color_img)
        io.imsave(AUGMENTED_DIR + SCAN_DIR + str(i) + '_t{}.png'.format(k), s)
        io.imsave(AUGMENTED_DIR + COLOR_DIR + str(i) + '_t{}.png'.format(k), c)

    choice = random.randint(0,4)
    
    if choice == 0:
        pass
    elif choice == 1:
        scan_flip = np.fliplr(scan_img)
        color_flip = np.fliplr(color_img)
        
        # i += 1

        io.imsave(AUGMENTED_DIR + SCAN_DIR + str(i) +'_flip.png', scan_flip)
        io.imsave(AUGMENTED_DIR + COLOR_DIR + str(i) + '_flip.png', color_flip)
    
    elif choice == 2:
        scan_rot = np.rot90(scan_img)
        color_rot = np.rot90(color_img)
        
        # i += 1

        io.imsave(AUGMENTED_DIR + SCAN_DIR + str(i) +'_rot90.png', scan_rot)
        io.imsave(AUGMENTED_DIR + COLOR_DIR + str(i) + '_rot90.png', color_rot)

    elif choice==3:
        scan_rot = np.rot90(scan_img, 3)
        color_rot = np.rot90(color_img, 3)
    
        io.imsave(AUGMENTED_DIR + SCAN_DIR + str(i) +'_rot270.png', scan_rot)
        io.imsave(AUGMENTED_DIR + COLOR_DIR + str(i) + '_rot270.png', color_rot)
    
    else:
        scan_ud = np.flipud(scan_img)
        color_ud = np.flipud(color_img)
        io.imsave(AUGMENTED_DIR + SCAN_DIR + str(i) +'_ud.png', scan_ud)
        io.imsave(AUGMENTED_DIR + COLOR_DIR + str(i) + '_ud.png', color_ud)

        # i += 1

    # break


    # break
print('done')