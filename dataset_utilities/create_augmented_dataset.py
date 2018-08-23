import numpy as np
from skimage import io
import os
import random
from tqdm import tqdm
PREV_DATASET_DIR = '../../datasets/128dim_slices/slices/'
AUGMENTED_DIR = '../../datasets/128dim_slices_augmented/slices/'
COLOR_DIR = '/color_slices/'
SCAN_DIR = '/scan_slices/'

scan_file_names = os.listdir(PREV_DATASET_DIR + SCAN_DIR)
color_file_names = os.listdir(PREV_DATASET_DIR + COLOR_DIR)

if len(scan_file_names)!= len(color_file_names):
	raise Exception("Inconsistent file numbers")
i = 0
for i, (scan, color) in tqdm(enumerate(zip(scan_file_names, color_file_names))):
	if scan != color:
		raise Exception('Inconsistent file names')

	scan_img = io.imread(PREV_DATASET_DIR+SCAN_DIR+scan)
	color_img = io.imread(PREV_DATASET_DIR+COLOR_DIR+color)
	choice = random.randint(0,3)

	io.imsave(AUGMENTED_DIR + SCAN_DIR + str(i) +'.png', scan_img)
	io.imsave(AUGMENTED_DIR + COLOR_DIR + str(i) + '.png', color_img)
	
	# print(choice)

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

	else:
		scan_rot = np.rot90(scan_img, 3)
		color_rot = np.rot90(color_img, 3)
		
		# i += 1

		io.imsave(AUGMENTED_DIR + SCAN_DIR + str(i) +'_rot270.png', scan_rot)
		io.imsave(AUGMENTED_DIR + COLOR_DIR + str(i) + '_rot270.png', color_rot)
	i += 1
	# break
print('done')