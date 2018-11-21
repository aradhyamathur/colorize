
# import matplotlib
# matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import numpy as np
import tqdm
import argparse
import os
from model import AutoEncoder, Discriminator, EdgeLossLaplace3CHANNEL, EdgeLoss, EdgeLossSobel3Channel
from dataloader_rgb import *
import datetime
from itertools import cycle
import random
import gc
import skimage
from skimage.io import imsave
import resource
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from utils import *
from torchvision.utils import *
from torchvision import transforms




parser = argparse.ArgumentParser()
parser.add_argument('--data_path', help='path to data folder', required=True)
parser.add_argument('--image_dim', type=int, help='image dimensions', required=True)
parser.add_argument('--load_prev_model_gen', help='path to previous generator model')
parser.add_argument('--load_prev_model_disc', help='path to previous discriminator model')
parser.add_argument('--batch_size_train', type=int, help="train batch size")
parser.add_argument('--batch_size_test', type=int, help="test batch size")
parser.add_argument('--reset_files', help='reset file stats(True/False)')
parser.add_argument("--start_epoch", type=int, help="specify start epoch to continue from")
parser.add_argument("--end_epoch", type=int, help="specify end epoch to continue to")
parser.add_argument("--learning_rate_edge", type=float ,help="edge learning rate")
parser.add_argument("--learning_rate_gen", type=float ,help="generator learning rate")
parser.add_argument("--learning_rate_disc", type=float ,help="discriminator learning rate")
parser.add_argument("--test_mode", type=bool, help="run in test mode")
parser.add_argument("--device", nargs='?', const='cuda', type=str) 
parser.add_argument("--criterion_edge", nargs="?", const='laplace', type=str)
parser.add_argument("--custom_name", type=str, help='custom folder to save results in')
parser.add_argument("--description", type=str, help='training description')
parser.add_argument("--load_segmentation", type=str, help='training description')
args = parser.parse_args()

now = str(datetime.datetime.now())
custom_name = args.custom_name

if args.test_mode:
	now += '_test_mode'

if args.load_segmentation:
	now += '_segmented'

if not custom_name:
	now += '/'
else:
	now+='_' + custom_name + '/'

if args.test_mode:
	print('.....................................RUNNING IN TEST MODE................................')


LOG_DIR = './edge_gan_log_dir/'

np.set_printoptions(threshold=np.nan)

summary_writer = None 
if not args.device:
	device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
	device = torch.device(args.device)

SAVED_MODEL_DIR = './edge_gan_trained_models/'
RANDOM_OUTPUTS_DIR = './edge_gan_rand_outputs/'
EVAL_DIR = './edge_gan_test_outputs/'
EVAL_IMG_DIR = './edge_gan_test_output_images/'
TRAIN_FILE = RANDOM_OUTPUTS_DIR + now + '/'+ 'train_batch_info.txt'
TEST_FILE = EVAL_IMG_DIR + now + '/' + 'test_batch_info.txt'

if not os.path.exists(SAVED_MODEL_DIR):
	os.makedirs(SAVED_MODEL_DIR)
if not os.path.exists(RANDOM_OUTPUTS_DIR):
	os.makedirs(RANDOM_OUTPUTS_DIR)
if not os.path.exists(EVAL_IMG_DIR):
	os.makedirs(EVAL_IMG_DIR)
if not os.path.exists(LOG_DIR):
	os.makedirs(LOG_DIR)

BATCH_SIZE = 25

def train(model_g, model_d, learning_rate_gen, learning_rate_disc, learning_rate_edge, train_dataloader, test_dataloader, now):
	
	print("Total Train batches :", len(train_dataloader), "Total test batches:", len(test_dataloader))
	global summary_writer
	draw_iter = 100
	all_save_iter = 500
	cur_save_iter = 100
	test_iter = 1000

	if args.test_mode:
		draw_iter = 1
		all_save_iter = 1
		cur_save_iter = 1
		test_iter = 1

	cur_model_dir = SAVED_MODEL_DIR + now + '/'
	filenames = os.listdir(args.data_path + COLOR_DIR)
	if not os.path.exists(RANDOM_OUTPUTS_DIR + now):
		os.makedirs(RANDOM_OUTPUTS_DIR + now)
	f = open(TRAIN_FILE,'w')
	f.writelines('')
	f.close()
	if args.start_epoch:
		start_epoch = args.start_epoch
	else:start_epoch = 0

	if args.end_epoch:
		end_epoch = args.end_epoch
	else:end_epoch = 100
	
	# edge_detector = Edge(torch.cuda.is_available())

	
	# criterion = nn.BCELoss()
	criterion = nn.MSELoss()

	if args.criterion_edge == 'grad'  :
		criterion_edge = EdgeLoss(device)
	elif args.criterion_edge == 'laplace' or args.criterion_edge is None:
		# criterion_edge = EdgeLossLaplace3CHANNEL(device)
		criterion_edge = EdgeLossSobel3Channel(device)
	else:
		raise Exception('ValueError: Illegal criterion specified')
	optimizer_g = optim.Adam(model_g.parameters(), lr=learning_rate_gen, weight_decay=0.001) # wgan 
	optimizer_d = optim.Adam(model_d.parameters(), lr=learning_rate_disc, weight_decay=0.001) # wgan

	# optimizer_g = optim.Adam(model_g.parameters(), lr=learning_rate_gen, weight_decay=0.001) # normgan
	# optimizer_d = optim.Adam(model_d.parameters(), lr=learning_rate_disc, weight_decay=0.001) # normgan
	lowest = 0.0 
	save_model_info(model_g, model_d, learning_rate_gen, learning_rate_disc, cur_model_dir, start_epoch, end_epoch, learning_rate_edge, optimizer_g, optimizer_d, args.description) # to be changed
	# print(type(criterion_edge))
	for i in range(start_epoch, end_epoch):
		for j, (name, x, y) in enumerate(train_dataloader):

			model_g.train()
			model_d.train()
			# print(x.shape)
			target_y = torch.ones(len(y)).to(device)
			target_x = torch.zeros(len(y)).to(device)
			# noise = torch.normal(torch.zeros(x.shape), torch.ones(x.shape)*0.25)
			# print(x.max())
			# print(x.min())
			# print(y.max(), y.min())
			# save_image(x, RANDOM_OUTPUTS_DIR + now + 'cimg_' + str(i) +'_'+ str(j) + '_' + 'in.png')
			# save_image(y, RANDOM_OUTPUTS_DIR + now + 'cimg_' + str(i) +'_'+ str(j) + '_' + 'y.png')
			# exit()
			# x = x + torch.randn(x.shape)*1e-2 
			# x = x + noise
			x = x.to(device)
			y = y.to(device)
			edge_image_x = x.repeat(1,3, 1, 1)
			optimizer_g.zero_grad()

			for k in range(1):
				optimizer_d.zero_grad()
				out = model_g(x)
				#print(out.shape)
				#print(edge_image_x.shape)
				d_real = model_d(y)
				d_fake = model_d(out)
				# loss_edge, g1, g2 = criterion_edge(out, edge_image_x)
				d_loss_real = criterion(d_real.squeeze(1), target_y)
				d_loss_fake =  criterion(d_fake.squeeze(1), target_x) # add .squeeze for BCE LOSS
				d_l = 	 d_loss_fake + d_loss_real #GAN LOSS
				# d_l = -(torch.mean(d_real) - torch.mean(d_fake))  # wasserstein D loss
				d_loss = d_l
				d_loss.backward()
				optimizer_d.step()

			for p in model_d.parameters():
				p.data.clamp_(-3.0, 3.0)
			optimizer_d.zero_grad()
			for k in range(1):
				optimizer_g.zero_grad()

				out = model_g(x)
				d_fake = model_d(out)
				loss_edge, g1, g2 = criterion_edge(out, edge_image_x)
				# tv_loss = torch.sum(torch.abs(out[:, :, :, :-1] - out[:, :, :, 1:])) + torch.sum(torch.abs(out[:, :, :-1, :] - out[:, :, 1:, :]))
				# tv_loss = 1e-6*tv_loss

				g_loss = criterion(d_fake.squeeze(1), target_y) # GAN Loss
				# g_loss = -torch.mean(d_fake) # Wasserstein G loss
				loss_G = g_loss + loss_edge # * 1e-3
				# if lowest > loss_G:
				# 	lowest = loss_G
				loss_G.backward()
				optimizer_g.step()
			# print('done.......')
			# exit()

			value = 'Iter : %d Batch: %d Edge loss: %.4f G Loss: %.4f TV Loss: %.4f D Loss: %.4f Total Gloss: %.4f Total DLoss %.4f\n'%(i, j, loss_edge.item(), g_loss.item(), 0, d_l.item(), loss_G.item(), d_loss.item())
			print(value)	
			summary_writer.add_scalar("Edge Loss", loss_edge.item())
			summary_writer.add_scalar("Gen Loss", g_loss.item())
			summary_writer.add_scalar("Gen Loss Total", loss_G.item())
			summary_writer.add_scalar("Disc Loss", d_l.item())

			update_readings(cur_model_dir + 'train_loss_batch.txt', value)
			if j % draw_iter == 0:
				save_batch_image_names(name, TRAIN_FILE, j, i)

				save_image(x, RANDOM_OUTPUTS_DIR + now + 'cimg_' + str(i) +'_'+ str(j) + '_' + 'in.png')
				save_image(g1, RANDOM_OUTPUTS_DIR + now + 'cimg_' + str(i) +'_'+ str(j) + '_' + 'out_lap.png', normalize=True)
				save_image(g2, RANDOM_OUTPUTS_DIR + now + 'cimg_' + str(i) +'_'+ str(j) + '_' + 'in_lap.png', normalize=True)
				save_image(out, RANDOM_OUTPUTS_DIR + now +'cimg_' + str(i) +'_'+ str(j) + '_' + 'out.png')
				save_image(y, RANDOM_OUTPUTS_DIR + now +'cimg_' + str(i) +'_'+ str(j) + '_' + 'y.png')
				# draw_outputs(i, model, now, args.data_path, filenames, j)
			
			if j % all_save_iter == 0:
				print('..SAVING MODEL BATCH')
				torch.save(model_g.state_dict(), cur_model_dir + 'colorize2gen_batch_' + str(i) + '_{}.pt'.format(j))
				print('GEN SAVED ')
				torch.save(model_d.state_dict(), cur_model_dir + 'colorize2disc_batch_' + str(i) + '_{}.pt'.format(j))
				print('Disc SAVED')

			if j % cur_save_iter == 0:
				print('SAVING MODEL')
				torch.save(model_g.state_dict(), cur_model_dir + 'colorize_gen_cur.pt')
				torch.save(model_d.state_dict(), cur_model_dir + 'colorize_disc_cur.pt')
				# if loss_G < lowest:
				# 	torch.save(model_g.state_dict(), cur_model_dir + 'colorize_gen_cur_low.pt')
				print('SAVED CURRENT')

			if args.test_mode or (j % test_iter == 0 and j != 0) :

				test_losses = test_model(model_g, test_dataloader, i, now, j, criterion_edge)
				avg_test_loss = np.average(test_losses)
				summary_writer.add_scalar("Test loss", avg_test_loss)
				print('Test loss Avg: ', avg_test_loss)
				test_loss_val = '%d, %.4f\n' % (i, avg_test_loss)
				update_readings(cur_model_dir + 'test_loss_avg.txt', test_loss_val)

			if args.test_mode:
				break
		if args.test_mode:
			break

def test_model(model, test_loader, epoch, now, batch_idx, criterion_edge):

	global summary_writer

	if not os.path.exists(EVAL_IMG_DIR + now):
		os.makedirs(EVAL_IMG_DIR + now)
	
	f = open(TEST_FILE,'w')
	f.writelines('')
	f.close()
	
	model.eval()
	model.train_stat = False
	test_losses = []
	diffs_avg = []

	with torch.no_grad():
		for i, (name, x, y) in enumerate(test_loader):

			x = x.to(device)
			# y_l = y_l.to(device)

			# out = edge_detector(model(x).cpu())
			out = model(x)
			loss, g1, g2 = criterion_edge(out, x.repeat(1, 3, 1, 1))
			# tv_loss = torch.sum(torch.abs(out[:, :, :, :-1] - out[:, :, :, 1:])) + torch.sum(torch.abs(out[:, :, :-1, :] - out[:, :, 1:, :]))
			# tv_loss = 1e-6*tv_loss
			# loss = F.mse_loss(out, x_in)
			print('Test batch %d Edge Loss %.4f TV Loss %.4f'%(i, loss.item(), 0))

			test_losses.append(loss.item())

			out_sq = out.squeeze(1)
			output = out_sq.cpu().numpy()
			j = random.randint(0, len(output) - 1)
			
			# normalized_out = normalize(output[j])

			image_edge_in = g2[j].squeeze(0).cpu().numpy()
			# image_edge_in = normalize(image_edge_in)
			
			image_edge_out = g1[j].squeeze(0).cpu().numpy()
			# image_edge_out = normalize(image_edge_out)
			
			image = x[j].squeeze(0).cpu().numpy()

			if i % 100 == 0:
				# file_name = EVAL_IMG_DIR + now + '/' + 'cimg_' + str(epoch) + '_' + str(batch_idx)+ '_' + str(j) + '_'   + name[j]
				save_image(x, EVAL_IMG_DIR + now +'cimg_' + str(epoch) +'_'+ str(batch_idx) + '_'+ str(i)+'_' + 'in.png', normalize=True)
				save_image(g1, EVAL_IMG_DIR + now + 'cimg_' + str(epoch) +'_'+ str(batch_idx) + '_'+ str(i)+'_' + 'out_lap.png', normalize=True)
				save_image(g2, EVAL_IMG_DIR +  now +'cimg_' + str(epoch) +'_'+ str(batch_idx) + '_'+ str(i)+'_' + 'in_lap.png', normalize=True)
				save_image(out, EVAL_IMG_DIR + now + 'cimg_' + str(epoch) +'_'+ str(batch_idx) + '_'+ str(i)+'_' + 'out.png', normalize=True)
				
				save_batch_image_names(name, TEST_FILE, i, epoch)

			if args.test_mode:
				print('show image')

			if i !=0:
				# break
				if args.test_mode:
					break
			if args.test_mode:
				break
		# summary_writer.add_scalar('lab difference', np.average(diffs_avg))
	return test_losses


def main():
	global summary_writer


	cur_model_dir = SAVED_MODEL_DIR + now

	os.makedirs(cur_model_dir)

	summary_writer = SummaryWriter(LOG_DIR + now)

	if args.reset_files and (args.reset_files == 'False'):
		reset = False
	else:
		reset = True

	if reset:
		print('Resetting files')
		f = open(cur_model_dir + '/'+ 'train_loss_avg.txt','w')
		f.writelines('')
		f.close()
		f = open(cur_model_dir + '/'+ 'train_loss_batch.txt', 'w')
		f.writelines('')
		f.close()

		f = open(cur_model_dir + '/'+ 'test_loss_avg.txt','w')
		f.writelines('')
		f.close()
	else:
		print('no reset , appending to former data')



	if args.learning_rate_edge:
		learning_rate_edge = args.learning_rate_edge
	else:
		learning_rate_edge = 5e-4
	if args.learning_rate_gen:
		learning_rate_gen = args.learning_rate_gen
	else:
		learning_rate_gen = 1e-2
		# learning_rate_gen = 3e-3
	if args.learning_rate_disc:
		learning_rate_disc = args.learning_rate_disc
	else:
		learning_rate_disc = 1e-4
	

	batch_size_train = BATCH_SIZE
	batch_size_test = BATCH_SIZE
	if args.batch_size_train:
		batch_size_train = args.batch_size_train
	if args.batch_size_test:
		batch_size_test = args.batch_size_test 

	
	X_train, X_test, y_train, y_test = generate_train_test_split(args.data_path)
	train_dataloader = create_dataloader(args.data_path, X_train, y_train, batch_size_train)
	test_dataloader = create_testdataloader(args.data_path, X_test, y_test, batch_size_test)

	generator = AutoEncoder(out_channels=3)
	discriminator = Discriminator(128, 3)

	if args.load_prev_model_gen:
		generator.load_state_dict(torch.load(args.load_prev_model_gen))
		print('loaded generator successfully')
	if args.load_prev_model_disc:
		generator.load_state_dict(torch.load(args.load_prev_model_disc))
		print('loaded discriminator successfully')

	generator = generator.to(device)
	discriminator = discriminator.to(device)

	if args.load_segmentation:
		print('Loading segmentation model')
		path = args.load_segmentation
		generator = load_external(generator, path)

	generator = nn.DataParallel(generator)
	discriminator = nn.DataParallel(discriminator)

	train(generator, discriminator, learning_rate_gen, learning_rate_disc, learning_rate_edge, train_dataloader, test_dataloader, now)


if __name__ == '__main__':
	main()
