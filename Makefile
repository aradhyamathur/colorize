run:
	python main2.py --data_path ../datasets/128dim_slices/slices/ --image_dim 128
test:
	python main2.py --data_path ../datasets/128dim_slices/slices/ --image_dim 128 --test_mode True
dataloader:
	python dataloader_rgb.py ../datasets/randSlice/subslices/

test_ae:
	python main_autoencoder.py --data_path ../datasets/128dim_slices/slices/  --image_dim 128 --test_mode True
	#python main_autoencoder.py --data_path ../color_my_pytorch_slices/data/slices --image_dim 128 --test_mode True

run_ae:
	python main_autoencoder.py --data_path ../datasets/128dim_slices/slices/  --image_dim 128
	# python main_autoencoder.py --data_path ../color_my_pytorch_slices/data/slices --image_dim 128

test_edge:
	python main_edge_detect.py --data_path ../datasets/128dim_slices/slices/  --image_dim 128 --test_mode True --device cuda
	# python main_edge_detect.py --data_path ../color_my_pytorch_slices/data/slices  --image_dim 128 --test_mode True --device cuda

run_edge:
	python main_edge_detect.py --data_path ../datasets/128dim_slices/slices/  --image_dim 128 --device cuda
	# python main_edge_detect.py --data_path ../color_my_pytorch_slices/data/slices  --image_dim 128 --device cuda

test_edge_gan:
	python main_edge_gan.py --data_path ../datasets/random_axis/subslice_f/  --image_dim 128 --test_mode True --device cuda --custom_name random_axis_wasserstein_lite_test_sobel_randslice --description __ 
	#--load_segmentation "../segmentation vol_generate/data/128dim_slices/fcn/weights.pth"
	# python main_edge_gan.py --data_path ../datasets/128dim_slices_augmented/slices/  --image_dim 128 --test_mode True --device cuda --custom_name wasserstein_lite
	# python main_edge_detect.py --data_path ../color_my_pytorch_slices/data/slices  --image_dim 128 --test_mode True --device cuda

run_edge_gan:
	 python main_edge_gan.py --data_path ../datasets/random_axis/subslice_f/  --image_dim 128 --device cuda --custom_name random_axis_wasserstein_lite_sobel --description sobel_with_edge_no_tv_gen_iter_2 
	# python main_edge_gan.py --data_path ../datasets/random_axis/subslice_f/  --image_dim 128 --device cuda --custom_name random_axis_wasserstein_lite_sobel --description sobel_with_edge_no_tv_gen_iter_2_hard_tanh_disc 
	#--load_segmentation "../segmentation vol_generate/data/128dim_slices/fcn/weights.pth"
# 	python main_edge_gan.py --data_path ../datasets/128dim_slices/slices/  --image_dim 128 --device cuda --custom_name wasserstein_lite_lap4_edge_mult_1_gfast_weight_clip_5_point_0
	# python main_edge_gan.py --data_path ../datasets/128dim_slices_augmented/slices/  --image_dim 128 --device cuda --custom_name wasserstein_lite
	# python main_edge_detect.py --data_path ../color_my_pytorch_slices/data/slices  --image_dim 128 --device cuda


test_edge_detect_script:
	python edge_detect_conv.py

evaluate_images:
	python evaluate_image.py  --load_prev_model_gen edge_gan_trained_models/2018-11-27\ 22:12:54.163338_wasserstein_lite_sobel_randslice/colorize2gen_batch_0_5000.pt --input_dir ../datasets/volumes/Slices/scan/  --color_dir ../datasets/volumes/Slices/color/ --custom_name _randslice_withtv_batch_0_5000_complete_dsets  --image_scale 1.0