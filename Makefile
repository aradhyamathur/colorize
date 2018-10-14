run:
	python main2.py --data_path ../datasets/128dim_slices/slices/ --image_dim 128
test:
	python main2.py --data_path ../datasets/128dim_slices/slices/ --image_dim 128 --test_mode True
dataloader:
	python dataloader_efficient.py ../datasets/128dim_slices/slices/

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
	python main_edge_gan.py --data_path ../datasets/128dim_slices/slices/  --image_dim 128 --test_mode True --device cuda --custom_name wasserstein_lite --description _lap4_gfast_weight_clip_3_point_0_edge_1point0_tv_loss_1eminus8
	# python main_edge_gan.py --data_path ../datasets/128dim_slices_augmented/slices/  --image_dim 128 --test_mode True --device cuda --custom_name wasserstein_lite
	# python main_edge_detect.py --data_path ../color_my_pytorch_slices/data/slices  --image_dim 128 --test_mode True --device cuda

run_edge_gan:

	python main_edge_gan.py --data_path ../datasets/128dim_slices_augmented/slices/  --image_dim 128 --device cuda --custom_name wasserstein_lite --description _lap4_gfast_weight_clip_3_point_0_edge_1point0_tv_loss_1eminus6_ae_model_layer2removed_layer1_512
# 	python main_edge_gan.py --data_path ../datasets/128dim_slices/slices/  --image_dim 128 --device cuda --custom_name wasserstein_lite_lap4_edge_mult_1_gfast_weight_clip_5_point_0
	# python main_edge_gan.py --data_path ../datasets/128dim_slices_augmented/slices/  --image_dim 128 --device cuda --custom_name wasserstein_lite
	# python main_edge_detect.py --data_path ../color_my_pytorch_slices/data/slices  --image_dim 128 --device cuda


test_edge_detect_script:
	python edge_detect_conv.py

evaluate_images:
	python evaluate_image.py  --load_prev_model_gen edge_gan_trained_models/2018-10-11\ 00:12:03.096879_wasserstein_lite/colorize2gen_1.pt --input_dir ../datasets/volumes/Slices/scan/  --color_dir ../datasets/volumes/Slices/color/ --custom_name _epoch_0_scale_point_0.5  --image_scale 0.5