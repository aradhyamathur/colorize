run:
	python main2.py --data_path ../datasets/128dim_slices/slices/ --image_dim 128
test:
	python main2.py --data_path ../datasets/128dim_slices/slices/ --image_dim 128 --test_mode True