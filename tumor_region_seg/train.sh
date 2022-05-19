cd ./c-MET/c-met_yellofi/tumor_region_seg

python3 train.py --data_type 'samsung' --fold 1 --local_rank 0 1 2 3 4 5 6 7 --n_epoch 100 --batch_size 144