cd ./c-MET/c-met_yellofi/tumor_region_seg

python3 train.py --data_type 'samsung' --fold 1 --local_rank 0 1 2 3 4 5 6 7 --n_epoch 200 --batch_size 144 --model_dir '/mnt/hdd1/model/Lung_c-MET IHC_scored/UNet/06_baseline_samsung_data'

python3 train.py --data_type 'samsung' --fold 1 --local_rank 0 1 2 3 4 5 --n_epoch 200 --batch_size 144 --data_dir '/mnt/ssd1/biomarker/c-met/data/LOGONE_AT2/DL_dataset/2205_anno' --model_dir '/mnt/ssd1/biomarker/c-met/tumor_seg/model/06_baseline_samsung_data'