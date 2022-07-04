cd /mnt/ssd1/biomarker/c-met/algorithm/develop/tumor_seg

python3 train.py --data_type 'samsung' --fold 1 --data_dir '/mnt/ssd1/biomarker/c-met/data/LOGONE_AT2/DL_dataset/2205_anno' --model_dir '/mnt/ssd1/biomarker/c-met/tumor_seg/model/06_selective_samsung_data' --model_arch 'SelectiveUNet' --selective 1 --loss 'CE' --local_rank 0 1 2 3 4 5 6 7 --n_epoch 200 --batch_size 160 

python3 train.py --data_type 'samsung' --fold 1 --data_dir '/mnt/ssd1/biomarker/c-met/data/LOGONE_AT2/DL_dataset/2205_anno' --model_dir '/mnt/ssd1/biomarker/c-met/tumor_seg/model/06_UNet_samsung_data' --model_arch 'UNet' --loss 'CE' --local_rank 0 1 2 3 4 5 6 7 --n_epoch 200 --batch_size 120 