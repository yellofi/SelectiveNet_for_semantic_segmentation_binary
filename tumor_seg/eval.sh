cd /mnt/ssd1/biomarker/c-met/algorithm/develop/tumor_region_seg

python3 eval.py --data_dir /mnt/ssd1/biomarker/c-met/data/LOGONE_AT2/DL_dataset/2205_anno --model_path /mnt/ssd1/biomarker/c-met/output/model/06_baseline_samsung_data/1-fold/checkpoint/model_epoch197.pth --save_dir /mnt/ssd1/biomarker/c-met/output/model/06_baseline_samsung_data/1-fold/output --local_rank 0 1 2 3 --batch_size 320 --test_fold 1