cd ./c-MET/c-met_yellofi/patch_generation

python3 get_patch.py --label_dir '/mnt/hdd1/c-MET_datasets/SLIDE_DATA/tumor_annotation/annotation' --save_dir '/mnt/hdd1/c-MET_datasets/SLIDE_DATA/DL-based_tumor_seg_dataset/2205_1차anno/patch' --patch_size 256

python3 get_patch.py --label_dir '/mnt/hdd1/c-MET_datasets/SLIDE_DATA/tumor_annotation/annotation' --save_dir '/mnt/hdd1/c-MET_datasets/SLIDE_DATA/DL-based_tumor_seg_dataset/2205_1차anno/patch' --patch_size 512

python3 get_patch.py --label_dir '/mnt/hdd1/c-MET_datasets/SLIDE_DATA/tumor_annotation/annotation' --save_dir '/mnt/hdd1/c-MET_datasets/SLIDE_DATA/DL-based_tumor_seg_dataset/2205_1차anno/patch' --patch_size 1024



python3 get_patch.py --save_dir '/mnt/hdd1/c-MET_datasets/SLIDE_DATA/DL-based_tumor_seg_dataset/2205_1차anno/label_level/0' --patch_size 1024 --label_level 0

python3 get_patch.py --save_dir '/mnt/hdd1/c-MET_datasets/SLIDE_DATA/DL-based_tumor_seg_dataset/2205_1차anno/label_level/1' --patch_size 1024 --label_level 1

python3 get_patch.py --save_dir '/mnt/hdd1/c-MET_datasets/SLIDE_DATA/DL-based_tumor_seg_dataset/2205_1차anno/label_level/2' --patch_size 1024 --label_level 2

python3 get_patch.py --save_dir '/mnt/hdd1/c-MET_datasets/SLIDE_DATA/DL-based_tumor_seg_dataset/2205_1차anno/label_level/3' --patch_size 1024 --label_level 3
