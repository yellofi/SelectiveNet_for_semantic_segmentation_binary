cd /mnt/ssd1/biomarker/c-met/algorithm/develop/patch_generation

python3 get_patch.py --ROI_dir /mnt/ssd1/biomarker/c-met/data/LOGONE_AT2/ROI_annotation/annotation --save_dir /mnt/ssd1/biomarker/c-met/data/LOGONE_AT2/patch --patch_mag 200 --patch_size 256

python3 get_patch.py --ROI_dir /mnt/ssd1/biomarker/c-met/data/LOGONE_AT2/ROI_annotation/annotation --label_dir /mnt/ssd1/biomarker/c-met/data/LOGONE_AT2/tumor_annotation/annotation --save_dir /mnt/ssd1/biomarker/c-met/data/LOGONE_AT2/patch --patch_mag 200 --patch_size 1024