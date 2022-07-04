python3 inference.py --slide_name S-LC0027-MET.svs --ROI_dir /mnt/ssd1/biomarker/c-met/data/LOGONE_AT2/ROI_annotation/annotation --model_dir /mnt/ssd1/biomarker/c-met/algorithm/pipeline/model/sample_NT_ensemble --ens_scale 'minmax' --batch_size 8 --local_rank 0 1 2 3 --save_dir /mnt/ssd1/biomarker/c-met/final_output/sample_NT_ensemble

python3 inference.py --slide_name S-LC0027-MET.svs --ROI_dir /mnt/ssd1/biomarker/c-met/data/LOGONE_AT2/ROI_annotation/annotation --model_dir /mnt/ssd1/biomarker/c-met/algorithm/pipeline/model/samsung --batch_size 8 --local_rank 0 1 2 3 --save_dir /mnt/ssd1/biomarker/c-met/final_output/xml/samsung

python3 inference.py --ROI_dir /mnt/ssd1/biomarker/c-met/data/LOGONE_AT2/ROI_annotation/annotation --save_dir /mnt/ssd1/biomarker/c-met/final_output/patch_grid

python3 inference.py --ROI_dir /mnt/ssd1/biomarker/c-met/data/LOGONE_AT2/ROI_annotation/annotation --model_dir /mnt/ssd1/biomarker/c-met/algorithm/pipeline/model/sample_NT_ensemble --ens_scale 'minmax' --batch_size 8 --local_rank 0 1 2 3 --save_dir /mnt/ssd1/biomarker/c-met/final_output/sample_NT_ensemble