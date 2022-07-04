# c-MET

## patch generation

- ROI mask: tissue region to be analyzed (excluding artifacts such as internal control, black lines, dust, and so on)
- tissue mask: otsu's thresholding -> sobel edge detection
- blurrity check: variance of gray-scale image fed into laplacian filter 
- thresholds: tissue threshlod = 0.1 / variance for blurrity check = 100

tissue mask
<img src = './patch_generation/output/S-LC0007-MET_tissue_mask.jpg'>

patch extraction (200x)
<img src = './patch_generation/output/S-LC0007-MET_200x_tissue_intensity-0.1_blur_th-100_num-1910.jpg'>

## tumor segmentation

- U-Net
- data collection and annotation

patch (200x, 1024x1024) / heatmap overlay / predicted tumor region 
<img src = './tumor_region_seg/output/S-LC0007-MET_67584_77824_prediction_overlay.jpg'>