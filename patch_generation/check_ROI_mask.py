import os 
import openslide
import cv2
import numpy as np
import time
from xml.etree.ElementTree import parse
import matplotlib.pyplot as plt

def get_ROI_mask(slide, xml):
    """
    inputs
        slide: WSI (Whole Slide Image), slide = openslide.OpenSlide('/~/*.svs')
        xml: ROI (Region of Interest), xml = parse('/~/*.xml').getroot()

    get a ROI mask using a pair of an WSI (slide) and the corresponding ROI annotation (xml) 
    it processes on the lowest resolution (level 3) of the WSI 

    outputs
        slide_thumbnail (np.array): a rgb image 
        ROI mask (np.array): a gray image (binary mask)
    """

    if slide.level_count < 4:
        level_index = slide.level_count-1
    else:
        level_index = 3

    slide_thumbnail = slide.get_thumbnail(slide.level_dimensions[level_index])
    slide_thumbnail = slide_thumbnail.convert('RGB')
    
    mask_slide_ratio = round(slide.level_downsamples[level_index])

    annotations = []
    patterns = []

    for anno in xml.iter('Annotation'):
        pattern = anno.get('class')
        patterns.append(pattern)
        annotation = []
        for i, coors in enumerate(anno):
            if i == 0: 
                continue
            coordinates = []
            for coor in coors:
                coordinates.append([round(float(coor.get('x'))//mask_slide_ratio), round(float(coor.get('y'))//mask_slide_ratio)])

            annotation.append(coordinates)
        annotations.append(annotation)

    width, height = slide.level_dimensions[level_index]
    ROI_mask = np.zeros((height, width)).astype(np.uint8)

    for anno in annotations:
        _anno = []
        for coors in anno:
            _anno.append(np.array(coors))

        cv2.drawContours(ROI_mask, _anno, -1, 255, -1)

    return np.array(slide_thumbnail), ROI_mask

if __name__ == "__main__":

    slide_dir = '/mnt/nfs0/jycho/SLIDE_DATA/록원재단/AT2/C-MET_slide'
    ROI_dir = '/mnt/hdd1/c-MET_datasets/SLIDE_DATA/ROI_annotation' 
    plot_save_dir = '/mnt/hdd1/c-MET_datasets/SLIDE_DATA/ROI_annotation/check'

    os.makedirs(plot_save_dir, exist_ok = True)

    issues = [118]
    slide_list = sorted([svs for svs in os.listdir(slide_dir) if 'svs' in svs and int(svs.split('-')[1][2:]) not in issues])
    ROI_list = sorted([xml for xml in os.listdir(ROI_dir) if 'xml' in xml and int(xml.split('-')[1][2:]) not in issues])

    total_time = 0
    for i, (slide_file, ROI_file) in enumerate(zip(slide_list, ROI_list)):
        start_time = time.time()
        
        if slide_file[:-4] != ROI_file[:-4]:
            print("Check the pairness between slide and ROI annotation")
            break
        
        slide_name = slide_file[:-4]
        slide_path = os.path.join(slide_dir, slide_file)
        ROI_path = os.path.join(ROI_dir, ROI_file)

        slide = openslide.OpenSlide(slide_path)
        ROI = parse(ROI_path).getroot()
        
        slide_thumbnail, ROI_mask = get_ROI_mask(slide, ROI)

        plt.figure(figsize=(30, 20))
        plt.imshow(slide_thumbnail)
        plt.imshow(ROI_mask, alpha = 0.3)
        plt.savefig(f'{plot_save_dir}/{slide_name}_ROI_overlay.jpg', bbox_inches = 'tight')
        plt.close()

        end_time = time.time()
        taken = end_time - start_time
        print(f'{slide_name} | size: {ROI_mask.shape} | time: {round(taken, 2)} sec')
        total_time += taken

    print(f'total time: {round(total_time, 2)} sec')