import os
from posixpath import basename
import cv2
import numpy as np

def extract_sample(img_path, save_dir, sample_size, overlap_rate = 0):

    img_name = os.path.basename(img_path)[:-4]
    image = cv2.imread(img_path)

    height, width, _ = image.shape
    
    points = []
    for i in range(width//int(sample_size*(1-overlap_rate))):
        for j in range(height//int(sample_size*(1-overlap_rate))):
            points.append((i, j))

    try: os.makedirs(save_dir)
    except: pass

    image_ = image.copy()
    
    i = 0
    for (x, y) in points:

        x_coor = int(x*sample_size*(1-overlap_rate))
        y_coor = int(y*sample_size*(1-overlap_rate))

        if x_coor+sample_size <= width and y_coor+sample_size <= height:
            i += 1

            img_ = image[y_coor:y_coor+sample_size, \
                        x_coor:x_coor+sample_size, \
                        :]

            cv2.imwrite(os.path.join(save_dir, f'{img_name}_{i:03d}_sample.png'), img_)

            image_ = cv2.rectangle(image_, (x_coor, y_coor), 
                            (x_coor+sample_size, y_coor+sample_size), color=(0, 255, 0), thickness=2)

    print(f'# of extracted sample: {i}')
    
    _save_dir = os.path.join(save_dir, 'visualization')
    try: os.makedirs(_save_dir)
    except: pass
    
    cv2.imwrite(os.path.join(_save_dir, f'{img_name}_size-{sample_size}_overlap-{overlap_rate}.png'), image_)

img_dir = '/mnt/hdd1/c-MET_datasets/SLIDE_DATA/록원재단/AT2/C-MET_slide/patch/S-LC0001-MET/x200'
save_dir = '/mnt/hdd1/c-MET_datasets/SLIDE_DATA/DL-based_tumor_seg_dataset'

if __name__ == '__main__':

    img_list = sorted([img for img in os.listdir(img_dir) if '.jpg' in img])
    
    print(f'image directory: {img_dir}')
    
    sample_size = 256
    test_overlap_rate = 0

    print("")
    print(f'testing dataset | size = {sample_size}x{sample_size} | overlap = {test_overlap_rate}')
    for i, img in enumerate(img_list):
        print(f'{i+1} | image: {img}')
        img_path = os.path.join(img_dir, img)
        extract_sample(img_path, save_dir + '/test', sample_size = sample_size, overlap_rate = test_overlap_rate)

    print("")
    print(f'# of testing dataset = {(len(os.listdir(save_dir + "/test"))-1)//2}')