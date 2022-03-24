import os
from posixpath import basename
import cv2
import numpy as np


def split_data(img_list):
    
    total_n = len(img_list)
    
    test_idx = np.random.choice(total_n, size = int(total_n*0.2), replace = False)
    train_idx = np.setdiff1d(np.array([i for i in range(total_n)]), test_idx)
    valid_idx = train_idx[np.random.choice(len(train_idx), size = int(total_n*0.2), replace = False)]
    train_idx = np.setdiff1d(train_idx, valid_idx)

    return train_idx, valid_idx, test_idx

def extract_sample(img_path, mask_path, save_dir, sample_size, overlap_rate = 0):

    img_name = os.path.basename(img_path)[:-4]
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    height, width = mask.shape
    
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
            mask_ = mask[y_coor:y_coor+sample_size, \
                        x_coor:x_coor+sample_size]

            cv2.imwrite(os.path.join(save_dir, f'{img_name}_{i:03d}_sample.png'), img_)
            cv2.imwrite(os.path.join(save_dir, f'{img_name}_{i:03d}_label.png'), mask_)

            image_ = cv2.rectangle(image_, (x_coor, y_coor), 
                            (x_coor+sample_size, y_coor+sample_size), color=(0, 255, 0), thickness=2)

    print(f'# of extracted sample: {i}')
    
    _save_dir = os.path.join(save_dir, 'visualization')
    try: os.makedirs(_save_dir)
    except: pass
    
    cv2.imwrite(os.path.join(_save_dir, f'{img_name}_size-{sample_size}_overlap-{overlap_rate}.png'), image_)

img_dir = '/mnt/hdd1/c-MET_datasets/Lung_c-MET IHC_scored/sample'
mask_dir = os.path.join(img_dir, 'annotation')
save_dir = '/mnt/hdd1/c-MET_datasets/Lung_c-MET IHC_scored/DL-based_tumor_seg_dataset'

if __name__ == '__main__':
    img_list = sorted([img for img in os.listdir(img_dir) if '.png' in img])
    mask_list = sorted([mask for mask in os.listdir(mask_dir) if '.png' in mask])

    try: bool(img_list == mask_list)
    except: print("Check each pair of image and mask")        
    
    print(f'image directory: {img_dir}')
    print(f'mask directory: {mask_dir}')

    train_idx, valid_idx, test_idx = split_data(img_list)

    print(f'train_idx ({len(train_idx)}) = {train_idx}')
    print(f'valid_idx ({len(valid_idx)}) = {valid_idx}')
    print(f'test_idx ({len(test_idx)}) = {test_idx}')
    
    sample_size = 256
    tr_overlap_rate = 0.5
    test_overlap_rate = 0

    print("")
    print(f'training dataset | size = {sample_size}x{sample_size} | overlap = {tr_overlap_rate}')
    for i, t_idx in enumerate(train_idx):
        img, mask = img_list[t_idx], mask_list[t_idx]
        print(f'{i+1} | image: {img} | mask: {mask}')
        img_path, mask_path = os.path.join(img_dir, img), os.path.join(mask_dir, mask)
        extract_sample(img_path, mask_path, save_dir + '/train', sample_size = sample_size, overlap_rate = tr_overlap_rate)

    print("")
    print(f'validation dataset | size = {sample_size}x{sample_size} | overlap = {test_overlap_rate}')
    for i, v_idx in enumerate(valid_idx):
        img, mask = img_list[v_idx], mask_list[v_idx]
        print(f'{i+1} | image: {img} | mask: {mask}')
        img_path, mask_path = os.path.join(img_dir, img), os.path.join(mask_dir, mask)
        extract_sample(img_path, mask_path, save_dir + '/valid', sample_size = sample_size, overlap_rate = test_overlap_rate)

    print("")
    print(f'testing dataset | size = {sample_size}x{sample_size} | overlap = {test_overlap_rate}')
    for i, t_idx in enumerate(test_idx):
        img, mask = img_list[t_idx], mask_list[t_idx]
        print(f'{i+1} | image: {img} | mask: {mask}')
        img_path, mask_path = os.path.join(img_dir, img), os.path.join(mask_dir, mask)
        extract_sample(img_path, mask_path, save_dir + '/test', sample_size = sample_size, overlap_rate = test_overlap_rate)

    print("")
    print(f'# of training dataset = {(len(os.listdir(save_dir + "/train"))-1)//2}')
    print(f'# of validation dataset = {(len(os.listdir(save_dir + "/valid"))-1)//2}')
    print(f'# of testing dataset = {(len(os.listdir(save_dir + "/test"))-1)//2}')