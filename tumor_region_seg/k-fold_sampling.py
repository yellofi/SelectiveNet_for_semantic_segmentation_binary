import os
from posixpath import basename
import cv2
import numpy as np
from sklearn.model_selection import KFold as KF

def split_data(img_list):
    
    total_n = len(img_list)
    
    test_idx = np.random.choice(total_n, size = int(total_n*0.2), replace = False)
    train_idx = np.setdiff1d(np.array([i for i in range(total_n)]), test_idx)
    valid_idx = train_idx[np.random.choice(len(train_idx), size = int(total_n*0.2), replace = False)]
    train_idx = np.setdiff1d(train_idx, valid_idx)

    return train_idx, valid_idx, test_idx

def extract_sample(img_files, mask_files, save_dir, mode = 'train', sample_size = 256, overlap_rate = 0):

    for img, mask in zip(img_files, mask_files):
        img_path = os.path.join(img_dir, img)
        mask_path = os.path.join(mask_dir, mask)

        img_name = os.path.basename(img_path)[:-4]
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        height, width = mask.shape
        
        points = []
        for i in range(width//int(sample_size*(1-overlap_rate))):
            for j in range(height//int(sample_size*(1-overlap_rate))):
                points.append((i, j))

        try: os.makedirs(os.path.join(save_dir, f'{mode}'))
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

                cv2.imwrite(os.path.join(save_dir, f'{mode}', f'{img_name}_{i:03d}_sample.png'), img_)
                cv2.imwrite(os.path.join(save_dir, f'{mode}', f'{img_name}_{i:03d}_label.png'), mask_)

                image_ = cv2.rectangle(image_, (x_coor, y_coor), 
                                (x_coor+sample_size, y_coor+sample_size), color=(0, 255, 0), thickness=2)

        print(f'{img_name} | {i} samples')
        
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

    kf = KF(n_splits = 5, shuffle = True, random_state = 44)
    # kf = KF(n_splits = 5, shuffle = False)

    sample_size = 256
    tr_overlap_rate = 0.5
    test_overlap_rate = 0

    for i, (train_idx, test_idx) in enumerate(kf.split(img_list)):
        print("TRAIN:", train_idx, "TEST:", test_idx)
        
        _save_dir = os.path.join(save_dir, f'{i+1}-fold')

        try: os.makedirs(_save_dir)
        except: pass

        print("TRAIN")
        extract_sample(img_files= np.asarray(img_list)[train_idx], 
                    mask_files = np.asarray(mask_list)[train_idx], 
                    save_dir = _save_dir, mode = 'train',
                    sample_size = sample_size, 
                    overlap_rate = tr_overlap_rate)

        print("TEST")
        extract_sample(img_files= np.asarray(img_list)[test_idx], 
                    mask_files = np.asarray(mask_list)[test_idx], 
                    save_dir = _save_dir, mode = 'valid',
                    sample_size = sample_size, 
                    overlap_rate = test_overlap_rate)

        print(f'# of training dataset = {len(os.listdir(_save_dir + "/train"))//2}')
        print(f'# of testing dataset = {len(os.listdir(_save_dir + "/valid"))//2}\n')