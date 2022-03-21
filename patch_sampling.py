import os
from posixpath import basename
import cv2

img_dir = '/mnt/hdd1/c-MET_datasets/Lung_c-MET IHC_scored/sample'
mask_dir = os.path.join(img_dir, 'annotation')
save_dir = '/mnt/hdd1/c-MET_datasets/Lung_c-MET IHC_scored/DL-based_tumor_seg_dataset'

def extract_sample(img_path, mask_path, sample_size, overlap_rate = 0):

    img_name = os.path.basename(img_path)[:-4]
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    height, width = mask.shape
    
    points = []
    for i in range(width//int(sample_size*(1-overlap_rate))):
        for j in range(height//int(sample_size*(1-overlap_rate))):
            points.append((i, j))

    _save_dir = os.path.join(save_dir, img_name)
    try: os.makedirs(_save_dir)
    except: print(f'{_save_dir} Already exists')

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

            cv2.imwrite(os.path.join(_save_dir, f'{img_name}_{i:03d}_sample.png'), img_)
            cv2.imwrite(os.path.join(_save_dir, f'{img_name}_{i:03d}_label.png'), mask_)

            image_ = cv2.rectangle(image_, (x_coor, y_coor), 
                            (x_coor+sample_size, y_coor+sample_size), color=(0, 255, 0), thickness=2)

    print(f'# of extracted sample: {i}')
    cv2.imwrite(os.path.join(save_dir, f'{img_name}_size-{sample_size}_overlap-{overlap_rate}.png'), image_)


if __name__ == '__main__':
    img_list = sorted([img for img in os.listdir(img_dir) if '.png' in img])
    mask_list = sorted([mask for mask in os.listdir(mask_dir) if '.png' in mask])

    print(f'image directory: {img_dir}')
    print(f'mask directory: {mask_dir}')

    for i, (img, mask) in enumerate(zip(img_list, mask_list)):
        print(f'{i} | image: {img} | mask: {mask}')
        
        img_path = os.path.join(img_dir, img)
        mask_path = os.path.join(mask_dir, mask)

        extract_sample(img_path, mask_path, sample_size = 256, overlap_rate = 0.5)


