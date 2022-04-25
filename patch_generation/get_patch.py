import argparse
import os
import openslide
from xml.etree.ElementTree import parse
import cv2
import numpy as np
from random import shuffle
from model_laplacian import *
from multiprocessing import Pool, Manager
import time

model = load_model()
model = model.cuda()

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--slide_dir', action="store", type=str,
                        default='/mnt/nfs0/jycho/SLIDE_DATA/록원재단/AT2/C-MET_slide', help='WSI data (.svs) directory')
    parser.add_argument('--ROI_dir', action="store", type=str,
                        default='/mnt/hdd1/c-MET_datasets/SLIDE_DATA/ROI_annotation', help='ROI annotation (.xml) directory')
    parser.add_argument('--save_dir', action="store", type=str,
                        default='/mnt/hdd1/c-MET_datasets/SLIDE_DATA/록원재단/AT2/C-MET_slide/patch_on_ROI', help='directory where it will save patches')

    parser.add_argument('--mpp', action="store", type=int,
                        default=0.2532, help='millimeter per pixel')
    parser.add_argument('--wsl_mag', action="store", type=int,
                        default=400, help='the magnification of WSI images')

    parser.add_argument('--patch_mag', action="store", nargs='+', type=int,
                        default=[200], help='target magnifications of generated patches')
    parser.add_argument('--patch_size', action="store", type=int,
                        default=1024, help='a width/height length of squared patches')
    parser.add_argument('--tissue_th', action="store", type=float,
                        default=0.5, help='threshold about proportion of valued-region of each patch_ mask')
    parser.add_argument('--blur_th', action="store", type=int,
                        default=500, help='threshold about blurrity of each patch_')

    args = parser.parse_args()
    print('')
    print('args={}\n'.format(args))

    return args

def get_ROI_mask(slide, xml):
    """
    inputs
        slide: WSI (Whole Slide Image), slide = openslide.OpenSlide('/~/*.svs')
        xml: ROI (Region of Interest), xml = parse('/~/*.xml').getroot()

    get a ROI mask using a pair of an WSI (slide) and the corresponding ROI annotation (xml) 
    it processes on the lowest resolution (level 3) of the WSI 

    output
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

    return ROI_mask

def make_tissue_mask(slide, ROI_mask):
    """
    inputs
        slide: WSI (Whole Slide Image), slide = openslide.OpenSlide('/~/*.svs')
        ROI mask (np.array): ROI (Region of Interest) mask, a binary mask whose size is the same with the lowest resolution (level 3) of slide

    get a tissue mask on ROI, using Otsu's thresholding in HSV channel

    RGB2HSV -> S, V_inv (1 - V) -> S' = Otsu(S), V_inv' = Otsu(V_inv) 
    -> S" = MorpOp(S'), V_inv" MorpOp(V_inv') ->  tissue mask = OR(S", V_inv")

    output
        tissue mask (np.array): a gray image (binary mask)
    """

    if slide.level_count < 4:
        level_index = slide.level_count-1
    else:
        level_index = 3

    slide_thumbnail = slide.get_thumbnail(slide.level_dimensions[level_index]) 
    slide_thumbnail = slide_thumbnail.convert('RGB') 

    mask_slide_ratio = round(slide.level_downsamples[level_index])

    otsu_image = cv2.cvtColor(np.array(slide_thumbnail), cv2.COLOR_BGR2HSV)

    otsu_image_1 = otsu_image[:, :, 1]
    otsu_image_2 = 1 - otsu_image[:, :, 2]

    # on ROI 
    otsu_image_1 = cv2.bitwise_and(otsu_image_1, ROI_mask)
    otsu_image_2 = cv2.bitwise_and(otsu_image_2, ROI_mask)

    otsu_image_1 = cv2.threshold(otsu_image_1, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    otsu_image_2 = cv2.threshold(otsu_image_2, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    kernel = np.ones((11, 11), dtype=np.uint8)

    otsu_image_1 = cv2.morphologyEx(otsu_image_1, cv2.MORPH_CLOSE, kernel)
    otsu_image_1 = cv2.morphologyEx(otsu_image_1, cv2.MORPH_OPEN, kernel)
    otsu_image_2 = cv2.morphologyEx(otsu_image_2, cv2.MORPH_CLOSE, kernel)
    otsu_image_2 = cv2.morphologyEx(otsu_image_2, cv2.MORPH_OPEN, kernel)

    otsu_image = np.logical_or(otsu_image_1, otsu_image_2).astype(float)*255

    return otsu_image, mask_slide_ratio

def make_mask(slide):
    """ make tissue mask using HSV mode from openslide object.

    Args:
        slide (openslide object): slide

    Return:
        tissue_mask (np.array): output tissue mask.
        mask_slide_ratio (int): relative down magnification when compared to
            original size.

    """

    if slide.level_count < 4:
        level_index = slide.level_count-1
    else:
        level_index = 3

    slide_thumbnail = slide.get_thumbnail(slide.level_dimensions[level_index]) 
    slide_thumbnail = slide_thumbnail.convert('RGB') 

    mask_slide_ratio = round(slide.level_downsamples[level_index])

    slide_array = np.array(slide_thumbnail, dtype=np.uint8)

    kernel_size = 11
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    patch = slide_thumbnail.convert('HSV')
    patch = np.array(patch)
    patch = patch.astype(np.float32)

    h_test = np.zeros((patch.shape[0], patch.shape[1]))
    h_test[patch[:, :, 0] > 180] = 1
    h_test[patch[:, :, 0] < 20] = 1
    h_test[patch[:, :, 0] == 0] = 0

    s_test = np.zeros((patch.shape[0], patch.shape[1]))
    s_test[patch[:, :, 1] < 240] = 1
    s_test[patch[:, :, 1] < 20] = 0

    v_test = np.zeros((patch.shape[0], patch.shape[1]))
    v_test[patch[:, :, 2] > 30] = 1

    black_test = np.zeros((patch.shape[0], patch.shape[1]))
    patch = np.array(slide_array, float)
    patch_new = patch[:, :, 0] + patch[:, :, 1] + patch[:, :, 2]
    patch_new = patch_new / 3
    black_test[patch_new > 100] = 1

    patch_mask = h_test * s_test * v_test * black_test * 255

    tissue_mask = cv2.morphologyEx(patch_mask, cv2.MORPH_CLOSE, kernel)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)

    return tissue_mask.astype(float), mask_slide_ratio

def _make_patch(patch):
    patch = np.array(patch)
    patch = patch.astype(np.float32)
    patch = np.expand_dims(patch, axis=0)
    patch = np.ascontiguousarray(patch, dtype=np.float32)
    return patch

def read_regions_semi(params):

    queue, queue2, points, patch_size, window, slide_path, tissue_mask, tissue_threshold, mask_slide_ratio, mpp_ratio = params

    slide = openslide.OpenSlide(slide_path)
    step_size = patch_size//window

    target_step_size = mpp_ratio*step_size
    mask_step_size = target_step_size//mask_slide_ratio
    
    try:
        for idx, (x, y) in enumerate(points):
            try:
                patch_tissue = tissue_mask[y*mask_step_size:(y+window)*mask_step_size, x*mask_step_size:(x+window)*mask_step_size]
                tissue_region_ratio = round(patch_tissue.sum()/(((mask_step_size)**2)*255), 3) 
                
                if tissue_region_ratio < tissue_threshold:
                    continue
            except:
                continue
            queue2.put(1)

            patch_img = slide.read_region((x*target_step_size, y*target_step_size), 0, (target_step_size, target_step_size)).resize((patch_size, patch_size))

            patch_ = patch_img.convert('L').resize((patch_size//4, patch_size//4))
            patch_ = _make_patch(patch_)
            while True:
                if queue.full():
                    time.sleep(0.01) 
                else:
                    queue.put((patch_img, patch_, x*target_step_size, y*target_step_size))
                    break
        return True
    
    except Exception as e:
        return e

def generate_patch(args, slide_file, ROI_file, target_mag = 200):
    # slide_name, _organ, mpp, slide_dir, _, _, _, _, _ = line
    
    slide_name = slide_file[:-4]
    slide_path = args.slide_dir + '/' + slide_file
    ROI_path = args.ROI_dir + '/' + ROI_file

    # _save_dir = os.path.join(args.save_dir, 'patch', slide_name)
    _save_dir = os.path.join(args.save_dir, slide_name)

    try: os.makedirs(os.path.join(_save_dir))
    except: print(f"{os.path.join(_save_dir)} already exists")

    slide = openslide.OpenSlide(slide_path)
    ROI = parse(ROI_path).getroot()
    width, height = slide.dimensions
    slide_mag, mpp = int(args.wsl_mag), float(args.mpp)

    ROI_mask = get_ROI_mask(slide, ROI)
    tissue_mask, mask_slide_ratio = make_tissue_mask(slide, ROI_mask)

    # cv2.imwrite(args.save_dir + '/slide' + f'/{slide_name}_{target_mag}x_tissue_mask.jpg', tissue_mask)

    if target_mag == 400:
        tissue_th = 0.3
    elif target_mag == 200:
        tissue_th = 0.1
    else:
        tissue_th = args.tissue_th
    
    try: os.mkdir(os.path.join(_save_dir, f'x{target_mag}'))
    except: print(f"{os.path.join(_save_dir, f'x{target_mag}')} already exists")

    sliding_window = 1
    patch_size = args.patch_size
    mpp_ratio = slide_mag//target_mag

    total_point = []
    for i in range(sliding_window * width//(patch_size*mpp_ratio)):
        for j in range(sliding_window * height//(patch_size*mpp_ratio)):
            total_point.append((i, j))
    
    # shuffle(total_point)
    num_total_patch = len(total_point)
    print(f"{slide_name} | {target_mag}x | # of patch: {num_total_patch}")

    n_process = 12
    queue_size = 15 * n_process
    queue = Manager().Queue(queue_size)
    queue2 = Manager().Queue(num_total_patch)
    
    pool = Pool(n_process)
    split_points = []
    for ii in range(n_process):
        split_points.append(total_point[ii::n_process])

    result = pool.map_async(read_regions_semi, [(queue, queue2, allocated_points, patch_size, sliding_window, slide_path,
                                                 tissue_mask, tissue_th, mask_slide_ratio, mpp_ratio)
                                                for allocated_points in split_points])

    slide_thumbnail = slide.get_thumbnail((tissue_mask.shape[1], tissue_mask.shape[0])) 
    slide_thumbnail = slide_thumbnail.convert('RGB')
    slide_ = np.array(slide_thumbnail, dtype=np.uint8)

    mask_step_size = (mpp_ratio*(patch_size//sliding_window))//mask_slide_ratio

    batch_size = 64
    blur_th = args.blur_th 
    img_list, point_list, patch_list = [], [], []

    count = 0
    while True:
        if queue.empty():
            if not result.ready():
                time.sleep(0.5)
            elif result.ready() and len(patch_list) == 0:
                break
        else:
            img, patch_, x, y = queue.get()
            patch_list.append(patch_)
            point_list.append((x, y))
            img_list.append(img)
            
            if len(patch_list) == batch_size or \
            (result.ready() and queue.empty() and len(patch_list) > 0):

                with torch.autograd.no_grad():
                    batch = torch.FloatTensor(np.stack(patch_list)).cuda()
                    output = model(batch)
                output = output.cpu().data.numpy()
                output = np.var(output, axis=(1, 2, 3))
                for ii in range(len(patch_list)):
                    _x, _y = point_list[ii]
                    img = img_list[ii]
                    if output[ii] > blur_th:
                        img.convert('RGB').save(os.path.join(_save_dir, 'x{}'.format(target_mag), slide_name + '_' + str(_x)+'_'+str(_y)+'.jpg'))
                        slide_ = cv2.rectangle(slide_, (_x//mask_slide_ratio, _y//mask_slide_ratio), 
                        (_x//mask_slide_ratio+mask_step_size, _y//mask_slide_ratio+mask_step_size), color=(0, 255, 0), thickness=2)
                        count += 1

                img_list, point_list, patch_list = [], [], []
    if not result.successful():
        print('Something wrong in result')
    cv2.imwrite(args.save_dir + f'/{slide_name}_{target_mag}x_tissue_th-{tissue_th}_blur_th-{blur_th}_num-{count}.jpg', cv2.cvtColor(slide_, cv2.COLOR_BGR2RGB))
    print(f'# of actual saved patch_: {count}')
    pool.close()
    pool.join()
    
if __name__ == "__main__":

    args = parse_arguments()
    # issues = [7, 17, 25, 33, 39, 42, 43, 48, 53, 55, 69, 87, 89, 91, 92, 98, 102, 104, 112]
    issues = [118]
    slide_list = sorted([svs for svs in os.listdir(args.slide_dir) if 'svs' in svs and int(svs.split('-')[1][2:]) not in issues])
    ROI_list = sorted([xml for xml in os.listdir(args.ROI_dir) if 'xml'in xml and int(xml.split('-')[1][2:]) not in issues]) 
    total_time = 0
    for i, (slide_file, ROI_file) in enumerate(zip(slide_list, ROI_list)):
        for target_mag in args.patch_mag:
            start_time = time.time()
            if slide_file[:-4] != ROI_file[:-4]:
                print("Check the pairness between slide and ROI files")
                break
            generate_patch(args, slide_file, ROI_file, target_mag)
            end_time = time.time()
            taken = end_time - start_time
            print(f'time: {round(taken, 2)} sec')
            total_time += taken

    print(f'total time: {round(total_time, 2)} sec')