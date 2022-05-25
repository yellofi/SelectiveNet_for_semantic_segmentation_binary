import argparse
import os
import openslide
import numpy as np
from xml.etree.ElementTree import parse
from random import shuffle
from model_laplacian import *
from slide_utils import *
from multiprocessing import Pool, Manager
import time

model = load_model()
# model = model.cuda()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = torch.nn.DataParallel(model, device_ids=[i for i in range(8)])
# model = model.to(device)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--slide_dir', action="store", type=str,
                        default='/mnt/nfs0/jycho/SLIDE_DATA/록원재단/AT2/C-MET_slide', help='WSI data (.svs) directory')
    parser.add_argument('--ROI_dir', action="store", type=str,
                        default='/mnt/hdd1/c-MET_datasets/SLIDE_DATA/ROI_annotation', help='rough tissue region annotation (.xml) directory')
    parser.add_argument('--label_dir', action="store", type=str,
                        default='/mnt/hdd1/c-MET_datasets/SLIDE_DATA/tumor_annotation/annotation', help='tumor annotation (.xml) directory')

    parser.add_argument('--save_dir', action="store", type=str,
                        default='/mnt/hdd1/c-MET_datasets/SLIDE_DATA/록원재단/AT2/C-MET_slide/patch', help='directory where it will save patches')

    parser.add_argument('--tissue_mask_type', action="store", type=str,
                        default='sobel', help="otsu, sobel, ...")

    parser.add_argument('--patch_mag', action="store", nargs='+', type=int,
                        default=[200], help='target magnifications of generated patches')

    parser.add_argument('--patch_size', action="store", type=int,
                        default=1024, help='a width/height length of squared patches')
    parser.add_argument('--tissue_th', action="store", type=float,
                        default=0.1, help='threshold at tissue mask (proportion at otsu, intensity at sobel)')
    parser.add_argument('--blur_th', action="store", type=int,
                        default=100, help='threshold about blurrity of each patch_')

    parser.add_argument('--label_level', action="store", type=int, 
                        default = 1)

    args = parser.parse_args()
    print('')
    print('args={}\n'.format(args))

    return args

def read_regions_semi(params):

    queue, queue2, points, patch_size, window, slide_path, tissue_mask, tissue_threshold, slide_mask_ratio, mpp_ratio = params

    slide = openslide.OpenSlide(slide_path)
    step_size = patch_size//window

    slide_step_size = mpp_ratio*step_size
    mask_step_size = slide_step_size//slide_mask_ratio
    
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

            patch_img = slide.read_region((x*slide_step_size, y*slide_step_size), 0, (slide_step_size, slide_step_size)).resize((patch_size, patch_size))

            patch_ = patch_img.convert('L').resize((patch_size//4, patch_size//4))
            patch_ = make_patch_(patch_)
            while True:
                if queue.full():
                    time.sleep(0.01) 
                else:
                    queue.put((patch_img, patch_, x*slide_step_size, y*slide_step_size))
                    break
        return True
    
    except Exception as e:
        return e

def generate_patch(args, slide_file, ROI_file = None, label_file = None, target_mag = 200):
    
    slide_name = slide_file[:-4]
    slide_path = args.slide_dir + '/' + slide_file

    slide = openslide.OpenSlide(slide_path)
    # print(slide.level_downsamples)
    width, height = slide.dimensions

    slide_mag = float(slide.properties['openslide.objective-power']) 
    slide_mag = int(slide_mag*10)

    try: mpp =  float(slide.properties['openslide.mpp-x']) #svs, ndpi, mrxs, tif(Roche)
    except: mpp = 10000 / float(slide.properties['tiff.XResolution']) # tif

    # ROI_mask -> target tissue 영역 annotation
    # tumor_label -> target tissue 영역 안의 tumor annotation
    # non_target_label -> target tissue 영역 안의 non target object annotation (benign에도 비포함) 
    ROI_mask, tumor_label, non_target_label, label_level = None, None, None, None
    if ROI_file != None:
        ROI_path = args.ROI_dir + '/' + ROI_file
        RAN = Annotation(slide = slide, level = -1)
        ROI_annotations, _ = RAN.get_coordinates(xml_path = ROI_path, target = 'tissue_region')
        ROI_mask = RAN.make_mask(annotations=ROI_annotations, color = 255)
        # cv2.imwrite(args.save_dir + f'/{slide_name}_ROI_mask.jpg', ROI_mask)

    if label_file != None:
        label_path = args.label_dir + '/' + label_file
        label_level = args.label_level
        TAN = Annotation(slide = slide, level = label_level)
        tumor_annotations, non_target_annotations = TAN.get_coordinates(xml_path = label_path, target = 'tumor_region')
        tumor_label = TAN.make_mask(annotations=tumor_annotations, color = 255)
        # cv2.imwrite(args.save_dir + f'/{slide_name}_tumor_mask.jpg', tumor_label)
        if len(non_target_annotations) != 0:
            non_target_label = TAN.make_mask(annotations=non_target_annotations, color = 255)
            # cv2.imwrite(args.save_dir + f'/{slide_name}_non_target_mask.jpg', non_target_label)

    TM = TissueMask(slide = slide, level = -1, ROI_mask = ROI_mask, NOI_mask = non_target_label)
    tissue_mask, slide_mask_ratio = TM.get_mask_and_ratio(tissue_mask_type = args.tissue_mask_type)
    # cv2.imwrite(args.save_dir + f'/{slide_name}_tissue_mask.jpg', tissue_mask)

    if not os.path.isfile(args.save_dir + f'/{slide_name}_tissue_mask.jpg'):
        cv2.imwrite(args.save_dir + f'/{slide_name}_tissue_mask.jpg', tissue_mask)

    if target_mag > slide_mag:
        print(f'You can extract patches at lower magnification (slide magnification: {slide_mag}')
        return
    
    # if target_mag == 400:
    #     tissue_th = 0.3
    # elif target_mag == 200:
    #     tissue_th = 0.1

    sliding_window = 1
    patch_size = args.patch_size
    mpp_ratio = slide_mag//target_mag

    total_point = []
    for i in range(sliding_window * width//(patch_size*mpp_ratio)):
        for j in range(sliding_window * height//(patch_size*mpp_ratio)):
            total_point.append((i, j))
    
    # shuffle(total_point)
    num_total_patch = len(total_point)
    print(f"{slide_mag}x, mpp: {mpp} |=> patch mag: {target_mag}x, size: {patch_size}, # of total coordinates: {num_total_patch}")

    batch_size = 64
    tissue_th = args.tissue_th
    blur_th = args.blur_th 
    step_size = patch_size//sliding_window
    slide_step_size = mpp_ratio*step_size

    mask_step_size = slide_step_size//slide_mask_ratio

    if label_level != None:
        slide_label_ratio = round(slide.level_downsamples[label_level])
        label_step_size = slide_step_size//slide_label_ratio

        # print('slide mag:', slide_mag, 'patch mag:', target_mag, 'label level:', label_level, 'actual side length on label:', label_step_size)

    n_process = 12
    queue_size = 15 * n_process
    queue = Manager().Queue(queue_size)
    queue2 = Manager().Queue(num_total_patch)
    
    pool = Pool(n_process)
    split_points = []
    for ii in range(n_process):
        split_points.append(total_point[ii::n_process])

    result = pool.map_async(read_regions_semi, [(queue, queue2, allocated_points, patch_size, sliding_window, slide_path,
                                                 tissue_mask, tissue_th, slide_mask_ratio, mpp_ratio)
                                                for allocated_points in split_points])

    slide_thumbnail = slide.get_thumbnail((tissue_mask.shape[1], tissue_mask.shape[0])) 
    slide_thumbnail = slide_thumbnail.convert('RGB')
    slide_ = np.array(slide_thumbnail, dtype=np.uint8)

    img_list, point_list, patch_list = [], [], []

    count = 0

    _save_dir = os.path.join(args.save_dir, slide_name)
    
    os.makedirs(_save_dir, exist_ok = True)
    
    patch_save_dir = os.path.join(_save_dir, f'{target_mag}x_{patch_size}')
    try: os.mkdir(patch_save_dir)
    except: print(f"{patch_save_dir} already exists")

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
                    # batch = torch.FloatTensor(np.stack(patch_list)).cuda()
                    batch = torch.FloatTensor(np.stack(patch_list))
                    output = model(batch)

                # output = output.cpu().data.numpy()
                output = output.data.numpy()
                output = np.var(output, axis=(1, 2, 3))
                for ii in range(len(patch_list)):
                    _x, _y = point_list[ii]
                    img = img_list[ii]
                    if output[ii] > blur_th:
                        img.convert('RGB').save(os.path.join(patch_save_dir, slide_name + '_' + str(_x)+'_'+str(_y)+'_input.jpg'))
                        slide_ = cv2.rectangle(slide_, (_x//slide_mask_ratio, _y//slide_mask_ratio), 
                        (_x//slide_mask_ratio+mask_step_size, _y//slide_mask_ratio+mask_step_size), color=(0, 255, 0), thickness=2)

                        if type(tumor_label) != type(None):
                            img_label = tumor_label[_y//slide_label_ratio:_y//slide_label_ratio+label_step_size, 
                                                        _x//slide_label_ratio:_x//slide_label_ratio+label_step_size]

                            img_label = cv2.resize(img_label, (patch_size, patch_size))
                            img_label = ((img_label >= 0.5)*255).astype('uint8')

                            # binary는 jpg로 저장하면 0, 255가 아니고 불러왔을 때 {0, 1, 2, 3, 4, 5, 250, 251, 252, 253, 254, 255} 값이 나옴
                            # png는 0, 255 값을 보존하고 오히려 jpg로 저장했을 때보다 용량이 적게 차지 (통상적으로 png가 jpg보다 용량이 큼)
                            cv2.imwrite(os.path.join(patch_save_dir, slide_name + '_' + str(_x)+'_'+str(_y)+'_label.png'), 
                                        img_label)
                        
                        count += 1
                img_list, point_list, patch_list = [], [], []
    if not result.successful():
        print('Something wrong in result')

    cv2.imwrite(args.save_dir + f'/{slide_name}_{target_mag}x_{patch_size}_tissue_th-{tissue_th}_blur_th-{blur_th}_num-{count}.jpg', 
                cv2.cvtColor(slide_, cv2.COLOR_BGR2RGB)) 

    print(f'# of actual saved patch: {count}')
    pool.close()
    pool.join()
    
if __name__ == "__main__":

    args = parse_arguments()
    # issues = [7, 17, 25, 33, 39, 42, 43, 48, 53, 55, 69, 87, 89, 91, 92, 98, 102, 104, 112]
    issues = [118]
    slide_list = sorted([svs for svs in os.listdir(args.slide_dir) if 'svs' in svs and int(svs.split('-')[1][2:]) not in issues])

    
    # target_slides = [27, 32, 47, 59, 80, 87, 90, 94, 106, 107] # 1차 annotation
    # # target_slides = [80]
    # slide_list = sorted([svs for svs in os.listdir(args.slide_dir) if 'svs' in svs and int(svs.split('-')[1][2:]) in target_slides])

    print(slide_list)

    total_time = 0
    for i, (slide_file) in enumerate(slide_list):
        print(slide_file[:-4])

        slide_name = slide_file[:-4]
        ROI_file, label_file = None, None
        if os.path.isfile(os.path.join(args.ROI_dir, f'{slide_name}.xml')):
            ROI_file = f'{slide_name}.xml'
        if os.path.isfile(os.path.join(args.label_dir, f'{slide_name}.xml')):
            label_file = f'{slide_name}.xml'

        print("ROI_file:", ROI_file, "label_file:", label_file)

        for target_mag in args.patch_mag:
            start_time = time.time()
            generate_patch(args, slide_file, ROI_file, label_file, target_mag)
            end_time = time.time()
            taken = end_time - start_time
            print(f'time: {round(taken, 2)} sec')
            total_time += taken

    print(f'total time: {round(total_time, 2)} sec')