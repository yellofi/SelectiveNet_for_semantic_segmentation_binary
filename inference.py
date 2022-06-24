import argparse
import os
import time
from tqdm import tqdm
import numpy as np
from openslide import OpenSlide
from PIL import Image
import cv2

from patch_gen.slide_utils import *
from tumor_seg.Dataset_inference import *
from tumor_seg.net_utils import *
from tumor_seg.model import *

from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement

import torch.distributed as dist
import signal

def exit_gracefully(self, *args):
    dist.destroy_process_group()

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--slide_dir', action="store", type=str,
                        default='/mnt/nfs0/jycho/SLIDE_DATA/록원재단/AT2/C-MET_slide', help='WSI data (.svs) directory')
    parser.add_argument('--slide_name', action="store", type=str,
                        default=None, help='WSI data (.svs)')
    parser.add_argument('--ROI_dir', action="store", type=str,
                        default='*/ROI_annotation/annotation', help='rough tissue region annotation (.xml) directory')

    parser.add_argument('--tissue_mask_type', action="store", type=str,
                        default='sobel', choices=['otsu', 'sobel'])
    parser.add_argument('--patch_mag', action="store", type=int,
                        default=200, help='target magnifications of generated patches')
    parser.add_argument('--patch_size', action="store", type=int,
                        default=1024, help='a width/height length of squared patches')
    parser.add_argument('--patch_stride', action="store", type=int,
                        default=1024, help='window size from current patch to next patch')
    parser.add_argument('--patch_grid_save', action="store", type=bool,
                        default=False, help='Whether grid visualization of patches will be saved')

    parser.add_argument('--input_type', type=str, default='RGB')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=16, help='Dataloader num_workers')

    parser.add_argument('--model_dir', type=str, 
                        default='*/model', help='network ckpt (.pth) directory')
    parser.add_argument('--model_arch', type=str, nargs = '+',
                        default = ['UNet'], choices=['UNet'])
    parser.add_argument('--ens_scale', type=str,
                        default = 'None', choices=['None', 'clip', 'sigmoid', 'minmax'])
                
    parser.add_argument('--local_rank', type=int, nargs='+', default=[0], help='local gpu ids')

    parser.add_argument('--save_dir', action="store", type=str,
                        default='/mnt/ssd1/biomarker/c-met/final_output')


    args = parser.parse_args()
    print('')
    print('args={}\n'.format(args))

    return args

def make_wsi_mask_(slide, level, slide_patch_ratio, slide_mask_ratio, pred_dir):
    """
    make a wsi-level mask with already saved outputs 
    """

    width, height = slide.level_dimensions[level]

    # pred_list = [p for p in sorted(os.listdir(pred_dir)) if 'sample_NT_add_ens_pred' in p]
    pred_list = [p for p in sorted(os.listdir(pred_dir)) if 'baseline_403_pred' in p]

    mask = np.zeros((height, width))

    patch_size = 1024
    mask_step = patch_size*slide_patch_ratio//slide_mask_ratio

    for i in range(len(pred_list)):

        raw_x = int(pred_list[i].split('_')[1])
        raw_y = int(pred_list[i].split('_')[2])

        x = raw_x//slide_mask_ratio
        y = raw_y//slide_mask_ratio

        img = Image.open(os.path.join(pred_dir, pred_list[i])).convert('L')
        img = np.array(img)
        img = cv2.resize(img, (mask_step, mask_step), cv2.INTER_AREA)

        mask[y:y+mask_step, x:x+mask_step] += img

    mask = mask.astype('uint8')
    
    kernel_size = 61
    # kernel_size = 15
    mask = cv2.medianBlur(mask, kernel_size)

    return mask

def make_wsi_mask(slide, level, patch_size, slide_patch_ratio, slide_mask_ratio, pred_list, x_coord, y_coord):
    """
    make a wsi-level mask with a list of binary prediction 
    """
    width, height = slide.level_dimensions[level]

    mask = np.zeros((height, width))
    mask_step = patch_size*slide_patch_ratio//slide_mask_ratio

    # print(mask.shape)
    # print(mask_step)
    for i in range(len(pred_list)):
        
        raw_x = x_coord[i]
        raw_y = y_coord[i]

        x = raw_x//slide_mask_ratio
        y = raw_y//slide_mask_ratio

        img = pred_list[i]
        img = cv2.resize(img, (mask_step, mask_step), cv2.INTER_AREA)

        # print(i, raw_x, raw_y, x, y, x+mask_step, y+mask_step)
        try: mask[y:y+mask_step, x:x+mask_step] += img
        except: pass

    mask = mask.astype('uint8')
    # return mask, cv2.medianBlur(mask, 15), cv2.medianBlur(mask, 61)

    kernel_size = 61
    # kernel_size = 15
    mask = cv2.medianBlur(mask, kernel_size)

    return mask
    
def make_wsi_xml(mask, slide_mask_ratio, color = "-65536", ano_class = "Pattern5"):

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    total = Element("object-stream")
    Annotations = Element("Annotations")

    """
    <Annotations image="/Users/hyc-deepbio/Desktop/untitled folder/S-LC0027-MET.svs" score="None" primary="None" secondary="None" pni="false" quality="false" inflammation="false" benign="false" type="BRIGHTFIELD_H_DAB">
    """

    Annotations.attrib["image"] = slide_path
    Annotations.attrib["score"] = ""
    Annotations.attrib["primary"] = ""
    Annotations.attrib["secondary"] = ""
    Annotations.attrib["pni"] = "false"
    Annotations.attrib["quality"] = "false"
    Annotations.attrib["inflammation"] = "false"
    Annotations.attrib["benign"] = "false"
    Annotations.attrib["type"] = "BRIGHTFIELD_H_DAB"

    Comment  = Element("Comment")
    Comment.text = ""
    Annotations.append(Comment)

    Annotation_list = []

    """
    <Annotation class="Pattern5" type="Area" color="-65536">
    """

    for j, contour in enumerate(contours):
        # [이전 윤곽선, 다음 윤곽선, 내곽 윤곽선, 외곽 윤곽선]
        if hierarchy[0][j][3] == -1: #외곽선일 경우
            Annotation = Element("Annotation")
            Annotation.attrib["class"] = ano_class
            Annotation.attrib["type"] = "Area"
            Annotation.attrib["color"] = color

            Memo = Element("Memo")
            Memo.text = ""
            Annotation.append(Memo)
        else:
            Annotation_list.append(0)
            Annotation = Annotation_list[hierarchy[0][j][3]]

        Coordinates = Element("Coordinates")

        for points in range(contour.shape[0]):
            point_x, point_y = contour[points][0]
            SubElement(Coordinates, "Coordinate", x=str(point_x*(slide_mask_ratio)), y=str(point_y*(slide_mask_ratio)))

        try:
            Annotation.append(Coordinates)
        except:
            pass

        if hierarchy[0][j][3] == -1:
            Annotation_list.append(Annotation)
        else:
            Annotation_list[hierarchy[0][j][3]] = Annotation

    for Anno_candidate in Annotation_list:
        if Anno_candidate:
            Annotations.append(Anno_candidate)

    total.append(Annotations)
    return total
   
def main(args, slide_path, ROI_path):

    total_time = 0

    print("Patch Loader...")

    """
    Tissue Mask
    """

    patch_loader_time = 0
    start_time = time.time() 

    slide = OpenSlide(slide_path)
    slide_mpp = float(slide.properties['openslide.mpp-x'])

    if slide_mpp < 0.2:
        slide_mag = 800
    elif slide_mpp < 0.4:
        slide_mag = 400
    else:
        slide_mag = 200

    # slide_mag = float(slide.properties['openslide.objective-power']) 
    # slide_mag = int(slide_mag*10)
 
    ROI_mask = None
    SM = SlideMask(slide=slide)
    if os.path.isfile(ROI_path):
        ROI_annotation, _ = SM.get_coordinates(ROI_path, level = -1, target = 'tissue_region')
        ROI_mask = SM.make_mask(ROI_annotation, level = -1, color = 255)
    else:
        print(f'    Region of Interest file {ROI_path} does not exist')
        # print(f'ROI Mask Time: {round(ROI_mask_time, 2)} sec')

    tissue_mask, slide_mask_ratio = SM.get_tissue_mask(ROI_mask, NOI_mask = None, level = -1, tissue_mask_type=args.tissue_mask_type)
    
    white = SM.estimate_blankfield_white(ratio=0.01)

    # if not os.path.isfile(save_dir + f'/{slide_name}_tissue_mask.jpg'):
    #     cv2.imwrite(save_dir + f'/{slide_name}_tissue_mask.jpg', tissue_mask)

    """
    Coordinates, Dataset and Dataloader
    """

    patch_mag = args.patch_mag
    patch_size = args.patch_size
    patch_stride = args.patch_stride
    slide_patch_ratio = slide_mag//patch_mag

    size_on_slide = int(patch_size*slide_patch_ratio)
    step_on_slide = int(patch_stride*slide_patch_ratio)

    denomi = step_on_slide // slide_mask_ratio
    tissue_mask = cv2.resize(tissue_mask, None, fx=1/denomi, fy=1/denomi, interpolation=cv2.INTER_AREA)
    
    y_coord = list(np.where(tissue_mask > 0)[0])
    x_coord = list(np.where(tissue_mask > 0)[1])
    xy_coord = [(int(x_coord[i]*step_on_slide), int(y_coord[i]*step_on_slide)) for i in range(len(x_coord))]
    
    if args.patch_grid_save:
        slide_ = slide.get_thumbnail(slide.level_dimensions[-1]).convert('RGB')
        slide_ = cv2.cvtColor(np.array(slide_, dtype = np.uint8), cv2.COLOR_RGB2BGR)

        for (x, y) in xy_coord:
            mask_x, mask_y = x//slide_mask_ratio, y//slide_mask_ratio
            mask_step = step_on_slide//slide_mask_ratio
            slide_ = cv2.rectangle(slide_, 
            (mask_x, mask_y), (mask_x+mask_step, mask_y+mask_step), 
            color=(0, 255, 0), thickness=2)
        
        cv2.imwrite(os.path.join(save_dir, f'{slide_name}_x{patch_mag}_{patch_size}_num-{len(xy_coord)}.jpg'), slide_)

    test_set = Dataset(slide=slide, white=white, xy_coord=xy_coord, size_on_slide = size_on_slide, patch_size = patch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True, drop_last=False)

    end_time = time.time()
    patch_loader_time += (end_time - start_time)

    total_time += patch_loader_time

    print(f'    slide mag: {slide_mag}')
    print(f'    slide mpp: {slide_mpp}')
    print(f'    slide/tissue_mask ratio: {slide_mask_ratio}')
    print(f'    patch mag: {patch_mag}')
    print(f'    patch size: {patch_size}')
    print(f'    patch stride: {patch_stride}')
    print(f'    slide/patch ratio: {slide_patch_ratio}') 
    print(f'    # of patch: {len(xy_coord)}')
    print(f'    batch size: {args.batch_size}')
    print(f'    num workers: {args.num_workers}')
    print(f'    Patch Loader Time: {round(patch_loader_time, 2)} sec')

    """
    Load Tumor Segmentation Model
    """

    print("Load Tumor Segmentation Model...")

    rank = args.local_rank
    model_dir = args.model_dir
    input_type = args.input_type
    model_arch = args.model_arch # list

    load_model_time = 0
    start_time = time.time() 

    model_list = sorted([ckpt for ckpt in os.listdir(model_dir) if 'pth' in ckpt])
    
    if len(model_list) != 1 and len(model_arch) == 1:
        model_arch_ = model_arch[0]
        model_arch = [model_arch_ for _ in range(len(model_list))]

    model_dict = {'UNet': UNet}
    nets = []

    for i in range(len(model_list)):
 
        model_path = os.path.join(model_dir, model_list[i])
        print(f'    {model_path} - {model_arch[i]}')

        if len(rank) != 1:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            net = model_dict[model_arch[i]](input_type)
            net = net_test_load(model_path, net)
            net = torch.nn.DataParallel(net, device_ids=rank)
            net = net.to(device) 
        else:
            # single gpu -> device map location으로 불러와야 gpu 0을 안 씀
            device = torch.device(f'cuda:{rank[0]}')
            net = model_dict[model_arch[i]](input_type).to(device)
            net = net_test_load(model_path, net, device=device)
            
        net.train(False)
        nets.append(net)

    if len(rank) == 1:
        torch.cuda.set_device(rank[0])

    # cudnn.benchmark=True
    
    end_time = time.time()

    load_model_time += (end_time - start_time)
    total_time += load_model_time

    print(f'    input type: {input_type}')
    print(f'    local ranks: {rank}')
    print(f'    device: {device}')
    print(f'    Model Loading Time: {round(load_model_time, 2)} sec')

    """
    Tumor Prediction and IHC Analyzer
    """

    print("Tumor Prediction...")

    model_output = []
    x_coord, y_coord = [], []

    fn_tonumpy = lambda x : x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std : (x*std) + mean

    # ensemble
    ensemble_scale = args.ens_scale
    fn_scale_minmax = lambda x : (x-x.min())/(x.max()-x.min())
    fn_sigmoid = lambda x : 1/(1+ np.exp(-(x.astype('float64')-0.5)))
    fn_clip = lambda x: np.clip(x, 0, 1)

    fn_classifier = lambda x : 1.0 * (x > 0.5)

    model_inf_time = 0
    start_time = time.time() 

    with torch.no_grad():
        # net.eval()

        for data in tqdm(test_loader, total = len(test_loader), desc = '   model inference'):
            input = data['input'].to(device)
            x, y = data['x'], data['y']
            
            if len(nets) == 1:
                output = nets[0](input)
                output = np.squeeze(fn_tonumpy(output), axis = -1)
            else:
                outputs = []
                for net in nets:
                    output = net(input)
                    if ensemble_scale == 'None':
                        outputs.append(np.squeeze(fn_tonumpy(output), axis = -1))
                    elif ensemble_scale == 'clip':
                        outputs.append(np.squeeze(fn_tonumpy(fn_clip(output)), axis = -1))
                    elif ensemble_scale == 'minmax':
                        outputs.append(np.squeeze(fn_tonumpy(fn_scale_minmax(output)), axis = -1))
                    elif ensemble_scale == 'sigmoid':
                        outputs.append(np.squeeze(fn_tonumpy(fn_sigmoid(output)), axis = -1))
                output = np.mean(np.asarray(outputs), axis = 0)
                del outputs

            # input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            
            pred = fn_classifier(output)
            model_output.extend(np.uint8(pred*255))

            x_coord.extend(x)
            y_coord.extend(y)

            """
            아마 여기서 input과 output(tumor mask)를 가지고 analyzer에 입력해야할 수 있음

            for (i, o) in zip(input, ouput):
                f, v = Analyzer
                features.extend(f)
                visualizations.extend(v)
            """

            del input, output, pred
            torch.cuda.empty_cache()
        
    end_time = time.time()

    model_inf_time += (end_time - start_time)
    total_time += model_inf_time

    print(f'    Model Inference Time: {round(model_inf_time, 2)} sec')

    """
    Mask wsi-level mask and Save mask as .xml
    """
    print("Predictied Tumor Region as .xml...")

    tumor_color = "-65536"
    tumor_class = "Pattern5"

    wsi_mask_time = 0
    start_time = time.time() 

    mask_level = 1
    slide_mask_ratio = round(slide.level_downsamples[mask_level]) 
    # 위의 tissue mask에 대한 slide_mask_ratio는 32 or 64, 여기선 4.. 

    tumor_mask = make_wsi_mask(slide, mask_level, patch_size, slide_patch_ratio, slide_mask_ratio, pred_list = model_output, x_coord=x_coord, y_coord=y_coord)
    # mask, mask_mb_15, mask_mb_61 = make_wsi_mask(slide, mask_level, patch_size, slide_patch_ratio, slide_mask_ratio, pred_list = model_output, x_coord=x_coord, y_coord=y_coord)
    end_time = time.time()
    wsi_mask_time += (end_time - start_time)
    total_time += wsi_mask_time

    tumor_mask_ = cv2.resize(tumor_mask, slide.level_dimensions[-1], interpolation=cv2.INTER_AREA)

    print(f'    slide/tumor_mask ratio: {slide_mask_ratio}')
    print(f'    Making WSI-level Mask Time: {round(wsi_mask_time, 2)} sec')

    cv2.imwrite(os.path.join(save_dir, f'{slide_name}_wsi-level_tumor_mask.png'), tumor_mask_)
    # cv2.imwrite(os.path.join(save_dir, f'{slide_name}_wsi-level_tumor_mask_median_blur_15.png'), mask_mb_15)
    # cv2.imwrite(os.path.join(save_dir, f'{slide_name}_wsi-level_tumor_mask_median_blur_61.png'), mask_mb_61)
    

    mask_xml_time = 0
    start_time = time.time() 

    xml_save_dir = os.path.join(save_dir, 'xml')
    os.makedirs(xml_save_dir, exist_ok = True)

    xml = make_wsi_xml(tumor_mask, slide_mask_ratio, color = tumor_color, ano_class = tumor_class)
    ET.ElementTree(xml).write(os.path.join(xml_save_dir, f'{slide_name}.xml'))

    end_time = time.time()
    mask_xml_time += (end_time - start_time)
    total_time += mask_xml_time
    print(f'    WSI Mask to .xml Time: {round(mask_xml_time, 2)} sec')
    print(f'    Total Elapsed Time: {round(total_time, 0)}s')
    m, s = divmod(total_time, 60)
    print(f'    Total Elapsed Time: {m}m {round(s, 0)}s')
    print('')
    
    return total_time

if __name__ == "__main__":

    args = parse_arguments()

    slide_dir = args.slide_dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    if args.slide_name != None:
        slide_path = os.path.join(slide_dir, args.slide_name)    
        slide_name = args.slide_name[:-4]
        print(slide_name)
        ROI_file = f'{slide_name}.xml'
        ROI_path = os.path.join(args.ROI_dir, ROI_file)
        main(args, slide_path, ROI_path)
    else:
        issues = [118]
        slide_list = sorted([svs for svs in os.listdir(slide_dir) if 'svs' in svs and int(svs.split('-')[1][2:]) not in issues])
        
        # target_slides = [27, 32, 47, 59, 80, 87, 90, 94, 106, 107]
        # slide_list = sorted([svs for svs in os.listdir(slide_dir) if 'svs' in svs and int(svs.split('-')[1][2:]) in target_slides])

        total_time = 0
        for i, (slide_file) in enumerate(slide_list):
            print(slide_file[:-4])
            slide_path = os.path.join(slide_dir, slide_file)
            slide_name = slide_file[:-4]
            ROI_file = f'{slide_name}.xml'
            ROI_path = os.path.join(args.ROI_dir, ROI_file)
            elsapsed_time = main(args, slide_path, ROI_path)
            total_time += elsapsed_time
        
        print(slide_list)
        print(f'    Total Elapsed Time: {round(total_time, 2)}s')
        m, s = divmod(total_time, 60)
        h, m = divmod(m, 60)
        print(f'    Total Elapsed Time: {h}h {m}m {round(s, 2)}s')
    

    signal.signal(signal.SIGINT, exit_gracefully)  # 인터럽트 발생시
    signal.signal(signal.SIGHUP, exit_gracefully)  # 터미널과의 연결이 끊겼을 시
    signal.signal(signal.SIGTERM, exit_gracefully)  # Soft Kill
    
