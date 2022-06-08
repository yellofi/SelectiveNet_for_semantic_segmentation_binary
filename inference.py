import argparse
import os
import time
import numpy as np
from openslide import OpenSlide
from PIL import Image
import cv2

from patch_gen.slide_utils import *
from tumor_region_seg.Dataset_inference import *
from torch.utils.data import DataLoader
from tumor_region_seg.model import *
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

    parser.add_argument('--model_path', type=str, 
                        default='*/model/model.pth', help='model path (*.pth)')
    parser.add_argument('--model_name', type=str, default='0_baseline')

    parser.add_argument('--save_dir', action="store", type=str,
                        default='/mnt/ssd1/biomarker/c-met/final_output', help='directory where it will save patches')

    parser.add_argument('--tissue_mask_type', action="store", type=str,
                        default='sobel', choices=['otsu', 'sobel'])

    parser.add_argument('--patch_mag', action="store", type=int,
                        default=200, help='target magnifications of generated patches')
    parser.add_argument('--patch_size', action="store", type=int,
                        default=1024, help='a width/height length of squared patches')
    parser.add_argument('--patch_stride', action="store", type=int,
                        default=1024, help='window size from current patch to next patch')

    parser.add_argument('--local_rank', type=int, nargs='+', default=[0], help='local gpu ids')

    parser.add_argument('--input_type', type=str, default='RGB')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=24, help='Dataloader num_workers')

    args = parser.parse_args()
    print('')
    print('args={}\n'.format(args))

    return args

def net_test_load(model_path, net, device=None):
    if device != None:
        ckpt = torch.load(model_path, map_location=device)
    else:
        ckpt = torch.load(model_path)

    try: ckpt['net'] = remove_module(ckpt)
    except: pass
    net.load_state_dict(ckpt['net'])

    # print('     model: ', model_path)
    return net

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
    
    # kernel_size = 61
    kernel_size = 15
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

    """
    Tissue Mask
    """

    print("Mask Tissue Mask...")

    slide = OpenSlide(slide_path)
    slide_mpp = float(slide.properties['openslide.mpp-x'])

    if slide_mpp < 0.2:
        slide_mag = 800
    elif slide_mpp < 0.4:
        slide_mag = 400
    else:
        slide_mag = 200

    # print(slide.dimensions)

    # slide_mag = float(slide.properties['openslide.objective-power']) 
    # slide_mag = int(slide_mag*10)

    ROI_mask_time = 0
    start_time = time.time()    
    ROI_mask = None

    if os.path.isfile(ROI_path):
        RAN = Annotation(slide = slide, level = -1)
        ROI_annotations, _ = RAN.get_coordinates(xml_path = ROI_path, target = 'tissue_region')
        ROI_mask = RAN.make_mask(annotations=ROI_annotations, color = 255)
        end_time = time.time()
        ROI_mask_time += (end_time - start_time)
        total_time += ROI_mask_time

        # print(f'ROI Mask Time: {round(ROI_mask_time, 2)} sec')

    tissue_mask_time = 0
    start_time = time.time() 
    TM = TissueMask(slide = slide, level = -1, ROI_mask = ROI_mask)
    tissue_mask, slide_mask_ratio = TM.get_mask_and_ratio(tissue_mask_type = args.tissue_mask_type)

    # if not os.path.isfile(save_dir + f'/{slide_name}_tissue_mask.jpg'):
    #     cv2.imwrite(save_dir + f'/{slide_name}_tissue_mask.jpg', tissue_mask)

    end_time = time.time()
    tissue_mask_time += (end_time - start_time)
    total_time += tissue_mask_time

    print(f'    slide mag: {slide_mag}')
    print(f'    slide mpp: {slide_mpp}')
    print(f'    slide/tissue_mask ratio: {slide_mask_ratio}')

    print(f'    ROI Mask Time: {round(ROI_mask_time, 2)} sec')
    print(f'    Tissue Mask Time: {round(tissue_mask_time, 2)} sec')

    """
    Dataloader
    """

    print("Define Dataloader...")

    patch_mag = args.patch_mag
    patch_size = args.patch_size
    patch_stride = args.patch_stride
    slide_patch_ratio = slide_mag//patch_mag

    slide_loader_time = 0
    start_time = time.time() 

    size_on_slide = int(patch_size*slide_patch_ratio)
    step_on_slide = int(patch_stride*slide_patch_ratio)

    denomi = step_on_slide // slide_mask_ratio
    tissue_mask = cv2.resize(tissue_mask, None, fx=1/denomi, fy=1/denomi, interpolation=cv2.INTER_AREA)
    
    y_coord = list(np.where(tissue_mask > 0)[0])
    x_coord = list(np.where(tissue_mask > 0)[1])
    xy_coord = [(int(x_coord[i]*step_on_slide), int(y_coord[i]*step_on_slide)) for i in range(len(x_coord))]

    test_set = Dataset(args=args, slide=slide, xy_coord=xy_coord, slide_patch_ratio = slide_patch_ratio)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True, drop_last=False)

    end_time = time.time()
    slide_loader_time += (end_time - start_time)

    total_time += slide_loader_time

    print(f'    patch mag: {patch_mag}')
    print(f'    patch size: {patch_size}')
    print(f'    patch stride: {patch_stride}')
    print(f'    slide/patch ratio: {slide_patch_ratio}') 
    print(f'    # of patch: {len(xy_coord)}')
    print(f'    Slide Loader Time: {round(slide_loader_time, 2)} sec')

    # for i, (input, x, y) in enumerate(test_loader):
    #     print(i, type(input), input.size(), x, y)


    """
    Load Tumor Segmentation Model
    """

    print("Load Tumor Segmentation Model...")

    rank = args.local_rank
    model_path = args.model_path
    # model_path = '/mnt/ssd1/biomarker/c-met/tumor_seg/model/06_baseline_samsung_data/1-fold/checkpoint/model_epoch197.pth'
    # model_path = '/mnt/ssd1/biomarker/c-met/tumor_seg/model/06_baseline_samsung_data/1-fold/checkpoint/model_epoch403.pth'
    input_type = args.input_type

    print(f'    model path: {model_path}')
    print(f'    input type: {input_type}')
    print(f'    local ranks: {rank}')

    load_model_time = 0
    start_time = time.time() 

    if len(rank) != 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = UNet(input_type)
        net = net_test_load(model_path, net)
        net = torch.nn.DataParallel(net, device_ids=rank)
        net = net.to(device)
    else:
        # single gpu -> device map location으로 불러와야 gpu 0을 안 씀
        device = torch.device(f'cuda:{rank[0]}')
        net = UNet(input_type).to(device)
        net = net_test_load(model_path, net, device=device) 
        torch.cuda.set_device(rank[0])

    # cudnn.benchmark=True
    # net.train(False)

    print(f'    device: {device}')
    
    end_time = time.time()

    load_model_time += (end_time - start_time)
    total_time += load_model_time

    print(f'    Load Model Time: {round(load_model_time, 2)} sec')

    """
    Tumor Prediction and IHC Analyzer
    """

    print("Tumor Prediction...")

    model_output = []
    x_coord, y_coord = [], []

    fn_tonumpy = lambda x : x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std : (x*std) + mean
    fn_norm = lambda x : (x-x.min())/(x.max()-x.min())
    fn_classifier = lambda x : 1.0 * (x > 0.5)

    model_inf_time = 0
    start_time = time.time() 

    with torch.no_grad():
        net.eval()

        for i, data in enumerate(test_loader):
            input = data['input'].to(device)
            # print(input.max(), input.min())
            output = net(input)

            x, y = data['x'], data['y']

            # input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = np.squeeze(fn_tonumpy(output), axis=-1)
            pred = fn_classifier(output)

            model_output.extend(np.uint8(pred*255))

            # for (ii, pi, xi, yi) in zip(input, pred, x, y):
            #     img = Image.fromarray(np.uint8(ii*255)).convert('RGB')
            #     img.save(os.path.join(save_dir, f'{slide_name}_{xi}_{yi}_input.png'))

            #     pred_ = Image.fromarray(np.uint8(pi*255)).convert('L')
            #     pred_.save(os.path.join(save_dir, f'{slide_name}_{xi}_{yi}_pred.png'))

            """
            아마 여기서 input과 output(tumor mask)를 가지고 analyzer에 입력해야할 수 있음

            for (i, o) in zip(input, ouput):
                f, v = Analyzer
                features.extend(f)
                visualizations.extend(v)
            """

            x_coord.extend(x)
            y_coord.extend(y)

            del input, output, pred
            torch.cuda.empty_cache()

            print(f'    batch - {i+1} / {len(test_loader)}')
        
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
    slide_mask_ratio = int(slide.level_downsamples[mask_level]) 
    # 위의 tissue mask에 대한 slide_mask_ratio는 32 or 64, 여기선 4.. 

    tumor_mask = make_wsi_mask(slide, mask_level, patch_size, slide_patch_ratio, slide_mask_ratio, pred_list = model_output, x_coord=x_coord, y_coord=y_coord)
    end_time = time.time()
    wsi_mask_time += (end_time - start_time)
    total_time += wsi_mask_time

    print(f'    slide/tumor_mask ratio: {slide_mask_ratio}')
    print(f'    patch-level mask to WSI Mask Time: {round(wsi_mask_time, 2)} sec')

    # cv2.imwrite(os.path.join(save_dir, f'{slide_name}_wsi-level_tumor_mask.png'), tumor_mask)

    mask_xml_time = 0
    start_time = time.time() 
    xml = make_wsi_xml(tumor_mask, slide_mask_ratio, color = tumor_color, ano_class = tumor_class)
    ET.ElementTree(xml).write(os.path.join(save_dir, f'{slide_name}.xml'))

    end_time = time.time()
    mask_xml_time += (end_time - start_time)
    total_time += mask_xml_time
    print(f'    WSI Mask to Xml Time: {round(mask_xml_time, 2)} sec')

    # print('')
    print(f'    Total Time: {round(total_time, 2)} sec')
    print('')

if __name__ == "__main__":


    # signal.signal(signal.SIGINT, exit_gracefully)  # 인터럽트 발생시
    # signal.signal(signal.SIGHUP, exit_gracefully)  # 터미널과의 연결이 끊겼을 시
    # signal.signal(signal.SIGTERM, exit_gracefully)  # Soft Kill
    
    args = parse_arguments()

    slide_dir = args.slide_dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    if args.slide_name != None:

        slide_path = os.path.join(slide_dir, args.slide_name)    
        slide_name = args.slide_name[:-4]
        ROI_file = f'{slide_name}.xml'
        ROI_path = os.path.join(args.ROI_dir, ROI_file)
        main(args, slide_path, ROI_path)

    else:
        issues = [1, 2, 3, 27, 118]
        slide_list = sorted([svs for svs in os.listdir(slide_dir) if 'svs' in svs and int(svs.split('-')[1][2:]) not in issues])
        
        # target_slides = [27, 32, 47, 59, 80, 87, 90, 94, 106, 107]
        # slide_list = sorted([svs for svs in os.listdir(slide_dir) if 'svs' in svs and int(svs.split('-')[1][2:]) in target_slides])

        for i, (slide_file) in enumerate(slide_list):
            print(slide_file[:-4])

            slide_path = os.path.join(slide_dir, slide_file)
            slide_name = slide_file[:-4]
            ROI_file = f'{slide_name}.xml'
            ROI_path = os.path.join(args.ROI_dir, ROI_file)

            main(args, slide_path, ROI_path)


    


    
