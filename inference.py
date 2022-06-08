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

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--slide_dir', action="store", type=str,
                        default='/mnt/nfs0/jycho/SLIDE_DATA/록원재단/AT2/C-MET_slide', help='WSI data (.svs) directory')
    parser.add_argument('--slide_name', action="store", type=str,
                        default='*.svs', help='WSI data (.svs)')
    parser.add_argument('--ROI_dir', action="store", type=str,
                        default='/mnt/hdd1/c-MET_datasets/SLIDE_DATA/ROI_annotation', help='rough tissue region annotation (.xml) directory')

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
        dict_model = torch.load(model_path, map_location=device)
    else:
        dict_model = torch.load(model_path, map_location='cpu')

    k = list(dict_model['net'].keys())[0]
    if "module" in k:
        dict_model['net'] = remove_module(dict_model)
    net.load_state_dict(dict_model['net'])

    print('     model: ', model_path)
    return net

def slide_loader(args, tissue_mask, patch_stride, slide_patch_ratio, slide_mask_ratio):

    denomi = (patch_stride*slide_patch_ratio) // slide_mask_ratio
    tissue_mask = cv2.resize(tissue_mask, None, fx=1/denomi, fy=1/denomi, interpolation=cv2.INTER_AREA)

    # if not os.path.isfile(save_dir + f'/{slide_name}_tissue_mask.jpg'):
    #     cv2.imwrite(save_dir + f'/{slide_name}_tissue_mask.jpg', tissue_mask)
    
    y_coord = list(np.where(tissue_mask > 0)[0])
    x_coord = list(np.where(tissue_mask > 0)[1])
    xy_coord = [(x_coord[i]*patch_stride*slide_patch_ratio, y_coord[i]*patch_stride*slide_patch_ratio) for i in range(len(x_coord))]

    test_set = Dataset(args=args, slide=slide, xy_coord=xy_coord, slide_patch_ratio = slide_patch_ratio)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True, drop_last=False)

    return test_loader

def make_wsi_mask_(slide, level, pred_dir):
    """
    make wsi-level mask with already saved outputs 
    """

    width, height = slide.level_dimensions[level]

    # pred_list = [p for p in sorted(os.listdir(pred_dir)) if 'sample_NT_add_ens_pred' in p]
    pred_list = [p for p in sorted(os.listdir(pred_dir)) if 'baseline_403_pred' in p]

    mask = np.zeros((height, width))

    slide_patch_ratio = slide_mag//patch_mag
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

def make_wsi_mask(slide, level, pred_list, x_coord, y_coord):
    """
    make wsi-level mask with already saved outputs 
    """
    width, height = slide.level_dimensions[level]
    slide_mask_ratio = int(slide.level_downsamples[level])

    mask = np.zeros((height, width))
    mask_step = patch_size*slide_patch_ratio//slide_mask_ratio

    for i in range(len(pred_list)):

        raw_x = x_coord[i]
        raw_y = y_coord[i]

        x = raw_x//slide_mask_ratio
        y = raw_y//slide_mask_ratio

        img = pred_list[i]
        img = cv2.resize(img, (mask_step, mask_step), cv2.INTER_AREA)

        mask[y:y+mask_step, x:x+mask_step] += img

    mask = mask.astype('uint8')
    
    kernel_size = 61
    # kernel_size = 15
    mask = cv2.medianBlur(mask, kernel_size)

    return mask


def make_wsi_xml(mask, color = "-65536", ano_class = "Pattern5"):

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
    ET.ElementTree(total).write(os.path.join(save_dir, f'{slide_name}.xml'))


if __name__ == "__main__":
    
    args = parse_arguments()

    slide_path = os.path.join(args.slide_dir, args.slide_name)    
    slide_name = args.slide_name[:-4]
    ROI_file = f'{slide_name}.xml'
    ROI_path = os.path.join(args.ROI_dir, ROI_file)

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    total_time = 0

    """
    Tissue Mask
    """

    

    slide = OpenSlide(slide_path)
    slide_mpp = float(slide.properties['openslide.mpp-x'])

    if slide_mpp < 0.2:
        slide_mag = 800
    elif slide_mpp < 0.4:
        slide_mag = 400
    else:
        slide_mag = 200

    slide.dimensions

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

        print(f'ROI Mask Time: {round(tissue_mask_time, 2)} sec')

    tissue_mask_time = 0
    start_time = time.time() 
    TM = TissueMask(slide = slide, level = -1, ROI_mask = ROI_mask)
    tissue_mask, slide_mask_ratio = TM.get_mask_and_ratio(tissue_mask_type = args.tissue_mask_type)

    end_time = time.time()
    tissue_mask_time += (end_time - start_time)
    total_time += tissue_mask_time

    print(f'Tissue Mask Time: {round(tissue_mask_time, 2)} sec')

    """
    Dataloader
    """

    

    patch_mag = 200 # args.patch_mag
    patch_size = 1024 # args.patch_size
    patch_stride = 1024 # args.patch_stride
    slide_patch_ratio = slide_mag/patch_mag

    slide_loader_time = 0
    start_time = time.time() 
    test_loader = slide_loader(args, tissue_mask, patch_stride, slide_patch_ratio, slide_mask_ratio)
    end_time = time.time()
    slide_loader_time += (end_time - start_time)

    total_time += slide_loader_time

    print(f'Slide Loader Time: {round(slide_loader_time, 2)} sec')

    """
    Load Tumor Segmentation Model
    """

    

    rank = args.local_rank
    model_path = args.model_path
    # model_path = '/mnt/ssd1/biomarker/c-met/tumor_seg/model/06_baseline_samsung_data/1-fold/checkpoint/model_epoch197.pth'
    # model_path = '/mnt/ssd1/biomarker/c-met/tumor_seg/model/06_baseline_samsung_data/1-fold/checkpoint/model_epoch403.pth'
    input_type = args.input_type

    load_model_time = 0
    start_time = time.time() 

    if len(rank) != 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = UNet(input_type, DataParallel=True)
        net = net_test_load(model_path, net)
        net = torch.nn.DataParallel(net, device_ids=rank)
        net = net.to(device)
    else:
        # single gpu -> device map location으로 불러와야 gpu 0을 안 씀
        device = torch.device(f'cuda:{rank[0]}')
        net = UNet(input_type).to(device)
        net = net_test_load(model_path, net, device=device) 
        torch.cuda.set_device(rank[0])

    cudnn.benchmark=True
    end_time = time.time()

    load_model_time += (end_time - start_time)
    total_time += load_model_time

    print(f'Load Model Time: {round(load_model_time, 2)} sec')

    """
    Tumor Prediction and IHC Analyzer
    """

    model_output = []
    model_selection = []
    x_coord, y_coord = [], []

    fn_tonumpy = lambda x : x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std : (x*std) + mean
    fn_norm = lambda x : (x-x.min())/(x.max()-x.min())
    fn_classifier = lambda x : 1.0 * (x > 0.5)

    print("test start!")

    model_inf_time = 0
    start_time = time.time() 

    with torch.no_grad():
        for i, (input, x, y) in enumerate(test_loader):
            input = input.to(device)
            output = net(input)

            # input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = np.squeeze(fn_tonumpy(output), axis=-1)
            pred = fn_classifier(output)

            model_output.extend(pred)

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
        
    end_time = time.time()

    model_inf_time += (end_time - start_time)
    total_time += model_inf_time

    print(f'Model Inference Time: {round(model_inf_time, 2)} sec')

    """
    Mask wsi-level mask and Save mask as .xml
    """

    tumor_color = "-65536"
    tumor_class = "Pattern5"

    wsi_mask_time = 0
    start_time = time.time() 
    tumor_mask = make_wsi_mask(slide, level = 1, pred_list = model_output, x_coord=x_coord, y_coord=y_coord)
    end_time = time.time()
    wsi_mask_time += (end_time - start_time)
    total_time += wsi_mask_time
    print(f'patch-level mask to WSI Mask Time: {round(wsi_mask_time, 2)} sec')

    mask_xml_time = 0
    start_time = time.time() 
    make_wsi_xml(tumor_mask, color = tumor_color, ano_class = tumor_class)
    end_time = time.time()
    mask_xml_time += (end_time - start_time)
    total_time += mask_xml_time
    print(f'WSI Mask to Xml Time: {round(mask_xml_time, 2)} sec')


    print(f'Total Time: {round(total_time, 2)} sec')


    
