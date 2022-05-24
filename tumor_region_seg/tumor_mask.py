import os
from xml.etree.ElementTree import parse
import cv2
import numpy as np
import argparse

def read_xml(xml_path):
    """
    read .xml file and extract coordinates of annotation for each pattern 
    annotation done by Deep Bio pathologist HyeYoon Chang

    Args:
        xml_path (str)

    Return:
        annotations (list): coordinates of annotation for each pattern
    """

    xml = parse(xml_path).getroot()

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
                coordinates.append([round(float(coor.get('x'))), round(float(coor.get('y')))])

            annotation.append(coordinates)

        annotations.append(annotation)
    return annotations

def make_tumor_mask(img_rgb, annotations):
    """
    make a binary tumor mask by using annotation extracted from .xml file, 
    whose size is the same with a patch image
    
    Args:
        annoataions (list): coordinates of annotation for each pattern
    
    Return:
        tumor_mask (np.array): a binary mask, 0 (Non-tumor area) or 255 (Tumor area)
    """
    
    tumor_mask = np.zeros(img_rgb.shape[:-1]).astype(np.uint8)

    for anno in annotations:
        _anno = []
        for coors in anno:
            _anno.append(np.array(coors))

        cv2.drawContours(tumor_mask, _anno, -1, 255, -1)

    return tumor_mask

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_dir', action="store", type=str,
                        default='/mnt/hdd1/c-MET_datasets/Lung_c-MET IHC_scored/sample', help='sample image data directory')

    args = parser.parse_args()
    print('')
    print('args={}\n'.format(args))

    return args


if __name__ == '__main__':
    args = parse_arguments()
    img_dir = args.img_dir
    xml_dir = img_dir + '/annotation'
    output_dir = xml_dir

    img_list = [img for img in sorted(os.listdir(img_dir)) if '.png' in img]
    xml_list = [xml for xml in sorted(os.listdir(xml_dir)) if '.xml' in xml]

    for i, (img, xml) in enumerate(zip(img_list, xml_list)):
        print(i, img, xml)
        annotations = read_xml(os.path.join(xml_dir, xml))
        img_rgb = cv2.cvtColor(cv2.imread(os.path.join(img_dir, img)), cv2.COLOR_BGR2RGB)
        tumor_mask = make_tumor_mask(img_rgb, annotations)
        cv2.imwrite(os.path.join(output_dir, img), tumor_mask)
