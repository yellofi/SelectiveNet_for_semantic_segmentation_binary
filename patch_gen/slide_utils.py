import numpy as np
import cv2
from xml.etree.ElementTree import parse
from PIL import Image

def get_ROI_mask(slide, xml_path):
    """
    inputs
        slide: WSI (Whole Slide Image), slide = openslide.OpenSlide('/~/*.svs')
        xml: ROI (Region of Interest), xml = parse('/~/*.xml').getroot()

    get a ROI mask using a pair of an WSI (slide) and the corresponding ROI annotation (xml) 
    it processes on the lowest resolution (level 3) of the WSI 

    output
        ROI mask (np.array): a gray image (binary mask)
    """
    
    xml = parse(xml_path).getroot()

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

"""
Mask Maker
"""
class SlideMask:
    def __init__(self, slide):
        self.slide = slide

    def get_coordinates(self, xml_path, level, target = 'tumor_region'):
        xml_path = xml_path
        xml = parse(xml_path).getroot()
        
        slide_mask_ratio = round(self.slide.level_downsamples[level])

        annotations = []
        non_target_annotations = []
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
                    # coordinates.append([round(float(coor.get('x'))), round(float(coor.get('y')))])
                    coordinates.append([round(float(coor.get('x'))//slide_mask_ratio), round(float(coor.get('y'))//slide_mask_ratio)])
                annotation.append(coordinates)
            if target == 'tumor_region':
                if pattern == 'Pattern5':
                    annotations.append(annotation)
                if pattern == 'Pattern3':
                    non_target_annotations.append(annotation)
            elif target == 'tissue_region':
                annotations.append(annotation)
                
        return annotations, non_target_annotations
        
    def make_mask(self, annotations, level, color = 255):
        width, height = self.slide.level_dimensions[level]
        mask = np.zeros((height, width)).astype(np.uint8)

        for anno in annotations:
            _anno = []
            for coors in anno:
                _anno.append(np.array(coors))
            cv2.drawContours(mask, _anno, -1, color, -1)

        return mask

    def make_tissue_mask_otsu(self, ROI_mask, NOI_mask, level):
        """
        get a tissue mask on ROI, using Otsu's thresholding in HSV channel

        RGB2HSV -> S, V_inv (1 - V) -> S' = Otsu(S), V_inv' = Otsu(V_inv) 
        -> S" = MorpOp(S'), V_inv" MorpOp(V_inv') ->  tissue mask = OR(S", V_inv")

        Args:
            slide: WSI (Whole Slide Image), slide = openslide.OpenSlide('/~/*.svs')
            ROI mask (np.array): ROI (Region of Interest) mask, a binary mask whose size is the same with the lowest resolution (level 3) of slide

        Return:
            tissue mask (np.array): a gray image (binary mask)
        """

        # if slide.level_count < 4:
        #     level_index = slide.level_count-1
        # else:
        #     level_index = 3

        slide_thumbnail = self.slide.get_thumbnail(self.slide.level_dimensions[level]) 
        slide_mask_ratio = round(self.slide.level_downsamples[level])
        
        slide_rgb = slide_thumbnail.convert('RGB') 
        otsu_image = cv2.cvtColor(np.array(slide_rgb), cv2.COLOR_BGR2HSV)

        otsu_image_1 = otsu_image[:, :, 1]
        otsu_image_2 = 1 - otsu_image[:, :, 2]

        if type(ROI_mask) != type(None):
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

        otsu_image = np.logical_or(otsu_image_1, otsu_image_2)

        if type(NOI_mask) != type(None):
            exclusion = cv2.bitwise_and(otsu_image, NOI_mask)
            otsu_image = otsu_image - exclusion
        
        return otsu_image.astype(float)*255, slide_mask_ratio

    def make_tissue_mask_sobel(self, ROI_mask, NOI_mask, level):
        """
        get a tissue mask on ROI, using Edge Detection
        RGB2GRAY -> Edge Detection (Sobel) -> Morphological Op. (Closing and Opening) 

        Args:
            slide: WSI (Whole Slide Image), slide = openslide.OpenSlide('/~/*.svs')
            ROI mask (np.array): ROI (Region of Interest) mask, a binary mask whose size is the same with the lowest resolution (level 3) of slide

        Return:
            edge mask (np.array): a gray image (binary mask)
        """

        # if slide.level_count < 4:
        #     level_index = slide.level_count-1
        # else:
        #     level_index = 3

        slide_thumbnail = self.slide.get_thumbnail(self.slide.level_dimensions[level]) 
        slide_gray = np.array(slide_thumbnail.convert('L')) 

        slide_mask_ratio = round(self.slide.level_downsamples[level])

        if type(ROI_mask) != type(None):
            slide_gray = cv2.bitwise_and(slide_gray, ROI_mask)

        edge_mask = cv2.Sobel(src=slide_gray, ddepth=cv2.CV_64F, dx=1,dy=1, ksize=5)
        edge_mask = cv2.convertScaleAbs(edge_mask) # to 0 ~ 255, 크게 상관은 없음

        # edge detection이 목적이면 sobel edge detection을 x방향, y방향으로 따로 한 뒤 합쳐주는 게 맞지만,
        # tissue를 background와 구분하기 위함이라면 앞에 sobel edge detection을 xy방향으로 한 번만 해줘도 크게 문제될 게 없음

        kernel = np.ones((11, 11), dtype=np.uint8)

        edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
        edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_OPEN, kernel)

        if type(NOI_mask) != type(None):
            NOI_mask = cv2.resize(NOI_mask, (edge_mask.shape[1], edge_mask.shape[0]))
            exclusion = cv2.bitwise_and(edge_mask, NOI_mask)
            edge_mask = edge_mask - exclusion
            
        return edge_mask, slide_mask_ratio

    def get_tissue_mask(self, ROI_mask, NOI_mask, level, tissue_mask_type = 'sobel'):
        if tissue_mask_type == 'otsu':
            tissue_mask, slide_mask_ratio = self.make_tissue_mask_otsu(ROI_mask, NOI_mask, level)
        elif tissue_mask_type == 'sobel':
            tissue_mask, slide_mask_ratio = self.make_tissue_mask_sobel(ROI_mask, NOI_mask, level)
        return tissue_mask, slide_mask_ratio

    def estimate_blankfield_white(self, ratio=0.01):
        """
        estimate blankfield of slide  

        Args
            ratio: a ratio to determine a threshold for blankfield in accumulated histogram 
                        N = maximum value of accumulated histogram
                        k = (N - ratio / 2)

                        If, at an index, the value of a accumulated histogram is not higher than N - k, 
                        the index becomes a threshold to determine blankfield
        Return
            white: tuple, (R_white_mean, G_white_mean, B_white_mean)
        """

        slide_thumbnail = self.slide.get_thumbnail(self.slide.level_dimensions[-1]) 
        slide_rgb= np.array(slide_thumbnail.convert('RGB'))

        rgb = slide_rgb.copy()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        for i in range(1, 256):
            hist[i][0] += hist[i - 1][0]
        N = hist[-1][0]
        k = round(N * ratio / 2)
        threshold = 255
        while hist[threshold][0] > N - k:
            threshold -= 1
        mask = (gray >= threshold).astype(np.uint8)
        white = np.array(cv2.mean(rgb, mask=mask)[:3], dtype=np.uint8)
        return white

"""
Blankfield Correction
"""
# 영상내 밝기가 높은 값을 찾고, 그 영역의 밝기 평균을 구함
def estimate_blankfield_white(image, ratio=0.01):
    """
    Args
        image: a RGB image, (H, W, 3)
        ratio: a ratio to determine a threshold for blankfield in accumulated histogram 
                    N = maximum value of accumulated histogram
                    k = (N - ratio / 2)

                    If, at an index, the value of a accumulated histogram is not higher than N - k, 
                    the index becomes a threshold to determine blankfield
    Return
        white: tuple, (R_white_mean, G_white_mean, B_white_mean)
    """
    rgb = image.copy()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    for i in range(1, 256):
        hist[i][0] += hist[i - 1][0]
    N = hist[-1][0]
    k = round(N * ratio / 2)
    threshold = 255
    while hist[threshold][0] > N - k:
        threshold -= 1
    mask = (gray >= threshold).astype(np.uint8)
    white = np.array(cv2.mean(rgb, mask=mask)[:3], dtype=np.uint8)
    return white

# 영상내 밝기 높은 영역의 평균이 255가 되도록, 0~255로 변경
def correct_background(image, white=None, ratio=0.01, target=255):
    """
    Args
        image: a RGB image, numpy.ndarray, (H, W, 3)
    
    """
    if white is None:
        white = estimate_blankfield_white(image, ratio=ratio)
    rgb = image.copy()
    divider = np.zeros_like(rgb)
    for ch in range(0, 3):
        divider[:, :, ch] = white[ch]  
    cv2.divide(rgb, divider, rgb, scale=target)
    return rgb