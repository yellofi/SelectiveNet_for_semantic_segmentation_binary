import numpy as np
import cv2
from xml.etree.ElementTree import parse

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
Label maker
"""
import multiprocessing as mp
# pool = mp.Pool(processes=12)
# pool.map(count, num_list)
# pool.close()
# pool.join()
class Annotation:
    def __init__(self, slide, level = -1):

        # self.slide_path = slide_path
        # self.slide = openslide.OpenSlide(slide_path)

        self.slide = slide
        self.level = level

        # slide_thumbnail = self.slide.get_thumbnail(self.slide.level_dimensions[self.level])
        # slide_thumbnail = slide_thumbnail.convert('RGB')
        # self.slide_thumbnail = np.array(slide_thumbnail)

    def get_coordinates(self, xml_path, target = 'tumor_region'):

        xml_path = xml_path
        xml = parse(xml_path).getroot()
        
        slide_mask_ratio = round(self.slide.level_downsamples[self.level])

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

    # def draw_contours(self, mask, anno, color):
    #     _anno = []
    #     for coors in anno:
    #         _anno.append(np.array(coors))
    #     cv2.drawContours(mask, _anno, -1, color, -1)
    
    def make_mask(self, annotations, color = 255):

        width, height = self.slide.level_dimensions[self.level]
        mask = np.zeros((height, width)).astype(np.uint8)
        # mask = np.zeros((height, width, 3)).astype(np.uint8)
       
        # pool = mp.Pool(processes=12)
        # pool.map(self.draw_contours, [(mask, anno, color) for anno in annotations])
        # # result = pool.map_async(self.draw_contours, [(mask, anno, color) for anno in annotations])
        # pool.close()
        # pool.join()

        for anno in annotations:
            _anno = []
            for coors in anno:
                _anno.append(np.array(coors))
            cv2.drawContours(mask, _anno, -1, color, -1)

        return mask



"""
Tissue Masking
"""

class TissueMask:
    def __init__(self, slide, level = -1, ROI_mask = None, NOI_mask = None):
        self.slide = slide
        self.level = level
        self.ROI_mask = ROI_mask
        self.NOI_mask = NOI_mask

    def make_tissue_mask_otsu(self, slide, ROI_mask, NOI_mask, level):
        """
        Args:
            slide: WSI (Whole Slide Image), slide = openslide.OpenSlide('/~/*.svs')
            ROI mask (np.array): ROI (Region of Interest) mask, a binary mask whose size is the same with the lowest resolution (level 3) of slide

        get a tissue mask on ROI, using Otsu's thresholding in HSV channel

        RGB2HSV -> S, V_inv (1 - V) -> S' = Otsu(S), V_inv' = Otsu(V_inv) 
        -> S" = MorpOp(S'), V_inv" MorpOp(V_inv') ->  tissue mask = OR(S", V_inv")

        Return:
            tissue mask (np.array): a gray image (binary mask)
        """

        # if slide.level_count < 4:
        #     level_index = slide.level_count-1
        # else:
        #     level_index = 3

        slide_thumbnail = slide.get_thumbnail(slide.level_dimensions[level]) 
        slide_mask_ratio = round(slide.level_downsamples[level])
        
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

    def make_tissue_mask_sobel(self, slide, ROI_mask, NOI_mask, level):

        """
        Args:
            slide: WSI (Whole Slide Image), slide = openslide.OpenSlide('/~/*.svs')
            ROI mask (np.array): ROI (Region of Interest) mask, a binary mask whose size is the same with the lowest resolution (level 3) of slide

        get a tissue mask on ROI, using Edge Detection

        RGB2GRAY -> Edge Detection (Sobel) -> Morphological Op. (Closing and Opening) 

        Return:
            edge mask (np.array): a gray image (binary mask)
        """

        # if slide.level_count < 4:
        #     level_index = slide.level_count-1
        # else:
        #     level_index = 3

        slide_thumbnail = slide.get_thumbnail(slide.level_dimensions[level]) 
        slide_gray = np.array(slide_thumbnail.convert('L')) 

        slide_mask_ratio = round(slide.level_downsamples[level])

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
            # print(type(NOI_mask), type(edge_mask))
            # print(NOI_mask)
            # print(NOI_mask.shape, edge_mask.shape)
            NOI_mask = cv2.resize(NOI_mask, (edge_mask.shape[1], edge_mask.shape[0]))
            # print(NOI_mask.shape, edge_mask.shape)
            exclusion = cv2.bitwise_and(edge_mask, NOI_mask)
            edge_mask = edge_mask - exclusion
            
        return edge_mask, slide_mask_ratio

    def get_mask_and_ratio(self, tissue_mask_type = 'sobel'):
        if tissue_mask_type == 'otsu':
            tissue_mask, slide_mask_ratio = self.make_tissue_mask_otsu(self.slide, self.ROI_mask, self.NOI_mask, self.level)
        elif tissue_mask_type == 'sobel':
            tissue_mask, slide_mask_ratio = self.make_tissue_mask_sobel(self.slide, self.ROI_mask, self.NOI_mask, self.level)
        return tissue_mask, slide_mask_ratio

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

def make_patch_(patch):
    """
    corresponding smaller sized gray-scale patch for blurrity check 
    """
    patch = np.array(patch)
    patch = patch.astype(np.float32)
    patch = np.expand_dims(patch, axis=0)
    patch = np.ascontiguousarray(patch, dtype=np.float32)
    return patch



"""
Blankfield Correction
"""
# 영상내 밝기가 높은 값을 찾고, 그 영역의 밝기 평균을 구함
def estimate_blankfield_white(image, ratio=0.01):
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
    if white is None:
        white = estimate_blankfield_white(image, ratio=ratio)
    rgb = image.copy()
    divider = np.zeros_like(rgb)
    for ch in range(0, 3):
        divider[:, :, ch] = white[ch]  
    cv2.divide(rgb, divider, rgb, scale=target)
    return rgb