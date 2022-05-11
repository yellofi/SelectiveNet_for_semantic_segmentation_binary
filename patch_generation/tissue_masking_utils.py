import numpy as np
import cv2


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

def make_tissue_mask_otsu(slide, ROI_mask):
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

def make_tissue_mask_sobel(slide, ROI_mask):

    """
    Args:
        slide: WSI (Whole Slide Image), slide = openslide.OpenSlide('/~/*.svs')
        ROI mask (np.array): ROI (Region of Interest) mask, a binary mask whose size is the same with the lowest resolution (level 3) of slide

    get a tissue mask on ROI, using Edge Detection

    RGB2GRAY -> Edge Detection (Sobel) -> Morphological Op. (Closing and Opening) 

    Return:
        edge mask (np.array): a gray image (binary mask)
    """

    if slide.level_count < 4:
        level_index = slide.level_count-1
    else:
        level_index = 3

    slide_thumbnail = slide.get_thumbnail(slide.level_dimensions[level_index]) 
    slide_thumbnail = np.array(slide_thumbnail.convert('L')) 

    mask_slide_ratio = round(slide.level_downsamples[level_index])

    edge_mask = cv2.bitwise_and(slide_thumbnail, ROI_mask)
    edge_mask = cv2.Sobel(src=edge_mask, ddepth=cv2.CV_64F, dx=1,dy=1, ksize=5)
    # edge_mask = cv2.convertScaleAbs(edge_mask) # to 0 ~ 255, 큰 상관없음

    # edge detection이 목적이면 sobel edge detection을 x방향, y방향으로 따로 한 뒤 합쳐주는 게 맞지만,
    # tissue를 background와 구분하기 위함이라면 앞에 sobel edge detection을 xy방향으로 한 번만 해줘도 크게 문제될 게 없음

    kernel = np.ones((11, 11), dtype=np.uint8)

    edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
    edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_OPEN, kernel)

    # print(edge_mask.dtype, edge_mask.shape, edge_mask.max(), edge_mask.min(), edge_mask.mean())

    return edge_mask, mask_slide_ratio


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