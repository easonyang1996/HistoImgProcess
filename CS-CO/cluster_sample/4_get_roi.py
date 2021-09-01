#!~/anaconda3/bin/python3
# ******************************************************
# Author: Pengshuai Yang
# Last modified: 2021-07-30 18:49
# Email: yps18@mails.tsinghua.edu.cn
# Filename: get_roi.py
# Description: 
#   save roi 
# ******************************************************
import numpy as np
from PIL import Image
import os
import openslide
import threadpool

import torch
from torchvision.transforms import transforms
from utils import check_directory, get_slide_info, get_mask_info


MICORMETTER_PER_PIXEL = 0.5
SLIDE = '../../diagnostic_slides/'
COOR = '../../sampling_per_slide/{}_um/sample_coors/'.format(MICORMETTER_PER_PIXEL)
slide_info = get_slide_info(SLIDE)  #get short name

MASK = '../../tiles_per_slide/{}_um/masks/'.format(MICORMETTER_PER_PIXEL)
mask_info = get_mask_info(MASK)     #get left top base coor

PATCH_SIZE = 256
RESIZE = 224
EDGE = 7



def get_one_roi(slide_name, coor_list, transform=None): 
    print(slide_name)
    slide = openslide.OpenSlide(SLIDE+slide_info[slide_name])
    
    mag = slide.properties['aperio.AppMag']
    s_scale = MICORMETTER_PER_PIXEL/0.25 if mag=='40' else MICORMETTER_PER_PIXEL/0.5

    left_base, top_base = mask_info[slide_name]
    level0_patch_size = round(PATCH_SIZE*s_scale)
    
    for coor in coor_list:
        left_top_coor = coor+[-(EDGE//2),-(EDGE//2)]
        #print(coor, left_top_coor)
        #print(RELATED_COOR_LIST)

        roi = slide.read_region((left_base+left_top_coor[1]*level0_patch_size,
                                 top_base+left_top_coor[0]*level0_patch_size,), 0,
                                (level0_patch_size*EDGE,
                                 level0_patch_size*EDGE)).convert('RGB').resize((PATCH_SIZE*EDGE,
                                                                                 PATCH_SIZE*EDGE))
        roi_name = slide_name+'_{}_{}.png'.format(coor[0], coor[1])
        #print(roi_name)
        roi.save('../../roi_features/{}_um/roi/{}'.format(MICORMETTER_PER_PIXEL,
                                                                                  roi_name))

    slide.close()
    """
    if transform is None:
        transform = transforms.Compose([transforms.RandomVerticalFlip(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3)])])
    
    roi = transform(roi)
    """
    


def main(roi_coor_list):
    args_list = []
    for name in roi_coor_list:
        slide_name = name.split('_')[0]
        tmp_roi_coor = np.load(COOR+name)
        #for coor in tmp_roi_coor:
        args_list.append(((slide_name, tmp_roi_coor), None))
    pool = threadpool.ThreadPool(8)
    requests = threadpool.makeRequests(get_one_roi, args_list)
    [pool.putRequest(req) for req in requests]
    pool.wait()



if __name__ == '__main__':
    check_directory('../../roi_features/{}_um/roi'.format(MICORMETTER_PER_PIXEL))
    roi_coor_list = os.listdir(COOR)
    #roi_coor_list = ['TCGA-G3-A25U-01Z-00-DX1_coor.npy']
    print('total {} slides!'.format(len(roi_coor_list)))
    
    main(roi_coor_list)

    #get_one_roi('TCGA-G3-A25U-01Z-00-DX1', np.array([48, 16]))


