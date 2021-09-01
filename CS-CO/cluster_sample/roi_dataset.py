#!~/anaconda3/bin/python3
# ******************************************************
# Author: Pengshuai Yang
# Last modified: 2021-09-01 19:27
# Email: yps18@mails.tsinghua.edu.cn
# Filename: roi_dataset.py
# Description: 
#       dataset for computing roi embedding 
# ******************************************************
import numpy as np
from PIL import Image
import os
import openslide

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

"""
def get_slide_info(path):
    slide_info = {}
    slide_list = os.listdir(path)
    for name in slide_list:
        short_name = name.split('.')[0]
        slide_info[short_name] = name
    return slide_info
def get_mask_info(path):
    mask_info = {}
    mask_list = os.listdir(path)
    for name in mask_list:
        short_name, left_base, top_base = name.split('_')[:3]
        mask_info[short_name] = (int(left_base), int(top_base))
    return mask_info 

SLIDE = '/work/yangpengshuai/TCGA_LIHC/diagnostic_slides/'
slide_info = get_slide_info(SLIDE)


MICORMETTER_PER_PIXEL = 1

MASK = '/work/yangpengshuai/TCGA_LIHC/tiles_per_slide/{}_um/masks/'.format(MICORMETTER_PER_PIXEL)
mask_info = get_mask_info(MASK)
"""

MICORMETTER_PER_PIXEL = 1
ROI = '../../roi_features/{}_um/roi/'.format(MICORMETTER_PER_PIXEL)
PATCH_SIZE = 256
RESIZE = 224
EDGE = 7

def get_related_coor_list(edge=EDGE):
    chess_board = np.zeros((edge, edge))
    chess_board[::2, ::2]=1
    chess_board[1::2, 1::2]=1
    #print(chess_board)
    related_coor = np.argwhere(chess_board==1)
    
    return related_coor

RELATED_COOR_LIST = get_related_coor_list(EDGE)
#print(RELATED_COOR_LIST)


def get_testing_set(slide_name, coor, single_channel, transform=None):
    """
    #related_coor = get_related_coor_list(coor)
    left_top_coor = coor+[-OFFSET[-1],-OFFSET[-1]]
    #print(coor, left_top_coor)
    #print(RELATED_COOR_LIST)
    slide = openslide.OpenSlide(SLIDE+slide_info[slide_name])

    mag = slide.properties['aperio.AppMag']
    s_scale = MICORMETTER_PER_PIXEL/0.25 if mag=='40' else MICORMETTER_PER_PIXEL/0.5

    left_base, top_base = mask_info[slide_name]
    level0_patch_size = round(PATCH_SIZE*s_scale)

    roi = slide.read_region((left_base+left_top_coor[1]*level0_patch_size,
                             top_base+left_top_coor[0]*level0_patch_size,), 0,
                            (level0_patch_size*EDGE,
                             level0_patch_size*EDGE)).convert('RGB').resize((PATCH_SIZE*EDGE,
                                                                             PATCH_SIZE*EDGE))

    slide.close()
    if transform is None:
        transform = transforms.Compose([transforms.RandomVerticalFlip(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3)])])
    
    roi = transform(roi)
    """
    roi_name = slide_name+'_{}_{}.png'.format(coor[0], coor[1])
    roi = Image.open(ROI+roi_name)
    pil_img_list = []
    for (i, j) in RELATED_COOR_LIST:
        img = roi.crop((j*PATCH_SIZE, i*PATCH_SIZE, (j+1)*PATCH_SIZE,
                        (i+1)*PATCH_SIZE))
        #img.show()
        pil_img_list.append(img)
    return ROI_Dataset(pil_img_list, RELATED_COOR_LIST, single_channel)


class ROI_Dataset(Dataset):
    def __init__(self, pil_img_list, related_coor, single_channel):
        super(ROI_Dataset, self).__init__()
        self.pil_img_list = pil_img_list
        self.related_coor = related_coor
        self.single_channel = single_channel
        self.transform = transforms.Compose([transforms.RandomCrop(224),
                                             transforms.ToTensor()])

    def __getitem__(self, index):
        # need to revise
        if self.single_channel == True:
            H = Image.open(os.path.join(self.data_dir+'H/', patch_name+'_H.png'))
            E = Image.open(os.path.join(self.data_dir+'E/', patch_name+'_E.png'))
            transform = transforms.ToTensor()
            H = transform(H)
            E = transform(E)
            return H, E, patch_name

        else:
            patch = self.pil_img_list[index]
            coor = self.related_coor[index]
            patch = self.transform(patch)
            return patch, '{}_{}'.format(coor[0], coor[1])

    def __len__(self):
        return len(self.pil_img_list)


if __name__ == '__main__':
    data = get_testing_set('TCGA-2Y-A9GS-01Z-00-DX1', np.array([14, 36]), False)
    print(data[0][0].shape)


