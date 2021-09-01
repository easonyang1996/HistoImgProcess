#!~/anaconda3/bin/python3
# ******************************************************
# Author: Pengshuai Yang
# Last modified: 2021-09-01 19:20
# Email: yps18@mails.tsinghua.edu.cn
# Filename: 3_get_slide_patches.py
# Description: 
#   Get slide patches for outcome prediction.
# ******************************************************
import numpy as np
import os
import cv2
import shutil
#import xml.etree.ElementTree as ET
#import xml.dom.minidom as minidom
import openslide
import threadpool
import random
from tqdm import tqdm
#import imgaug as ia
#from imgaug import augmenters as iaa 
#from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from PIL import Image

from utils import check_directory, get_bad_slide_list, remove_bad_slide

'''
import sys
sys.path.append('..')
from color_deconvolution import HE_Preprocessor
from color_normalizer import normalizeStaining
'''


"""
in openslide (width, height)
in numpy     (height, width)
"""

IMAGES_ROOT = '../diagnostic_slides/'
TILES_ROOT = '../tiles_per_slide/'
BAD_SLIDE_LIST = '../bad_slide_list.txt'

MICROMETTER_PER_PIXEL = 0.5
STRIDE = 2
#MICROMETTER_PER_PIXEL = 1   
#STRIDE = 2 
FG_THRESHOLD = 0.5
PATCH_SIZE = 256
SELECT_RATIO = 1
TILES_PER_SLIDE = 1000000


DATASET_ROOT = TILES_ROOT + '{}_um/'.format(MICROMETTER_PER_PIXEL)


def generate_data(name_list):
    '''
    make patches in multi-thread. 
    '''
    
    pool = threadpool.ThreadPool(16)
    requests = threadpool.makeRequests(make_patch, name_list)
    [pool.putRequest(req) for req in requests]
    pool.wait()


def make_patch(name):
    '''
    extract patches from single slide. 
    '''
    slide_name = name.split('.')[0]
    root = DATASET_ROOT

    check_directory(root+'slides/{}/'.format(slide_name))
    check_directory(root+'slides/{}/patches/'.format(slide_name))

    slide_path = IMAGES_ROOT + name
 
    slide = openslide.OpenSlide(slide_path)

    mag = slide.properties['aperio.AppMag']
    s_scale = MICROMETTER_PER_PIXEL/0.25 if mag=='40' else MICROMETTER_PER_PIXEL/0.5

    thumb_scale = 128 if mag=='40' else 64
    level0_dimension = slide.level_dimensions[0]
    thumb_size = (level0_dimension[0]//thumb_scale, level0_dimension[1]//thumb_scale)


    print('slide: {}, level_0_mag: {}, level0_dimension:{}, scale_to_{}_um: {}, thumb_scale: {}, thumb_dimension: {}'.format(slide_name, mag, level0_dimension, MICROMETTER_PER_PIXEL, s_scale, thumb_scale, thumb_size))    
    
    thumbnail = np.array(slide.get_thumbnail(thumb_size).convert('RGB')) #RGB
    thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2BGR)
    '''GRAY + OTSU 
    thumbnail_gray = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(thumbnail_gray, 0, 255,
                            cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = 255-mask
    
    '''
    thumbnail_hsv = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2HSV)
    _, mask = cv2.threshold(thumbnail_hsv[...,1], 0, 255,
                            cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    mask = np.array(mask)
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    areas = stats[1:,4]     #get the area of each connected component
    label_cc = areas.argmax()+1    #the label of the biggest connected component exclude background
    box_left, box_top, box_width, box_height = stats[label_cc, :4]
    mask[labels!=label_cc] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    left_base = box_left * thumb_scale
    top_base = box_top * thumb_scale

    level0_patch_size = round(PATCH_SIZE * s_scale)
    step = level0_patch_size // thumb_scale
    width = box_width // step  
    height = box_height // step
    #print(level0_patch_size, step, width, height)

    count = 0
    mask_max = step * step * 255. 

    exit_flag = 0

    t = tqdm(range(height))
    for i in t:
        if i % STRIDE != 0:
            continue
        if count == TILES_PER_SLIDE:
            break
        for j in range(width):
            if j % STRIDE != 0:
                continue
            if count == TILES_PER_SLIDE:
                break
            
            ran = random.random()
            mask_sum = mask[box_top+step*i:box_top+step*(i+1),
                            box_left+step*j:box_left+step*(j+1)].sum()
            foreground_ratio = mask_sum / mask_max

            if foreground_ratio>FG_THRESHOLD and ran <= SELECT_RATIO:
                count += 1
                patch_name = slide_name + '_' + str(i) + '_' + str(j)
                #print(patch_name)
                patch = slide.read_region((left_base+j*level0_patch_size,
                                           top_base+i*level0_patch_size), 0,
                                          (level0_patch_size,
                                           level0_patch_size)).convert('RGB')
                patch = patch.resize((PATCH_SIZE,PATCH_SIZE))
                patch.save(root+'slides/{}/patches/'.format(slide_name)+patch_name+'.png')
                '''
                #color deconvolution
                patch = HE_Preprocessor(patch).get_hematoxylin()
                cv2.imwrite(patch_name, patch)
                '''
                '''
                # color normalization
                _ = normalizeStaining(np.array(patch.convert('RGB')), path=root, saveFile=patch_name)
                #Image.fromarray(patch).save(patch_name)
                '''
                cv2.rectangle(thumbnail, (box_left+step*j, box_top+step*i),
                              (box_left+step*j+step, box_top+step*i+step),
                              (0,255,0), 1)
        
        t.set_postfix({'slide':slide_name, 'num_patches':count})
                    
    cv2.rectangle(thumbnail, (box_left, box_top),
                  (box_left+box_width, box_top+box_height), (255,0,0), 1)
    cv2.rectangle(mask, (box_left, box_top),
                  (box_left+box_width, box_top+box_height), (255,0,0), 1)

    cv2.imwrite(root+'/thumbnails/'+slide_name+'_thumbnail.png', thumbnail)

    cv2.imwrite(root+'/masks/'+slide_name+'_{}_{}_mask.png'.format(left_base,
                                                                  top_base), mask)  
    slide.close()



if __name__== '__main__':
    
    tot_name_list = os.listdir(IMAGES_ROOT)
    #tot_name_list = ['TCGA-DD-AAVS-01Z-00-DX1.F0E46BAD-903A-4325-B761-BBD9BAA85815.svs']
    bad_slide_list = get_bad_slide_list(BAD_SLIDE_LIST)

    name_list = remove_bad_slide(tot_name_list, bad_slide_list)

    print('{} slides in total, {} bad slides, {} good slides.'.format(len(tot_name_list), len(bad_slide_list), len(name_list)))

    check_directory(DATASET_ROOT)
    check_directory(os.path.join(DATASET_ROOT, 'slides'))
    check_directory(os.path.join(DATASET_ROOT, 'thumbnails'))
    check_directory(os.path.join(DATASET_ROOT, 'masks'))

    generate_data(name_list) 
    
