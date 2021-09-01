#!~/anaconda3/bin/python3
# ******************************************************
# Author: Pengshuai Yang
# Last modified: 2021-09-01 19:17
# Email: yps18@mails.tsinghua.edu.cn
# Filename: 0_get_slide_thumb.py
# Description: 
#   Get slide thumb for slide checking!
# ******************************************************
import numpy as np
import os
import cv2
import shutil
import openslide
import threadpool
import random

from PIL import Image
from utils import check_directory, get_bad_slide_list, remove_bad_slide

IMAGES_ROOT = '../diagnostic_slides/'
DATASET_ROOT = '../thumbnails/'
BAD_SLIDE_LIST = '../bad_slide_list.txt'


def generate_data(name_list):
    '''
    make slide thumbnails in multi-threads.
    '''
    pool = threadpool.ThreadPool(16)
    requests = threadpool.makeRequests(make_thumbnail, name_list)
    [pool.putRequest(req) for req in requests]
    pool.wait()


def make_thumbnail(name):
    '''
    make thumbnail of single slide.
    '''
    slide_name = name.split('.')[0]
    root = DATASET_ROOT

    slide_path = IMAGES_ROOT + name
 
    slide = openslide.OpenSlide(slide_path)

    #s_x_mpp = float(slide.properties['openslide.mpp-x'])
    #s_scale = MICROMETTER_PER_PIXEL / s_x_mpp

    mag = slide.properties['aperio.AppMag']
    s_scale = 128 if mag =='40' else 64

    level0_dimension = slide.level_dimensions[0]
    thumb_size = (level0_dimension[0]//s_scale, level0_dimension[1]//s_scale)

    print('slide: {}, thumb_size: {}'.format(slide_name, thumb_size))
    thumbnail = slide.get_thumbnail(thumb_size).convert('RGB')
    thumbnail.save(root + slide_name +'_{}_thumbnail.png'.format(mag))

    slide.close()
    

if __name__== '__main__':
    
    name_list = os.listdir(IMAGES_ROOT)
    #name_list = ['TCGA-DD-AAVS-01Z-00-DX1.F0E46BAD-903A-4325-B761-BBD9BAA85815.svs']

    check_directory(DATASET_ROOT)
    
    generate_data(name_list) 
    
