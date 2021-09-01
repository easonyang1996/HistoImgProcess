#!~/anaconda3/bin/python3
# ******************************************************
# Author: Pengshuai Yang
# Last modified: 2021-09-01 19:30
# Email: yps18@mails.tsinghua.edu.cn
# Filename: utils.py
# Description: 
#   auxillary functions 
# ******************************************************
import os

def check_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_img_list(path):
    total_list = os.listdir(path)
    total_list.sort()
    return total_list

def get_mask_info(path):
    '''get tissue area left top base'''
    mask_info = {}
    mask_list = os.listdir(path)
    for name in mask_list:
        short_name, left_base, top_base = name.split('_')[:3]
        mask_info[short_name] = (int(left_base), int(top_base))

    return mask_info

def get_slide_info(path):
    '''get short name'''
    slide_info = {}
    slide_list = os.listdir(path)
    for name in slide_list:
        short_name = name.split('.')[0]
        slide_info[short_name] = name

    return slide_info
