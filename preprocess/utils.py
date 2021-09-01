#!~/anaconda3/bin/python3
# ******************************************************
# Author: Pengshuai Yang
# Last modified: 2021-08-04 15:10
# Email: yps18@mails.tsinghua.edu.cn
# Filename: utils.py
# Description: 
#   auxillary functions 
# ******************************************************
import os

def check_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_bad_slide_list(txt_path):
    f = open(txt_path,'r')
    bad_slide_list = []
    for l in f.readlines():
        name = l.strip()
        bad_slide_list.append(name)

    return bad_slide_list

def remove_bad_slide(slide_list, bad_list):
    good_slide_list = []
    for item in slide_list:
        name = item.split('.')[0]
        if name not in bad_list:
            good_slide_list.append(item)

    return good_slide_list

