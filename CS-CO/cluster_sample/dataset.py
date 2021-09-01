#!~/anaconda3/bin/python3
# ******************************************************
# Author: Pengshuai Yang
# Last modified: 2021-07-15 14:03
# Email: yps18@mails.tsinghua.edu.cn
# Filename: dataset.py
# Description: 
#       dataset for linear evaluation 
# ******************************************************
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

import numpy as np
from PIL import Image
import os


def get_training_set(train_dir, single_channel, train_list=None):
    if train_list is None:
        patches_dir = train_dir + 'patches/'

        assert os.path.exists(patches_dir)
        img_list = os.listdir(patches_dir)
    else:
        img_list = train_list
    return DatasetFromFolder(train_dir, img_list, single_channel)


def get_validating_set(valid_dir, single_channel, valid_list=None):
    if valid_list is None:
        patches_dir = valid_dir + 'patches/'

        assert os.path.exists(patches_dir)
        img_list = os.listdir(patches_dir)
        img_list.sort()
    else:
        img_list = valid_list
    return DatasetFromFolder(valid_dir, img_list, single_channel)


def get_testing_set(test_dir, single_channel, test_list=None):
    if test_list is None:
        patches_dir = test_dir + 'patches/'

        assert os.path.exists(patches_dir)
        img_list = os.listdir(patches_dir)
        img_list.sort()
    else:
        img_list = test_list
    return DatasetFromFolder(test_dir, img_list, single_channel)


class DatasetFromFolder(Dataset):
    def __init__(self, data_dir, img_list, single_channel):
        super(DatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.img_list = img_list
        self.single_channel = single_channel

    def __getitem__(self, index):
        patch_name = self.img_list[index].split('.')[0]

        if self.single_channel == True:
            H = Image.open(os.path.join(self.data_dir+'H/', patch_name+'_H.png'))
            E = Image.open(os.path.join(self.data_dir+'E/', patch_name+'_E.png'))
            transform = transforms.ToTensor()
            H = transform(H)
            E = transform(E)
            return H, E, patch_name

        else:
            patch = Image.open(os.path.join(self.data_dir+'patches/',
                                            self.img_list[index]))
            transform = transforms.Compose([#transforms.Resize(224),
                                            transforms.ToTensor()])
            patch = transform(patch)
            return patch, patch_name

    def __len__(self):
        return len(self.img_list)






