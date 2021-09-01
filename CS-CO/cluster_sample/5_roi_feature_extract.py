#!~/anaconda3/bin/python3
# ******************************************************
# Author: Pengshuai Yang
# Last modified: 2021-07-23 15:16
# Email: yps18@mails.tsinghua.edu.cn
# Filename: roi_feature_extract.py
# Description: 
#   extract features from rois 
# ******************************************************
import os 
import numpy as np
import random
import argparse
from collections import defaultdict
from roi_dataset import get_testing_set

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim 
import torch.nn as nn 
import timm
from utils import check_directory


import sys
sys.path.append('..')
from model import Linear, Simsiam, Byol, Cs_co, Chen_mia, Xie_miccai 

MAG = '0.5_um'
COOR = '../../sampling_per_slide/{}/sample_coors/'.format(MAG)
SLIDE = '../../diagnostic_slides/'

def get_slide_embedding(method, backbone, roi_coor_list, device):
    if method == 'resnet-random':
        embedding_net = timm.create_model(backbone, pretrained=False, num_classes=0)
    elif method == 'resnet-pretrained':
        embedding_net = timm.create_model(backbone, pretrained=True, num_classes=0)
    elif method == 'byol':
        embedding_net = Byol(backbone, 224, return_embedding=True)
        embedding_state_dict = embedding_net.state_dict()
        pretrain_dict = torch.load('../checkpoint/byol/byol_SGD-cosine_None_256_0.05_1e-06_0.99_100_0.09318.pth')
        state_dict = {k:v for k,v in pretrain_dict.items() if
                      k in embedding_state_dict}
        embedding_net.load_state_dict(state_dict)
    elif method == 'simsiam':
        embedding_net = Simsiam(backbone, 224, return_embedding=True)
        embedding_state_dict = embedding_net.state_dict()
        pretrain_dict = torch.load('../checkpoint/simsiam/simsiam_SGD-cosine_None_256_0.05_1e-06_0.99_100_0.10016.pth')
        state_dict = {k:v for k,v in pretrain_dict.items() if
                      k in embedding_state_dict}
        embedding_net.load_state_dict(state_dict)
    elif method == 'cs':
        embedding_net = Cs_co(backbone, 1, return_embedding=True) 
        embedding_state_dict = embedding_net.state_dict()
        pretrain_dict = torch.load('../../generative_or_discriminative/model/recon_pretrain/0_ckpt0.04339.pth')
        state_dict = {k:v for k,v in pretrain_dict.items() if
                      k in embedding_state_dict}
        embedding_net.load_state_dict(state_dict)
    elif method == 'cs-co':
        embedding_net = Cs_co(backbone, 1, return_embedding=True) 
        embedding_state_dict = embedding_net.state_dict()
        pretrain_dict = torch.load('../datasize_co/checkpoint/all/csco_co_Adam-step_None_96_0.001_1e-06_1.0_9_0.07061.pth')
        #pretrain_dict = torch.load('../datasize_co/checkpoint/10000/csco_co_Adam-step_10000_96_0.001_1e-06_1.0_21_0.07079.pth')
        #pretrain_dict = torch.load('../TCGA_LIHCpoint/csco/csco_co_Adam-step_10000_96_0.001_1e-06_1.0_100_0.06461.pth')
        #pretrain_dict = torch.load('../../generative_or_discriminative/model/fix_decoder_hp/recon_NODE_b+r_Adam-step_10000_0.001_1e-06_1_l2.pth')
        state_dict = {k:v for k,v in pretrain_dict.items() if
                      k in embedding_state_dict}
        embedding_net.load_state_dict(state_dict)
    elif method == 'chen-mia':
        embedding_net = Chen_mia(backbone, 3, return_embedding=True)
        embedding_state_dict = embedding_net.state_dict()
        pretrain_dict = torch.load('../checkpoint/chen/chen_mia_Adam-step_None_64_0.001_1e-08_100_0.00308.pth')
        state_dict = {k:v for k,v in pretrain_dict.items() if
                      k in embedding_state_dict}
        embedding_net.load_state_dict(state_dict)
    elif method == 'xie-miccai':
        embedding_net = Xie_miccai(backbone, 3, return_embedding=True)
        embedding_state_dict = embedding_net.state_dict()
        pretrain_dict = torch.load('../checkpoint/xie/xie_miccai_SGD-step_None_64_0.001_1e-08_100_1.16703.pth')
        state_dict = {k:v for k,v in pretrain_dict.items() if
                      k in embedding_state_dict}
        embedding_net.load_state_dict(state_dict)



    embedding_net.to(device)

    
    single_channel = False if method not in ['cs', 'cs-co'] else True

    
    for name in roi_coor_list:
        slide_name = name.split('_')[0]
        tmp_roi_coor = np.load(COOR+name)
        for coor in tmp_roi_coor:
            test_dataset = get_testing_set(slide_name, coor, single_channel)
            #slide_root = COOR+'{}/'.format(name)
            #test_dataset = get_testing_set(slide_root, single_channel)
            test_loader = DataLoader(test_dataset, batch_size=25, shuffle=False,
                                     drop_last=False, num_workers=4)


            embeddings, coor_list = compute_embedding(test_loader, embedding_net,
                                                      device, single_channel)

            np.savez('../../roi_features/{}/all/{}_{}_{}.npz'.format(MAG, slide_name, coor[0], coor[1]),
                     embeddings=embeddings, coor_list=coor_list)

            print(name, coor)


def compute_embedding(data_loader, embedding_net, device, single_channel): 
    with torch.no_grad():
        embedding_net.eval()
        if single_channel:
            for batch_idx, (h, e, y) in enumerate(data_loader):
                h = h.to(device)
                e = e.to(device)

                tmp_embeddings = embedding_net(h,e)


                tmp_y_true = np.array(y)
                tmp_embeddings = tmp_embeddings.detach().cpu().numpy()

                if batch_idx == 0:
                    y_true = tmp_y_true
                    all_embeddings = tmp_embeddings
                else:
                    y_true = np.concatenate([y_true, tmp_y_true])
                    all_embeddings = np.concatenate([all_embeddings, tmp_embeddings])
        else:
            for batch_idx, (patches, y) in enumerate(data_loader):
                patches = patches.to(device)

                tmp_embeddings = embedding_net(patches)

                tmp_y_true = np.array(y)
                tmp_embeddings = tmp_embeddings.detach().cpu().numpy()

                if batch_idx == 0:
                    y_true = tmp_y_true
                    all_embeddings = tmp_embeddings
                else:
                    y_true = np.concatenate([y_true, tmp_y_true])
                    all_embeddings = np.concatenate([all_embeddings, tmp_embeddings])
    return all_embeddings, y_true

'''
class Narray_Dataset(Dataset):
    def __init__(self, embeddings, labels):
        super(Narray_Dataset, self).__init__()
        self.embeddings = embeddings 
        self.labels = labels

    def __getitem__(self, index):
        x = self.embeddings[index]
        y = self.labels[index]
        return torch.from_numpy(x), y

    def __len__(self):
        return len(self.labels)
'''



if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser(description='script for feature extraction.')
    parser.add_argument('-m', '--method', dest='method', type=str,
                        choices=['resnet-random', 'resnet-pretrained', 'byol',
                                 'simsiam', 'cs', 'cs-co', 'chen-mia', 'xie-miccai'], help='embedding method')
    parser.add_argument('-b', '--backbone', dest='backbone', type=str,
                        default='resnet18', choices=['resnet18', 'resnet50'], 
                        help='backbone')
    # get embedding method and backbone type
    args = parser.parse_args()

    # hyper-parameters
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    check_directory('../../roi_features/{}'.format(MAG))
    check_directory('../../roi_features/{}/all'.format(MAG))


    roi_coor_list = os.listdir(COOR)
    #roi_coor_list = ['TCGA-G3-A25U-01Z-00-DX1_coor.npy']
    
    print('total {} slides!'.format(len(roi_coor_list)))

    get_slide_embedding(args.method, args.backbone, roi_coor_list, device) 


