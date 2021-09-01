#!~/anaconda3/bin/python3
# ******************************************************
# Author: Pengshuai Yang
# Last modified: 2021-07-14 09:46
# Email: yps18@mails.tsinghua.edu.cn
# Filename: recon_test.py
# Description: 
#   the script to test the reconstruction performance 
# ******************************************************
import os
import sys
import numpy as np
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.optim import lr_scheduler 
from torch.utils.data import DataLoader

from csco_dataset import get_validating_set

sys.path.append('..')
from model import Cs_co


TEST_ROOT = '../../1_um/ss_dataset/'
RECONSTRUCT = './test_img/'

class Recon_Loss(nn.Module):
    def __init__(self, l1orl2):
        super(Recon_Loss, self).__init__()
        if l1orl2 == 'l1':
            self.fn = nn.L1Loss()
        elif l1orl2 == 'l2':
            self.fn = nn.MSELoss()

    def forward(self, pred_e, pred_h, h, e):
        loss = self.fn(pred_e, e)+self.fn(pred_h, h)
        return loss

def numpy2img(out_cpu):
    #out_cpu = np.transpose(out_cpu, (1,2,0))
    out_cpu = out_cpu * 255.
    out_cpu = np.clip(out_cpu, 0, 255)
    out_cpu = out_cpu.astype(np.uint8)
    out_cpu = Image.fromarray(out_cpu)
    return out_cpu 

def eval_epoch(eval_loader, model, loss_fn, device, model_type='cs'):
    with torch.no_grad():
        model.eval()
        val_loss = 0
        name_list = tuple()
        for batch_idx, (H, E, img_names) in enumerate(eval_loader):
            
            H = H.to(device)
            E = E.to(device)
            
            
            pred_one_e, pred_one_h = model(H, E)
            loss_outputs = loss_fn(pred_one_e, pred_one_h, H, E)
                                    
            val_loss += loss_outputs.item()

            name_list += img_names
            
            ### if model_type == recon / b+r
            for i in range(pred_one_e.size(0)):
                H_in = numpy2img(H[i,0,...].cpu().numpy())
                H_in.save(RECONSTRUCT+'H_in/'+img_names[i]+'_H_in.png')
                E_in = numpy2img(E[i,0,...].cpu().numpy())
                E_in.save(RECONSTRUCT+'E_in/'+img_names[i]+'_E_in.png')

                E_out = pred_one_e[i,0,...].cpu().numpy()
                #print(H_out.shape)
                E_img = numpy2img(E_out)
                E_img.save(RECONSTRUCT+'E_pred/'+img_names[i]+'_E.png')
                H_out = pred_one_h[i,0,...].cpu().numpy()
                H_img = numpy2img(H_out)
                H_img.save(RECONSTRUCT+'H_pred/'+img_names[i]+'_H.png')
            
        print('test loss {:.6f}'.format(val_loss/(batch_idx+1)))    


def data_list(path, k=None):
    img_list = os.listdir(path+'patches/')
    random.seed(10)
    random.shuffle(img_list)
    if k==None:
        return img_list
    else:
        return img_list[:k]


if __name__ == '__main__':
    # define gpu devices
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_type = 'cs'
    recon_pretrained = '../checkpoint/csco_cs_SGD-step_None_64_0.01_1e-08_1.0_95_0.08166.pth'
    model = Cs_co('resnet18', 1, 'cs', False, 'None', 1, False)
    model.load_state_dict(torch.load(recon_pretrained))

    if torch.cuda.device_count() > 1:
        print("Let's use {} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model) 
    model.to(device)
    
    patch_list = data_list(TEST_ROOT, k=32) 

    test_dataset = get_validating_set(TEST_ROOT, patch_list)
    test_loader = DataLoader(test_dataset, batch_size=32, 
                             shuffle=False, drop_last=False)

    loss_fn = Recon_Loss('l2')
   
    eval_epoch(test_loader, model, loss_fn, device)
    

    


