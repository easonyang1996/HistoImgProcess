import numpy as np
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import cv2
from PIL import Image
import mahotas
import pandas as pd
import glob
from utils import check_directory, get_mask_info

MICROMETER_PER_PIXEL = 0.5
MPP = '{}_um'.format(MICROMETER_PER_PIXEL)
PATCH_SIZE = 256
#NUM_BIN = 10

MASK = '../../tiles_per_slide/{}/masks'.format(MPP)
mask_info = get_mask_info(MASK)
#print(mask_info)


def draw_cluster_result(name):

    print(name)
    result = np.load('../../sampling_per_slide/{}/cluster_results/{}_result.npy'.format(MPP, name))

    ori_height = result[:,0].max()
    ori_width = result[:,1].max()

    #print(ori_height, ori_width)
    
    #print(np.unique(result[:,2]))

    height = ori_height//2+1 
    width = ori_width//2+1

    cluster_map = np.zeros((height, width)) 

    for i in range(result.shape[0]):
        cluster_map[result[i,0]//2][result[i,1]//2] = result[i,2]+1

    cluster_map = cluster_map.astype(np.uint8)
    
    plt.figure()
    plt.imshow(cluster_map)
    plt.tight_layout()
    plt.savefig('../../sampling_per_slide/{}/cluster_maps/{}_map.png'.format(MPP, name))
    plt.close()

    #sum_map = np.zeros(cluster_map.shape).astype(np.uint8)
    haralick = np.zeros((height*width, 4))
    df = pd.DataFrame(haralick, columns=['i','j','entropy', 'sum_average'])

    # using two haralick features: entropy and sum_average
    for i in range(3, height-3):
        for j in range(3, width-3):
            if cluster_map[i,j]!=0 and np.sum(cluster_map[i-3:i+4, j-3:j+4]==0)<4:
                
                #sum_map[i][j] = np.unique(cluster_map[i-2:i+3,j-2:j+3]).shape[0]
                #sum_map[i][j] = np.mean(cluster_map[i-2:i+3,j-2:j+3])
                #sum_map[i][j] = np.sum(cluster_map[i-3:i+4,j-3:j+4])/np.sum(cluster_map[i-3:i+4,j-3:j+4]!=0)
                
                tmp_haralick = mahotas.features.haralick(cluster_map[i-3:i+4,j-3:j+4],
                                                         ignore_zeros=True,
                                                         return_mean=True)
                tmp_entropy = tmp_haralick[8]   #6
                tmp_sum_avg = tmp_haralick[5]
                df.loc[i*width+j] = [i, j, tmp_entropy, tmp_sum_avg]

    # drop sum_average==0
    df = df[df['sum_average']!=0].copy().reset_index(drop=True)     
    df = df.sort_values(by='sum_average', ascending=True, ignore_index=True)
    print(df.head())

    NUM_BIN = 2*(len(np.unique(result[:,2])))

    # grouping by sum_average
    #interval = pd.qcut(x['sum_average'], NUM_BIN, labels=False,
    #                   duplicates='drop')      #if labels=False, interval is numpy.ndarray
    interval = pd.cut(df['sum_average'], NUM_BIN, labels=False)
    df['sum_average']=interval
    print(df.head())
    category = list(set(interval.values))
    coors = []
    for c in category:
        tmp_df = df.loc[df['sum_average']==c].copy()
        tmp_df = tmp_df.sort_values(by='entropy', ascending=True, ignore_index=True)
        if len(coors) == 0:
            min_tmp_h, min_tmp_w = tmp_df.iloc[0]['i']*2, tmp_df.iloc[0]['j']*2
            coors.append((min_tmp_h, min_tmp_w))
            max_tmp_h, max_tmp_w = tmp_df.iloc[-1]['i']*2, tmp_df.iloc[-1]['j']*2
            coors.append((max_tmp_h, max_tmp_w))
        else:
            min_idx = 0
            tmp_coor_len = len(coors)
            for _ in range(tmp_df.shape[0]//2):
                flag=0
                min_tmp_h, min_tmp_w = tmp_df.iloc[min_idx]['i']*2, tmp_df.iloc[min_idx]['j']*2
                for i_th, (i, j) in enumerate(coors):
                    if (np.abs(i-min_tmp_h) + np.abs(j-min_tmp_w))<5:
                        min_idx += 1
                        flag=1
                        break  
                if i_th == tmp_coor_len-1 and flag==0:
                    coors.append((min_tmp_h, min_tmp_w))
                    break

            max_idx = -1
            tmp_coor_len = len(coors)
            for _ in range(tmp_df.shape[0]//2):
                flag=0
                max_tmp_h, max_tmp_w = tmp_df.iloc[max_idx]['i']*2, tmp_df.iloc[max_idx]['j']*2
                for i_th, (i, j) in enumerate(coors):
                    if (np.abs(i-max_tmp_h) + np.abs(j-max_tmp_w))<5:
                        max_idx -= 1
                        flag=1
                        break
                if i_th == tmp_coor_len-1 and flag==0:
                    coors.append((max_tmp_h, max_tmp_w))
                    break

    coors = list(set(coors))
    #print(coors)

    np.save('../../sampling_per_slide/{}/sample_coors/{}_coor.npy'.format(MPP,name),
            np.array(coors).astype(np.int))

    thumb_name = glob.glob('../../thumbnails/{}*'.format(name))[0] 
    mag = thumb_name.split('/')[-1].split('_')[1]
    thumb_scale = 128 if mag=='40' else 64
    s_scale = MICROMETER_PER_PIXEL/0.25 if mag=='40' else MICROMETER_PER_PIXEL/0.5

    step = round(PATCH_SIZE*s_scale)//thumb_scale

    left, top = mask_info[name]

    sample_map = cv2.imread(thumb_name)
    
    for (i, j) in coors:
        left_top = (int(left/thumb_scale+j*step), int(top/thumb_scale+i*step))
        right_bottom = (int(left/thumb_scale+j*step+step),
                        int(top/thumb_scale+i*step+step))
        cv2.rectangle(sample_map, left_top, right_bottom, 255,1)
        
        #sum_map[int(i//2),int(j//2)]=20
    cv2.imwrite('../../sampling_per_slide/{}/sample_maps/{}_sample_map.png'.format(MPP,
                                                                                   name),
                sample_map)
    '''
    plt.figure()
    plt.imshow(sum_map)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('../../sampling_per_slide/{}/sample_maps/{}_smp_map.png'.format(MPP, name))
    #plt.show()
    plt.close()
    '''
    

if __name__ == '__main__':
    slide_list = np.load('./slide_list_{}.npy'.format(MPP))
    #slide_list = ['TCGA-DD-AAVS-01Z-00-DX1']
    check_directory('../../sampling_per_slide/{}/cluster_maps'.format(MPP))
    check_directory('../../sampling_per_slide/{}/sample_coors'.format(MPP))
    check_directory('../../sampling_per_slide/{}/sample_maps'.format(MPP))
    for name in slide_list:
        draw_cluster_result(name)
