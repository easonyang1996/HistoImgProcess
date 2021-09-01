import numpy as np
#from sklearn.cluster import MiniBatchKMeans, KMeans
import scanpy as sc
from sklearn import metrics
import time
import os 

MAG = '1_um'

def check_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def per_slide_cluster(name):
    '''
    name: slide name 
    '''
    patch_list = np.load('../../sampling_per_slide/{}/name_lists/{}_name.npy'.format(MAG, name))
    embedding = np.load('../../sampling_per_slide/{}/embeddings/{}_embed.npy'.format(MAG, name))

    print(name, patch_list.shape, embedding.shape)

    #print('start kmeans!')
    time1 = time.time()

    adata = sc.AnnData(embedding)
    sc.pp.neighbors(adata, use_rep='X')
    sc.tl.leiden(adata, resolution=0.75, key_added='leiden')

    #kmeans = MiniBatchKMeans(n_clusters=10, max_iter=10, batch_size=5000)
    #kmeans = KMeans(n_clusters=i, random_state=1024)
    #labels = kmeans.fit_predict(embedding)
    time2 = time.time()
    labels = np.array(adata.obs['leiden'].apply(int).values)
    print('{} finish kmeans, num_cluster: {}, time: {}'.format(name,
                                                               labels.max()+1, time2-time1))
    print('silhouette coefficient: {}'.format(metrics.silhouette_score(embedding, labels)))
    #print('CH index: {}'.format(metrics.calinski_harabaz_score(embedding, labels)))

    results = np.zeros((labels.shape[0], 3)) #i,j,cluster
    
    for i in range(labels.shape[0]):
        _, row_th, col_th = patch_list[i].split('_')
        results[i] = int(row_th), int(col_th), labels[i]
        

    #print(labels.shape)
    np.save('../../sampling_per_slide/{}/cluster_results/{}_result.npy'.format(MAG, name), results.astype(np.int))
 
if __name__ == '__main__':
    slide_list = np.load('./slide_list_{}.npy'.format(MAG))
    #slide_list = ['TCGA-DD-AAVS-01Z-00-DX1']
    check_directory('../../sampling_per_slide/{}/cluster_results'.format(MAG))
    for name in slide_list:
        per_slide_cluster(name)



