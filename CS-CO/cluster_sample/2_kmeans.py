import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn import metrics
import time
import os 
from utils import check_directory

MAG = '0.5_um'
CLUSTER = 5


def per_slide_cluster(name):
    '''
    name: slide name 
    '''
    patch_list = np.load('../../sampling_per_slide/{}/name_lists/{}_name.npy'.format(MAG, name))
    embedding = np.load('../../sampling_per_slide/{}/embeddings/{}_embed.npy'.format(MAG, name))

    print(name, patch_list.shape, embedding.shape)

    for i in [CLUSTER]: #[2,3,4,5,6,7,8,9,10]:


        #print('start kmeans!')
        time1 = time.time()
        #kmeans = MiniBatchKMeans(n_clusters=10, max_iter=10, batch_size=5000)
        kmeans = KMeans(n_clusters=i, random_state=1024)
        labels = kmeans.fit_predict(embedding)
        time2 = time.time()
        print('{} finish kmeans, num_cluster: {}, time: {}'.format(name, i, time2-time1))
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



