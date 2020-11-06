from sklearn.datasets import load_digits
import os, glob, random, math, operator, umap
import numpy as np
import nibabel as nii
#import patch_extraction
from scipy.ndimage.interpolation import zoom
from keras.models import load_model
from scipy import ndimage
import scipy.io as sio
import modelos
import statsmodels.api as sm
from scipy.signal import argrelextrema
from collections import OrderedDict, defaultdict
from skimage import measure
from scipy.stats import pearsonr
from keras import backend as K
import time,umap
import matplotlib
import Data_fast, data_augmentation
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import patch_extraction
from sklearn.decomposition import PCA
import data_augmentation
from sklearn.cluster import KMeans



def get_bottleneck_features_func(model):
    input1 = model.input               # input placeholder
    output1 = model.get_layer('bottleneck').output# all layer outputs
    fun = K.function([input1, K.learning_phase()],[output1])# evaluation function
    return fun

def features_from_names(listaT1,listaFLAIR,fun,listaMASK=None):
    file_names=[]

    for i in range(len(listaT1)):
        #print(listaT1[i])

        try:
            T1,FLAIR=load_modalities(listaT1[i],listaFLAIR[i],listaMASK[i])

        except:
            T1,FLAIR=load_modalities(listaT1[i],listaFLAIR[i])
        T1_cropped,FLAIR_cropped=to969696(T1,FLAIR)
        x_in=np.concatenate((np.expand_dims(T1_cropped,axis=3),np.expand_dims(FLAIR_cropped,axis=3)), axis=3)
        x_in=np.expand_dims(x_in,axis=0)
        #bottleneck_features[listaT1_isbi[i]]=bottleneck_out[0].mean(axis=(0,1,2,3))
        bottleneck_out = fun([x_in, 1.])
        bottleneck_out_mean=bottleneck_out[0].mean(axis=(0,1,2,3))
        if(i==0):
            bottleneck_features=np.expand_dims(bottleneck_out_mean,axis=0)
        else:
            #return np.repeat(bottleneck_features,len(listaT1),axis=0),file_names
            bottleneck_features=np.concatenate((bottleneck_features,np.expand_dims(bottleneck_out_mean,axis=0)),axis=0)
        file_names.append(listaT1[i])
    return bottleneck_features,file_names

def  get_x(list_np):
    out=list_np[0]
    for i in range(1,len(list_np)):
        out=np.concatenate((out,list_np[i]),axis=0)
    return out


def reducer_umap(x):
    start = time.time()
    reducer = umap.UMAP().fit(x)
    end = time.time()
    print(end - start)
    return reducer

def save_plot(reducer,plotname='plot_before.eps',labeled_num=21,pseudolabeled_num=0,unlabeled_num=2901):
    labels=np.concatenate((  np.ones(labeled_num)*0 ,  np.ones(pseudolabeled_num)*1, np.ones(unlabeled_num)*2 ))
    red_vis=np.concatenate((reducer.embedding_,labels.reshape((-1,1))),axis=1)
    tick = ['Labeled','pseudo-Labeled','Unlabeled']
    colors = ['green','blue','red']
    plt.scatter(   red_vis [:,0], red_vis [:,1], s=0.01,c=red_vis [:,2].astype(int), cmap=matplotlib.colors.ListedColormap(colors))
    cb = plt.colorbar()
    loc = np.arange(0,max(red_vis [:,2].astype(int)),max(red_vis [:,2].astype(int))/float(len(tick)))
    cb.set_ticks(loc)
    cb.set_ticklabels(tick)
    plt.savefig(plotname, format='eps')
    plt.clf()
    plt.cla()
    plt.close()

def normalize_image(vol, contrast):
    # copied from FLEXCONN
    # slightly changed to fit our implementation
    temp = vol[np.nonzero(vol)].astype(float)
    q = np.percentile(temp, 99)
    temp = temp[temp <= q]
    temp = temp.reshape(-1, 1)
    bw = q / 80
    # print("99th quantile is %.4f, gridsize = %.4f" % (q, bw))

    kde = sm.nonparametric.KDEUnivariate(temp)

    kde.fit(kernel='gau', bw=bw, gridsize=80, fft=True)
    x_mat = 100.0 * kde.density
    y_mat = kde.support

    indx = argrelextrema(x_mat, np.greater)
    indx = np.asarray(indx, dtype=int)
    heights = x_mat[indx][0]
    peaks = y_mat[indx][0]
    peak = 0.00
    # print("%d peaks found." % (len(peaks)))

    # norm_vol = vol
    if contrast.lower() in ["t1", "mprage"]:
        peak = peaks[-1]
        # print("Peak found at %.4f for %s" % (peak, contrast))
        # norm_vol = vol/peak
        # norm_vol[norm_vol > 1.25] = 1.25
        # norm_vol = norm_vol/1.25
    elif contrast.lower() in ['t2', 'pd', 'flair', 'fl']:
        peak_height = np.amax(heights)
        idx = np.where(heights == peak_height)
        peak = peaks[idx]
        # print("Peak found at %.4f for %s" % (peak, contrast))
        # norm_vol = vol / peak
        # norm_vol[norm_vol > 3.5] = 3.5
        # norm_vol = norm_vol / 3.5
    else:
        print("Contrast must be either t1,t2,pd, or flair. You entered %s. Returning 0." % contrast)

    # return peak, norm_vol
    return peak

def load_modalities(T1_name,FLAIR_name,MASK_name=None):
    T1_img = nii.load(T1_name)
    T1=T1_img.get_data()
    T1=T1.astype('float32')
    FLAIR_img = nii.load(FLAIR_name)
    FLAIR=FLAIR_img.get_data()
    FLAIR=FLAIR.astype('float32')
    if(not MASK_name==None):
        MASK_img = nii.load(MASK_name)
        MASK = MASK_img.get_data()
        MASK=MASK.astype('int')
        T1=T1*MASK
        FLAIR=FLAIR*MASK
    peak = normalize_image(T1, 't1')
    T1=T1/peak
    peak = normalize_image(FLAIR, 'flair')
    FLAIR=FLAIR/peak
    return T1,FLAIR

def crop_center(img,cropx,cropy,cropz):
    x,y,z = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    startz = z//2-(cropz//2)
    return img[starty:starty+cropy,startx:startx+cropx,startz:startz+cropz]

def to969696(T1,FLAIR):
    T1=crop_center(T1,96,96,96)
    FLAIR=crop_center(FLAIR,96,96,96)
    return T1,FLAIR

def graph_based_rank(X, Q):
    Q=Q.T
    X=X.T
    #X the Data, unlabeled
    #Q the querries, the labeled and pseudo
    K = 100 # approx 50 mutual nns
    QUERYKNN = 21#10
    R = 2000
    alpha = 0.9

    start = time.time()

    sim  = np.dot(X.T, Q)
    #""""
    qsim = sim_kernel(sim).T

    sortidxs = np.argsort(-qsim, axis = 1)
    for i in range(len(qsim)):
        qsim[i,sortidxs[i,QUERYKNN:]] = 0
    #"""
    qsim = sim_kernel(qsim)
    A = np.dot(X.T, X)
    W = sim_kernel(A).T
    W = topK_W(W, K)
    Wn = normalize_connection_graph(W)

    #plain_ranks = np.argsort(-sim, axis=0)
    cg_ranks =  cg_diffusion(qsim, Wn, alpha)
    #cg_trunk_ranks =  dfs_trunk(sim, A, alpha = alpha, QUERYKNN = QUERYKNN )
    #fast_spectral_ranks = fsr_rankR(qsim, Wn, alpha, R)

    end = time.time()
    print(end - start)
    return cg_ranks

def get_clusters_and_centroids(unlabeled_tsne,labeled_tsne):
    kmeans_u = KMeans(n_clusters=29, random_state=0).fit(unlabeled_tsne)
    kmeans_l = KMeans(n_clusters=1, random_state=0).fit(labeled_tsne)
    classes_u=kmeans_u.labels_
    classes_l=kmeans_l.labels_
    centroids_u=kmeans_u.cluster_centers_
    centroids_l=kmeans_l.cluster_centers_
    return classes_u,classes_l,centroids_u,centroids_l

def n_times_get_cluster_by_cluster(cluster_dis_from_labeled,classes_u, choice_type='closest'):
    bundles=[]
    clusters=[]
    cluster_number = classes_u.max()+1
    for i in range(cluster_number):
        cluster, indx_of_cluster_element= get_cluster_by_cluster(cluster_dis_from_labeled,classes_u, choice_type)
        bundles.append(indx_of_cluster_element)
        clusters.append(cluster)
    return clusters,bundles

def bundle_from_each_cluster(classes_u, number_of_bundles=29, bundle_size=100):
    bundles=[]
    max_classes=classes_u.max()+1
    not_selected= np.ones(classes_u.shape).astype('bool')
    cluster=0
    classes_u_idx=np.arange(classes_u.shape[0])
    np.random.seed(0)
    #while(not_selected.any()):
    for bundle_num in range(number_of_bundles):
        print(bundle_num)
        bundle=[]
        for i in range(bundle_size):
            idx=classes_u_idx[(classes_u==cluster) * not_selected]
            #print(idx)
            if(not len(idx)==0):
                np.random.shuffle(idx)
                bundle.append(idx[0])
                not_selected[idx[0]]=False
            else:
                i=i-1
            cluster=cluster+1
            if(cluster== max_classes):
                cluster=0
        bundles.append(bundle)


    return bundles

def get_cluster_by_cluster(cluster_dis_from_labeled,classes_u, choice_type='closest'):
    if (choice_type=='closest'):
        cluster= np.argmin(cluster_dis_from_labeled)
        cluster_dis_from_labeled[cluster]=9999999
    if (choice_type=='farest'):
        cluster= np.argmax(cluster_dis_from_labeled)
        cluster_dis_from_labeled[cluster]=0
    indx_of_cluster_element=np.where(classes_u==cluster)
    return cluster, indx_of_cluster_element

def give_n_closest_loop(ranks,n_indxs=100):
    indxs=[]
    querries_num=ranks.shape[1]
    #datapoints_num=ranks.shape[0]
    for j in range(n_indxs//querries_num +1 ):
        for i in range(querries_num):
            indx=ranks[:,i].argmin()
            #indx=ranks[:,i].argmax()
            indxs.append(indx)
            ranks[indx,:]=999999
            #ranks[indx,:]=0
    return indxs


def give_n_closest(ranks,n_indxs=100):
    indxs=[]
    querries_num=ranks.shape[1]
    for i in range(n_indxs):
        indx=np.unravel_index(np.argmin(ranks, axis=None), ranks.shape)[0]
        indxs.append(indx)
        ranks[indx,:]=9999999
    return indxs

def give_dist_for_Kclosest(ranks,n_indxs=100,k=50):
    indxs=[]
    querries_num=ranks.shape[1]
    argpartition=np.argpartition(ranks, k, axis=1)[:,:k]
    Kclosest=np.zeros(ranks.shape)
    for i in range(ranks.shape[0]):
        Kclosest[i,argpartition[i,:]]=1
    values_for_distance= Kclosest*ranks
    distance= np.sum(values_for_distance,axis=1)
    indxs=np.argpartition(distance, n_indxs)[:n_indxs]
    ranks[indxs,:]=9999999
    return indxs.tolist()

def give_dist_for_Kfarest(ranks,n_indxs=100,k=50):
    indxs=[]
    argpartition=np.argpartition(ranks, ranks.shape[1]-k, axis=1)[:,ranks.shape[1]-k:]
    Kfarest=np.zeros(ranks.shape)
    for i in range(ranks.shape[0]):
        Kfarest[i,argpartition[i,:]]=1
    values_for_distance= Kfarest*ranks
    distance= np.sum(values_for_distance,axis=1)
    indxs=np.argpartition(distance, distance.shape[0]-n_indxs)[distance.shape[0]-n_indxs:]
    ranks[indxs,:]=0
    return indxs.tolist()

def give_n_farest(ranks,n_indxs=100):
    indxs=[]
    querries_num=ranks.shape[1]
    for i in range(n_indxs):
        indx=np.unravel_index(np.argmax(ranks, axis=None), ranks.shape)[0]
        indxs.append(indx)
        ranks[indx,:]=0
    return indxs

#np.unravel_index(np.argmin(a, axis=None), a.shape)

def distance_measure(x1,x2):
    return np.sum((x1-x2)**2)
from sklearn.manifold import TSNE

def pca_rank(X, Q,n_components=16):
    x=np.concatenate((X,Q),axis=0)
    pca = PCA(n_components=n_components)
    x= pca.fit(x).transform(x)
    print(x)
    X,Q= x[:X.shape[0]],x[X.shape[0]:]
    return brute_force_rank(X, Q)

def tsne_rank(X, Q,n_components=16):
    X, Q= tsne(X, Q,n_components)
    return brute_force_rank(X, Q)

def tsne(X, Q,n_components=16):
    x=np.concatenate((X,Q),axis=0)
    tsne = TSNE(n_components=n_components)
    x= tsne.fit_transform(x)
    #print(x)
    X,Q= x[:X.shape[0]],x[X.shape[0]:]
    return X, Q

def brute_force_rank(X, Q):
    labeled_num=Q.shape[0]
    unlabeled_num=X.shape[0]
    res_dis=np.zeros((unlabeled_num,labeled_num))
    for i in range(labeled_num):
        #print(i)
        for j in range(unlabeled_num):
            res_dis[j,i]=distance_measure(X[j],Q[i])
    return res_dis


def load_isbi(one_out):
    Rootpath=os.getcwd()
    # number of networks per dimension
    nbNN=[5,5,5]
    ps=[96,96,96]
    np.random.seed(43)
    lib_path = os.path.join("..","lib","isbi_final_train_preprocessed")

    for i in range(0,75,3):

        if((i%2)==0):
            x_train_,y_train_,path,listaT1,lista =Data_fast.read_Tile_1mm_symetrie(i,lib_path,nbNN,yname='tiles_lab1_1mm_')
        else:
            x_train_,y_train_,path,listaT1,lista =Data_fast.read_Tile_1mm_symetrie(i,lib_path,nbNN,yname='tiles_lab2_1mm_')

        x_train_=x_train_[:,:,:,:,0:2]


        x_train_=np.concatenate((x_train_,y_train_),axis=4)
        idx_random=np.arange(x_train_.shape[0])
        np.random.shuffle(idx_random)
        x_train_=x_train_[idx_random]
        x_train_=x_train_[:11]
        x_train_=batch_rot90(x_train_)

        y_train_=x_train_[:,:,:,:,2:4]
        x_train_=x_train_[:,:,:,:,0:2]

        if(one_out):
            y_train_=y_train_[:,:,:,:,1:2]

        if(i==0):
            x_train=x_train_
            y_train=y_train_

        else:
            x_train=np.concatenate((x_train,x_train_),axis=0)
            y_train=np.concatenate((y_train,y_train_),axis=0)

    train_index = [g for g in range(y_train.shape[0])]
    np.random.shuffle(train_index)
    index_image = np.round(y_train.shape[0]*0.1)
    index_image = index_image.astype(int)

    x_val=x_train[train_index[-index_image:]]
    y_val=y_train[train_index[-index_image:]]
    x_train=x_train[train_index[0:-index_image]]
    y_train=y_train[train_index[0:-index_image]]

    return x_train,y_train,x_val,y_val



def data_gen_iqda_2it(datafolder,train_files_bytiles,same_ratio=0.33,sim='DICE'):
    while(1):
        list_x=sorted(glob.glob(datafolder+"x*.npy"))
        list_y=sorted(glob.glob(datafolder+"y*.npy"))
        random_idxs1= np.arange(len(list_x))
        #random_idxs2= np.arange(len(list_x))
        np.random.shuffle(random_idxs1)
        #np.random.shuffle(random_idxs2)
        for i in range(len(list_x)):
            op=np.random.choice(2,1,p=[same_ratio,1-same_ratio]) # 0 same 1 different
            if(op==0):
                x_1=np.load(list_x[random_idxs1[i]])
                y_1=np.load(list_y[random_idxs1[i]])
                x_2=np.load(list_x[random_idxs1[i]])
                y_2=np.load(list_y[random_idxs1[i]])
                if(sim=='DICE'):
                    inter_sim=1.0
                if(sim=='DICE_norm'):
                    inter_sim=(1.0-0.15)*4
                elif(sim=='input_diff'):
                    inter_sim=0.0
                else:
                    inter_sim=0.0
            else:
                x1_name= list_x[random_idxs1[i]]
                tile_num= int(x1_name.split('tile_')[-1].split('.npy')[0])
                x2_name= random.choice(train_files_bytiles[tile_num])
                y2_name= x2_name.replace('x', 'y')
                x_1=np.load(x1_name)
                y_1=np.load(list_y[random_idxs1[i]])
                x_2=np.load(x2_name)
                y_2=np.load(y2_name)
                #print(y_2.shape)
                if(sim=='DICE'):
                    tp=np.sum(y_2[:,:,:,:,1]*y_1[:,:,:,:,1])
                    fp=np.sum(y_2[:,:,:,:,1]*y_1[:,:,:,:,0])
                    fn=np.sum(y_2[:,:,:,:,0]*y_1[:,:,:,:,1])
                    inter_sim= 2*tp/(2*tp+fp+fn+1)

                if(sim=='DICE_norm'):
                    tp=np.sum(y_2[:,:,:,:,1]*y_1[:,:,:,:,1])
                    fp=np.sum(y_2[:,:,:,:,1]*y_1[:,:,:,:,0])
                    fn=np.sum(y_2[:,:,:,:,0]*y_1[:,:,:,:,1])
                    inter_sim= 2*tp/(2*tp+fp+fn+1)
                    inter_sim=(inter_sim-0.15)*4 #normalize
                elif(sim=='input_diff'):
                    inter_sim_1= np.sum(np.square(x_2[:,:,:,:,0]-x_1[:,:,:,:,0]))/(np.sum(x_2[:,:,:,:,0])+np.sum(x_1[:,:,:,:,0]))
                    inter_sim_2=np.sum(np.square(x_2[:,:,:,:,1]-x_1[:,:,:,:,1]))/(np.sum(x_2[:,:,:,:,1])+np.sum(x_1[:,:,:,:,1]))
                    inter_sim=inter_sim_1+inter_sim_2
                else:
                    #inter_sim= 2*np.sum(np.square(y_2[:,:,:,:,1]-y_1[:,:,:,:,1]))/(np.sum(y_2[:,:,:,:,1])+np.sum(y_1[:,:,:,:,1])+1)
                    inter_sim= np.sum(np.square(y_2[:,:,:,:,1]-y_1[:,:,:,:,1]))/ 350

            x_1=IQDA(x_1)
            x_2=IQDA(x_2)
            #print(x_2.shape)
            #print(x_1.shape)
            x_train_=np.concatenate((x_1,x_2,y_1,y_2),axis=4)
            x_train_=batch_rot90(x_train_)
            #print(x_train_.shape)
            x_1=x_train_[:,:,:,:,0:2]
            x_2=x_train_[:,:,:,:,2:4]
            y_1=x_train_[:,:,:,:,4:6]
            y_2=x_train_[:,:,:,:,6:8]
            #print(x_2.shape)
            x_=np.concatenate((x_1,x_2),axis=0)
            y_=np.concatenate((y_1,y_2),axis=0)
            #yield x_,[y_,inter_sim]
            yield x_,[y_,np.array([inter_sim,inter_sim])]

def update_with_new_pseudo(model,x_train,y_train,new_pseudo,listaT1,listaFLAIR,listaMASK=None):
    first=True
    for i in new_pseudo:
        print(listaT1[i])
        try:
            T1,FLAIR=load_modalities(listaT1[i],listaFLAIR[i],listaMASK[i])

        except:
            T1,FLAIR=load_modalities(listaT1[i],listaFLAIR[i])

        #x_in=one_crop(T1,FLAIR)
        """
        x_in=random_patches(T1,FLAIR,number=15)
        y_in=model.predict(x_in,batch_size=1)
        y_in=hard_labels(y_in)
        """
        #seg = seg_majvote(T1,FLAIR,model,nbNN=[5,5,5],ps=[96,96,96])
        seg = seg_majvote(T1,FLAIR,model,nbNN=[3,3,3],ps=[96,96,96])
        #x_in, y_in = random_patches_andLAB(T1,FLAIR,seg,number=5)
        x_in, y_in = notNull_patches_andLAB(T1,FLAIR,seg,number=5)

        if(first):
            x_new=x_in
            y_new=y_in
            first=False

        else:
            x_new=np.concatenate((x_new,x_in),axis=0)
            y_new=np.concatenate((y_new,y_in),axis=0)
    x_train=np.concatenate((x_train,x_new),axis=0)
    y_train=np.concatenate((y_train,y_new),axis=0)
    return x_train,y_train

def mixup(x1,x2,y1,y2,alfa=0.3):
    a=np.random.beta(alfa,alfa)
    x=a*x1+(1-a)*x2
    y=a*y1+(1-a)*y2
    return x,y

def IQDA(x_):
    op=np.random.choice(6,1,p=[0.1,0.15,0.15,0.2,0.2,0.2]) # 0:nada, 1:sharp, 2:blur, 3: axial blur 3, 4: axial blur 5, 5: axial blur 2
    if(op==1):
        for j in range(x_.shape[-1]):
            x_[0,:,:,:,j] = 2*x_[0,:,:,:,j]-ndimage.uniform_filter(x_[0,:,:,:,j], (3,3,3))
    if(op==2):
        for j in range(x_.shape[-1]):
            x_[0,:,:,:,j] = ndimage.uniform_filter(x_[0,:,:,:,j], (3,3,3))
    if(op==3):
        for j in range(x_.shape[-1]):
            #x_[:,:,:,j]=ndimage.uniform_filter(x_[:,:,:,j], (1,1,3))
            #x_[:,:,:,j]=ndimage.uniform_filter(x_[:,:,:,j], (1,2,1))
            x_[0,:,:,:,j]=ndimage.uniform_filter(x_[0,:,:,:,j], (1,1,3))
    if(op==4):
        for j in range(x_.shape[-1]):
            #x_[:,:,:,j] = ndimage.uniform_filter(x_[:,:,:,j], (1,1,5))
            #x_[:,:,:,j] = ndimage.uniform_filter(x_[:,:,:,j], (2,1,1))
            x_[0,:,:,:,j] = ndimage.uniform_filter(x_[0,:,:,:,j], (3,3,3))
    if(op==5):
        for j in range(x_.shape[-1]):
            x_[0,:,:,:,j] =ndimage.uniform_filter(x_[0,:,:,:,j], (1,1,2))
    return x_

def batch_rot90(lesion_batch):
    for i in range(lesion_batch.shape[0]):
        a=lesion_batch[i]
        op=np.random.choice(10,1)
        if(op==1):
            a=np.rot90(a,k=2,axes=(1,0))
        elif(op==9):
            a=np.rot90(a,k=1,axes=(1,0))

        elif(op==2):
            a=np.rot90(a,k=3,axes=(1,0))
        elif(op==3):
            a=np.rot90(a,k=2,axes=(2,0))
        elif(op==4):
            a=np.rot90(a,k=1,axes=(2,0))
        elif(op==5):
            a=np.rot90(a,k=3,axes=(2,0))
        elif(op==6):
            a=np.rot90(a,k=2,axes=(1,2))
        elif(op==7):
            a=np.rot90(a,k=1,axes=(1,2))
        elif(op==8):
            a=np.rot90(a,k=3,axes=(1,2))

        op=np.random.choice(4,1)
        if(op==1):
            a=a[-1::-1,:,:]
        elif(op==2):
            a=a[:,:,-1::-1]
        elif(op==3):
            a=a[:,-1::-1,:]
        lesion_batch[i]=a
    return lesion_batch

def data_gen_iqda(datafolder='data/'):
    while(1):
        list_x=sorted(glob.glob(datafolder+"x*.npy"))
        list_y=sorted(glob.glob(datafolder+"y*.npy"))
        random_idxs= np.arange(len(list_x))
        np.random.shuffle(random_idxs)
        for i in range(len(list_x)):
            x_=np.load(list_x[random_idxs[i]])
            y_=np.load(list_y[random_idxs[i]])
            x_=IQDA(x_)
            x_train_=np.concatenate((x_,y_),axis=4)
            x_train_=batch_rot90(x_train_)
            y_=x_train_[:,:,:,:,2:4]
            x_=x_train_[:,:,:,:,0:2]
            yield x_,y_

def data_gen_rot90(datafolder='data/'):
    while(1):
        list_x=sorted(glob.glob(datafolder+"x*.npy"))
        list_y=sorted(glob.glob(datafolder+"y*.npy"))
        random_idxs= np.arange(len(list_x))
        np.random.shuffle(random_idxs)
        for i in range(len(list_x)):
            x_=np.load(list_x[random_idxs[i]])
            y_=np.load(list_y[random_idxs[i]])
            x_train_=np.concatenate((x_,y_),axis=4)
            x_train_=batch_rot90(x_train_)
            y_=x_train_[:,:,:,:,2:4]
            x_=x_train_[:,:,:,:,0:2]
            yield x_,y_

def data_gen_mixup(datafolder='data/'):
    while(1):
        list_x=sorted(glob.glob(datafolder+"x*.npy"))
        list_y=sorted(glob.glob(datafolder+"y*.npy"))
        random_idxs1= np.arange(len(list_x))
        random_idxs2= np.arange(len(list_x))
        np.random.shuffle(random_idxs1)
        np.random.shuffle(random_idxs2)
        for i in range(len(list_x)):
            x1=np.load(list_x[random_idxs1[i]])
            x2=np.load(list_x[random_idxs2[i]])
            y1=np.load(list_y[random_idxs1[i]])
            y2=np.load(list_y[random_idxs2[i]])
            x_,y_=mixup(x1,x2,y1,y2)
            x_train_=np.concatenate((x_,y_),axis=4)
            x_train_=batch_rot90(x_train_)
            y_=x_train_[:,:,:,:,2:4]
            x_=x_train_[:,:,:,:,0:2]
            yield x_,y_

def update_data_folder(model,new_pseudo,listaT1,listaFLAIR,listaMASK=None,datafolder='data/',numbernotnullpatch=5,regularized=False):
    numb_data= len(sorted(glob.glob(datafolder+"x*.npy")))
    for i in range(len(new_pseudo)):
        print(listaT1[new_pseudo[i]])
        try:
            T1,FLAIR=load_modalities(listaT1[new_pseudo[i]],listaFLAIR[new_pseudo[i]],listaMASK[new_pseudo[i]])
        except:
            T1,FLAIR=load_modalities(listaT1[new_pseudo[i]],listaFLAIR[new_pseudo[i]])

        #seg = seg_majvote(T1,FLAIR,model,nbNN=[5,5,5],ps=[96,96,96])
        seg = seg_majvote(T1,FLAIR,model,nbNN=[3,3,3],ps=[96,96,96],regularized=regularized)
        #x_in, y_in = random_patches_andLAB(T1,FLAIR,seg,number=5)
        x_in, y_in , out_indx= notNull_patches_andLAB(T1,FLAIR,seg,number=numbernotnullpatch)
        for j in range(x_in.shape[0]):
            np.save(datafolder+'x_'+str(numb_data+j+i*numbernotnullpatch)+"_tile_"+str(out_indx[j])+'.npy',x_in[j:j+1])
            np.save(datafolder+'y_'+str(numb_data+j+i*numbernotnullpatch)+"_tile_"+str(out_indx[j])+'.npy',y_in[j:j+1])
    return True

def load_seg(path):
    seg_img = nii.load(path)
    seg=seg_img.get_data()
    seg=seg.astype('int')
    return seg


def keyword_toList(path,keyword):
    search=os.path.join(path,'*'+keyword+'*')
    lista=sorted(glob.glob(search))
    print("list contains: "+str( len(lista))+" elements")
    return lista


def update_labeled_folder(listaT1,listaFLAIR,listaSEG,listaMASK=None,datafolder='data/',numbernotnullpatch=15):
    numb_data= len(sorted(glob.glob(os.path.join(datafolder,"x*.npy"))))
    print(numb_data)
    for i in range(len(listaT1)):
        print(listaT1[i])
        try:
            T1,FLAIR=load_modalities(listaT1[i],listaFLAIR[i],listaMASK[i])
        except:
            T1,FLAIR=load_modalities(listaT1[i],listaFLAIR[i])
        seg=load_seg(listaSEG[i])
        x_in, y_in, out_indx = notNull_patches_andLAB(T1,FLAIR,seg,number=numbernotnullpatch)
        for j in range(x_in.shape[0]):
            np.save(os.path.join(datafolder,'x_'+str(numb_data+j+i*numbernotnullpatch)+"_tile_"+str(out_indx[j])+'.npy'),x_in[j:j+1])
            np.save(os.path.join(datafolder,'y_'+str(numb_data+j+i*numbernotnullpatch)+"_tile_"+str(out_indx[j])+'.npy'),y_in[j:j+1])
    return True

def labeld_to_data(x,y,datafolder='data/'):
    for j in range(x.shape[0]):
        np.save(datafolder+'x_'+str(j)+'.npy',x[j:j+1])
        np.save(datafolder+'y_'+str(j)+'.npy',y[j:j+1])
    return True

def one_crop(T1,FLAIR):
    T1_cropped,FLAIR_cropped=to969696(T1,FLAIR)
    x_in=np.concatenate((np.expand_dims(T1_cropped,axis=3),np.expand_dims(FLAIR_cropped,axis=3)), axis=3)
    x_in=np.expand_dims(x_in,axis=0)
    return x_in


def hard_labels(y_in):
    y_1= y_in.argmax(axis=4)
    y_0=1-y_1
    return np.concatenate((  np.expand_dims(y_0 , axis=4), np.expand_dims(y_1 , axis=4)),axis=4)

def patches(T1,FLAIR,nbNN=[3,3,3]):
    crop_bg = 4
    ps1,ps2,ps3=96,96,96

    overlap1 = np.floor((nbNN[0]*ps1 - (T1.shape[0]-2*crop_bg)) / (nbNN[0]-1))
    offset1 = ps1 - overlap1.astype('int')
    overlap2 = np.floor((nbNN[1]*ps2 - (T1.shape[1]-2*crop_bg)) / (nbNN[1]-1))
    offset2 = ps2 - overlap2.astype('int')
    overlap3 = np.floor((nbNN[2]*ps3 - (T1.shape[2]-2*crop_bg)) / (nbNN[2]-1))
    offset3 = ps3 - overlap3.astype('int')
    pT1=patch_extraction.patch_extract_3D_v2(T1,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
    pFLAIR=patch_extraction.patch_extract_3D_v2(FLAIR,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
    pT1 = np.expand_dims(pT1.astype('float32'),axis=4)
    pFLAIR =np.expand_dims(pFLAIR.astype('float32'),axis=4)
    x_in=np.concatenate((pT1,pFLAIR),axis=4)
    return x_in

def random_patches(T1,FLAIR,nbNN=[3,3,3],number=10):
    x_in= patches(T1,FLAIR,nbNN=[3,3,3])
    random_idxs= np.arange(x_in.shape[0])
    np.random.shuffle(random_idxs)
    return x_in[random_idxs[:number]]

def random_patches_andLAB(T1,FLAIR,LAB,nbNN=[3,3,3],number=10):
    x_in= patches(T1,FLAIR,nbNN=[3,3,3])
    y_in= patches(1-LAB,LAB,nbNN=[3,3,3]).astype('int')
    random_idxs= np.arange(x_in.shape[0])
    np.random.shuffle(random_idxs)
    return x_in[random_idxs[:number]], y_in[random_idxs[:number]]

def notNull_patches_andLAB(T1,FLAIR,LAB,nbNN=[3,3,3],number=10):
    x_in= patches(T1,FLAIR,nbNN=nbNN)
    y_in= patches(1-LAB,LAB,nbNN=nbNN).astype('int')
    sum_y_in=y_in[:,:,:,:,1].sum(axis=(1,2,3))
    #print(x_in.shape[0])
    random_idxs= np.arange(x_in.shape[0])
    np.random.shuffle(random_idxs)
    num=0
    x=x_in[random_idxs[0:1]]
    y=y_in[random_idxs[0:1]]
    indx_out= random_idxs[0:1]
    for i in random_idxs:
        if(num==number):
            break
        suum=sum_y_in[i]
        if suum>100:
            if(num==0):
                x=x_in[i:i+1]
                y=y_in[i:i+1]
                indx_out= np.array([random_idxs[i]])
            else:
                x=np.concatenate((x,x_in[i:i+1]),axis=0)
                y=np.concatenate((y,y_in[i:i+1]),axis=0)
                indx_out= np.concatenate((indx_out,np.array([random_idxs[i]])))
            num=num+1
    return x, y, indx_out

def seg_majvote(T1,FLAIR,model,nbNN=[5,5,5],ps=[96,96,96],regularized=False):
    MASK = (1-(T1==0).astype('int'))
    ind=np.where(MASK>0)
    indbg=np.where(MASK==0)
    crop_bg = 4
    overlap1 = np.floor((nbNN[0]*ps[0] - (T1.shape[0]-2*crop_bg)) / (nbNN[0]-1))
    offset1 = ps[0] - overlap1.astype('int')
    overlap2 = np.floor((nbNN[1]*ps[1] - (T1.shape[1]-2*crop_bg)) / (nbNN[1]-1))
    offset2 = ps[1] - overlap2.astype('int')
    overlap3 = np.floor((nbNN[2]*ps[2] - (T1.shape[2]-2*crop_bg)) / (nbNN[2]-1))
    offset3 = ps[2]- overlap3.astype('int')
    pT1=patch_extraction.patch_extract_3D_v2(T1,(ps[0],ps[1],ps[2]),nbNN,offset1,offset2,offset3,crop_bg)
    pT1= pT1.astype('float32')
    pFLAIR=patch_extraction.patch_extract_3D_v2(FLAIR,(ps[0],ps[1],ps[2]),nbNN,offset1,offset2,offset3,crop_bg)
    pFLAIR= pFLAIR.astype('float32')

    out_shape=(T1.shape[0],T1.shape[1],T1.shape[2],2)
    output=np.zeros(out_shape,T1.dtype)
    acu=np.zeros(out_shape[0:3],T1.dtype)

    ii=0 # Network ID

    for x in range(crop_bg,(nbNN[0]-1)*offset1+crop_bg+1,offset1):
        for y in range(crop_bg,(nbNN[1]-1)*offset2+crop_bg+1,offset2):
            for z in range(0,(nbNN[2]-1)*offset3+1,offset3):


                T = np.reshape(pT1[ii], (1,pT1.shape[1],pT1.shape[2],pT1.shape[3], 1))
                F = np.reshape(pFLAIR[ii], (1,pFLAIR.shape[1],pFLAIR.shape[2],pFLAIR.shape[3], 1))
                T=np.concatenate((T,F), axis=4)


                if(not ii< int(((nbNN[0]+1)/2)*nbNN[1]*nbNN[2])):

                    T=T[:,-1::-1,:,:,:]


                lista=np.array([0,1])
                if(regularized):
                    patches = model.predict(T)[0]
                else:
                    patches = model.predict(T)

                if(not ii< int(((nbNN[0]+1)/2)*nbNN[1]*nbNN[2])):
                    patches=patches[:,-1::-1,:,:,:]


                xx = x+patches.shape[1]
                if xx> output.shape[0]:
                    xx = output.shape[0]

                yy = y+patches.shape[2]
                if yy> output.shape[1]:
                    yy = output.shape[1]

                zz = z+patches.shape[3]
                if zz> output.shape[2]:
                    zz = output.shape[2]

                #store result
                local_patch = np.reshape(patches,(patches.shape[1],patches.shape[2],patches.shape[3],patches.shape[4]))

                output[x:xx,y:yy,z:zz,:]=output[x:xx,y:yy,z:zz,:]+local_patch[0:xx-x,0:yy-y,0:zz-z]
                acu[x:xx,y:yy,z:zz]=acu[x:xx,y:yy,z:zz]+1#pesos

                ii=ii+1

    ind=np.where(acu==0)
    mask_ind = np.where(acu>0)
    acu[ind]=1

    SEG= np.argmax(output, axis=3)
    SEG= np.reshape(SEG, SEG.shape[0:3])
    SEG_mask = SEG*MASK


    return SEG_mask
