from sklearn.datasets import load_digits
import os, glob, random, math, operator, umap
import numpy as np
import nibabel as nii
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
import scipy.io as sio
import modelos
import statsmodels.api as sm
from scipy.signal import argrelextrema
from collections import OrderedDict, defaultdict
from skimage import measure
from scipy.stats import pearsonr
import time,umap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,DBSCAN
import torch

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seg_metrics(seg_vol, truth_vol, output_errors=False):
    time_start = time.time()
    seg_total = np.sum(seg_vol)
    truth_total = np.sum(truth_vol)
    tp = np.sum(seg_vol[truth_vol == 1])
    dice = 2 * tp / (seg_total + truth_total)
    ppv = tp / (seg_total + 0.001)
    tpr = tp / (truth_total + 0.001)
    vd = abs(seg_total - truth_total) / truth_total

    # calculate LFPR
    seg_labels, seg_num = measure.label(seg_vol, return_num=True, connectivity=2)
    lfp_cnt = 0
    tmp_cnt = 0
    for label in range(1, seg_num + 1):
        tmp_cnt += np.sum(seg_vol[seg_labels == label])
        if np.sum(truth_vol[seg_labels == label]) == 0:
            lfp_cnt += 1
    lfpr = lfp_cnt / (seg_num + 0.001)

    # calculate LTPR
    truth_labels, truth_num = measure.label(truth_vol, return_num=True, connectivity=2)
    ltp_cnt = 0
    for label in range(1, truth_num + 1):
        if np.sum(seg_vol[truth_labels == label]) > 0:
            ltp_cnt += 1
    ltpr = ltp_cnt / truth_num

    # calculate Pearson's correlation coefficient
    corr = pearsonr(seg_vol.flatten(), truth_vol.flatten())[0]
    # print("Timed used calculating metrics: ", time.time() - time_start)

    return OrderedDict([('dice', dice), ('ppv', ppv), ('tpr', tpr), ('lfpr', lfpr),
                        ('ltpr', ltpr), ('vd', vd), ('corr', corr)])

def mdice(y_pred, y_true):
        acu=0
        size=y_true.get_shape().as_list()
        epsilon=0.00000000001
        for i in range(0,size[4]):
            a=y_true[:,:,:,:,i]
            b=y_pred[:,:,:,:,i]
            y_int = a[:]*b[:]
            acu=acu+(2*K.sum(y_int[:]) / (K.sum(a[:]) + K.sum(b[:]) + epsilon) )
        acu=acu/(size[4])
        return acu

def get_list_from_ropes(ropes):
    out=[]
    for rope in ropes[::-1]:
        out=out+rope[::-1]
    return out

def get_ropes(neibhors_and_dis_):
    neibhors_and_dis=np.copy(neibhors_and_dis_)
    all_classes=list(range(neibhors_and_dis.shape[0]))
    ropes=[]
    while(len(all_classes)>0):
        still_on_a_rope=True
        fist_element,not_neighbor=np.where( neibhors_and_dis[:,2:] == neibhors_and_dis[:,2:].max() )
        fist_element,not_neighbor=fist_element[0],not_neighbor[0]
        #print(fist_element)
        #print(not_neighbor)
        neibhors_and_dis[fist_element,2:]=-1000000
        all_classes.remove(fist_element)
        #print(all_classes)
        rope=Rope(fist_element)

        if(not_neighbor==0):
            next_element=neibhors_and_dis[fist_element,1]
        else:
            next_element=neibhors_and_dis[fist_element,0]
        next_element=int(next_element)
        rope.add_elements_end(next_element)
        all_classes.remove(next_element)
        neibhors_and_dis[fist_element,2:]=-1000000
        while(still_on_a_rope):
            #print(next_element)
            neighbor1,neighbor2= neibhors_and_dis[next_element,0], neibhors_and_dis[next_element,1]
            neighbor1,neighbor2= int(neighbor1),int(neighbor2)
            #print(neighbor1)
            #print(neighbor2)

            if( (neighbor1 in all_classes) and not (neighbor2 in all_classes ) ):
                next_element=neighbor1
            elif( (neighbor2 in all_classes) and not (neighbor1 in all_classes ) ):
                next_element=neighbor2
            elif( (neighbor1 in all_classes) and (neighbor2 in all_classes ) ):
                """
                if(neibhors_and_dis[next_element,2]>neibhors_and_dis[next_element,3]):
                    next_element=neighbor2
                else:
                    next_element=neighbor1
                """
                print('error1')
                return 1
            elif( not (neighbor1 in all_classes) and not (neighbor2 in all_classes ) ):
                print('generate next rope')
                still_on_a_rope=False
                continue
            else:
                print('error2')
                return 1
            next_element=int(next_element)
            rope.add_elements_end(next_element)
            all_classes.remove(next_element)
            neibhors_and_dis[fist_element,2:]=-1000000
        ropes.append(rope.rope_elements)
        print(ropes[-1])
    return ropes

class Rope:
    def __init__(self,first_element):
        self.rope_elements=[first_element]
    def get_last_element(self):
        return self.rope_elements[-1]
    def get_first_element(self):
        return self.rope_elements[0]
    def get_rope(self):
        return self.rope_elements
    def add_elements_start(self,elements):
        if(not isinstance(new_elements, list)):
            new_elements=[new_elements]
        self.rope_elements=new_elements+self.rope_elements
    def add_elements_end(self,new_elements):
        if(not isinstance(new_elements, list)):
            new_elements=[new_elements]
        self.rope_elements=self.rope_elements+new_elements

def get_neibhors_and_dis(unlabeled_tsne,labels):
    neibhors_and_dis=np.zeros((labels.max(),4))
    for i in range(labels.max()):
        print(i)
        tsnee=np.copy(unlabeled_tsne)
        tsnee[labels==i]=10000
        dis=brute_force_rank(tsnee,unlabeled_tsne[labels==i])
        label1=labels[np.where(dis==dis.min())[0][0]]
        neibhors_and_dis[i,0]=label1
        neibhors_and_dis[i,2]=dis.min()

        tsnee[labels==label1]=10000
        dis=brute_force_rank(tsnee,unlabeled_tsne[labels==i])
        label2=labels[np.where(dis==dis.min())[0][0]]
        neibhors_and_dis[i,1]=label2
        neibhors_and_dis[i,3]=dis.min()
    return neibhors_and_dis

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

def features_from_names_flair(listaFLAIR,fun,listaMASK=None,ps=[64,64,64]):
    file_names=[]

    for i in range(len(listaFLAIR)):
        #print(listaT1[i])

        try:
            FLAIR=load_flair(listaFLAIR[i],listaMASK[i])

        except:
            FLAIR=load_flair(listaFLAIR[i])
        FLAIR_cropped=crop_center(FLAIR,ps[0],ps[1],ps[2])
        x_in=np.expand_dims(FLAIR_cropped,axis=3)
        x_in=np.expand_dims(x_in,axis=0)
        #bottleneck_features[listaT1_isbi[i]]=bottleneck_out[0].mean(axis=(0,1,2,3))
        bottleneck_out = fun([x_in, 1.])
        bottleneck_out_mean=bottleneck_out[0].mean(axis=(0,1,2,3))
        if(i==0):
            bottleneck_features=np.expand_dims(bottleneck_out_mean,axis=0)
        else:
            #return np.repeat(bottleneck_features,len(listaT1),axis=0),file_names
            bottleneck_features=np.concatenate((bottleneck_features,np.expand_dims(bottleneck_out_mean,axis=0)),axis=0)
        file_names.append(listaFLAIR[i])
    return bottleneck_features,file_names

def features_from_names_pytorch(listaT1=None, listaFLAIR=None, idices=None, model=None,listaMASK=None,ps=[64,64,64]):
    model.eval()
    if(not idices is None):
        if(not listaT1 is None):
            listaT1=listaT1[idices]
        listaFLAIR=listaFLAIR[idices]
    
    file_names=[]

    for i in range(len(listaFLAIR)):
        if(listaT1 is None):
            try:
                FLAIR=load_flair(listaFLAIR[i],listaMASK[i])

            except:
                FLAIR=load_flair(listaFLAIR[i])

            FLAIR_cropped=crop_center(FLAIR,ps[0],ps[1],ps[2])
            x_in=np.expand_dims(FLAIR_cropped,axis=0)
        else:
            try:
                T1,FLAIR=load_modalities(listaT1[i],listaFLAIR[i],listaMASK[i])

            except:
                T1,FLAIR=load_modalities(listaT1[i],listaFLAIR[i])
            x_in=np.concatenate((np.expand_dims(crop_center(T1,ps[0],ps[1],ps[2]),axis=0),np.expand_dims(crop_center(FLAIR,ps[0],ps[1],ps[2]),axis=0)), axis=0)

        x_in=np.expand_dims(x_in,axis=0)
        x_in= torch.from_numpy(x_in).to(device).float()
        #bottleneck_features[listaT1_isbi[i]]=bottleneck_out[0].mean(axis=(0,1,2,3))
        with torch.no_grad():
            bottleneck_out,_,_,_ = model.encoder(x_in)

        bottleneck_out_mean=bottleneck_out.cpu().numpy().mean()
        
        if(i==0):
            bottleneck_features=np.expand_dims(bottleneck_out_mean,axis=0)
        else:
            bottleneck_features=np.concatenate((bottleneck_features,np.expand_dims(bottleneck_out_mean,axis=0)),axis=0)
        file_names.append(listaFLAIR[i])
    bottleneck_features= np.expand_dims(bottleneck_features,axis=1)
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

def load_flair(FLAIR_name,MASK_name=None):
    FLAIR_img = nii.load(FLAIR_name)
    FLAIR=FLAIR_img.get_data()
    FLAIR=FLAIR.astype('float32')
    if(not MASK_name==None):
        MASK_img = nii.load(MASK_name)
        MASK = MASK_img.get_data()
        MASK=MASK.astype('int')

        FLAIR=FLAIR*MASK

    peak = normalize_image(FLAIR, 'flair')
    FLAIR=FLAIR/peak
    return FLAIR

def crop_center(img,cropx,cropy,cropz):
    x,y,z = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    startz = z//2-(cropz//2)
    return img[startx:startx+cropx,starty:starty+cropy,startz:startz+cropz]

def crop_center2D(img,cropx,cropy):
    batch,x,y = img.shape
    starty = y//2-(cropy//2)
    startx = x//2-(cropx//2)
    return img[:,startx:startx+cropx,starty:starty+cropy]

def to969696(T1,FLAIR):
    T1=crop_center(T1,64,64,64)
    FLAIR=crop_center(FLAIR,64,64,64)
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

def get_clusters_with_labels(classes_u, ropes_list):
    bundles=[]
    clusters=[]
    for i in ropes_list:
        indx_of_cluster_element=np.where(classes_u==i)[0]
        print(indx_of_cluster_element)
        bundles.append(indx_of_cluster_element)
        clusters.append(i)
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

def get_clusters_dbscan(unlabeled_tsne,labeled_tsne):
    u = DBSCAN(eps=0.8).fit(unlabeled_tsne)
    l = DBSCAN(eps=0.8).fit(labeled_tsne)
    classes_u= u.labels_
    classes_l= l.labels_
    return classes_u,classes_l

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
    argsort=np.argsort(ranks, axis=1)[:,:k]
    Kclosest=np.zeros(ranks.shape)
    for i in range(ranks.shape[0]):
        Kclosest[i,argsort[i,:]]=1
    values_for_distance= Kclosest*ranks
    distance= np.sum(values_for_distance,axis=1)
    indxs=np.argsort(distance)[:n_indxs]
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
    ps=[64,64,64]
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

def data_gen_iqda_2it(datafolder,train_files_bytiles,same_ratio=0.8,sim='DICE'):
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
                #print(x_2.shape)
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
                #print(x2_name)
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

            #print(x_1.shape)
            #print(x_2.shape)
            x_1=IQDA(x_1)
            x_2=IQDA(x_2)
            x_train_=np.concatenate((x_1,x_2,y_1,y_2),axis=4)
            x_train_=batch_rot90(x_train_)
            size_in_x1=x_1.shape[4]
            size_in_x2=size_in_x1+x_2.shape[4]
            size_in_y1=size_in_x2+y_1.shape[4]
            size_in_y2=size_in_y1+y_2.shape[4]
            x_1=x_train_[:,:,:,:,0:size_in_x1]
            x_2=x_train_[:,:,:,:,size_in_x1:size_in_x2]
            y_1=x_train_[:,:,:,:,size_in_x2:size_in_y1]
            y_2=x_train_[:,:,:,:,size_in_y1:size_in_y2]
            #print(x_2.shape)
            x_=np.concatenate((x_1,x_2),axis=0)
            y_=np.concatenate((y_1,y_2),axis=0)
            #yield x_,[y_,inter_sim]
            yield x_,[y_,np.array([inter_sim,inter_sim])]

def data_gen_dual_task_reconstruct_seg(datafolder,datafolder_ssl):
    while(1):
        list_x=sorted(glob.glob(datafolder+"x*.npy"))
        list_x_ssl=sorted(glob.glob(datafolder_ssl+"x*.npy"))
        list_y=sorted(glob.glob(datafolder+"y*.npy"))
        random_idxs1= np.arange(len(list_x))
        random_idxs2= np.arange(len(list_x_ssl))
        np.random.shuffle(random_idxs1)
        np.random.shuffle(random_idxs2)
        for i in range(len(list_x)):
            x_1=np.load(list_x[random_idxs1[i]])
            y_1=np.load(list_y[random_idxs1[i]])
            x_2=np.load(list_x_ssl[random_idxs2[i]])
            x_1=IQDA(x_1)
            x_2=IQDA(x_2)
            x_1,y_1=random_rot(x_1,y_1)
            x_2=batch_rot90(x_2)
            y_2=np.concatenate((x_2,x_2),axis=4)
            x_=np.concatenate((x_1,x_2),axis=0)
            y_=np.concatenate((y_1,y_2),axis=0)
            #print(y_.shape)
            #print(x_.shape)
            yield x_,y_

def data_gen_consistency_reg(datafolder,datafolder_ssl):
    while(1):
        list_x=sorted(glob.glob(datafolder+"x*.npy"))
        list_x_ssl=sorted(glob.glob(datafolder_ssl+"x*.npy"))
        list_y=sorted(glob.glob(datafolder+"y*.npy"))
        random_idxs1= np.arange(len(list_x))
        random_idxs2= np.arange(len(list_x_ssl))
        np.random.shuffle(random_idxs1)
        np.random.shuffle(random_idxs2)
        for i in range(len(list_x)):
            x_1=np.load(list_x[random_idxs1[i]])
            y_1=np.load(list_y[random_idxs1[i]])
            x_2=np.load(list_x_ssl[random_idxs2[i]])
            x_3=np.copy(x_2)
            x_1=IQDA(x_1)
            x_2=IQDA(x_2)
            x_3=IQDA(x_3)
            x_1,y_1=random_rot(x_1,y_1)
            x_3,x_2=random_rot(x_3,x_2)
            y_2=np.copy(y_1)
            y_3=np.copy(y_1)
            #"""
            #op1=np.random.choice(4,1)
            op2=np.random.choice(4,1)
            op1=0
            #op3=np.random.choice(4,1)
            op4=np.random.choice(4,1)
            op3=0
            x_2=batch_rot90_with_choice(x_2,op1,op2)
            x_3=batch_rot90_with_choice(x_3,op3,op4)
            y_2[...,0]=op1
            y_2[...,1]=op2
            y_3[...,0]=op3
            y_3[...,1]=op4
            #"""
            x_=np.concatenate((x_1,x_2,x_3),axis=0)
            y_=np.concatenate((y_1,y_2,y_3),axis=0)
            #print(y_[1,0,0,0,0])
            #print(y_[1,0,0,0,1])
            #print(y_[2,0,0,0,0])
            #print(y_[2,0,0,0,1])
            yield x_,y_

def data_gen_uncertainty_pseudolab(datafolder,datafolder_ssl):
    while(1):
        list_x=sorted(glob.glob(datafolder+"x*.npy"))
        list_x_ssl=sorted(glob.glob(datafolder_ssl+"x*.npy"))
        list_y_ssl=sorted(glob.glob(datafolder_ssl+"soft*.npy"))
        list_entropy_ssl=sorted(glob.glob(datafolder_ssl+"entropy*.npy"))
        list_y=sorted(glob.glob(datafolder+"y*.npy"))
        random_idxs1= np.arange(len(list_x))
        random_idxs2= np.arange(len(list_x_ssl))
        np.random.shuffle(random_idxs1)
        np.random.shuffle(random_idxs2)
        for i in range(len(list_x)):
            x_1=np.load(list_x[random_idxs1[i]])
            y_1=np.load(list_y[random_idxs1[i]])
            x_2=np.load(list_x_ssl[random_idxs2[i]])
            y_2=np.load(list_y_ssl[random_idxs2[i]])
            entropy_2=np.load(list_entropy_ssl[random_idxs2[i]])
            x_1=IQDA(x_1)
            x_2=IQDA(x_2)
            x_1,y_1=random_rot(x_1,y_1)
            #entropy_2= np.exp(-1.5*entropy_2)
            entropy_2= np.exp(-3.5*entropy_2)
            #print(entropy_2.max())
            #print(entropy_2.min())
            #y_2=np.concatenate((y_2,entropy_2),axis=4)
            y_2[...,0:1]=entropy_2
            x_2,y_2=random_rot(x_2,y_2)
            x_=np.concatenate((x_1,x_2),axis=0)
            y_=np.concatenate((y_1,y_2),axis=0)
            #print(y_.shape)
            #print(x_.shape)
            yield x_,y_

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
        #seg = seg_majvote(T1,FLAIR,model,nbNN=[5,5,5],ps=[64,64,64])
        seg = seg_majvote(T1,FLAIR,model,nbNN=[5,5,5],ps=[64,64,64])
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
    for i in range(0,x_.shape[0]):
        op=np.random.choice(5,1,p=[0.15,0.25,0.2,0.2,0.2]) # 0:nada, 1:sharp, 2:blur, 3: axial blur 3, 4: axial blur 5, 5: axial blur 2
        for j in range(x_.shape[-1]):
            if(op==1):
                x_[i,:,:,:,j] = 2*x_[i,:,:,:,j]-ndimage.uniform_filter(x_[i,:,:,:,j], (3,3,3))
            if(op==2):
                x_[i,:,:,:,j] = ndimage.uniform_filter(x_[i,:,:,:,j], (3,3,3))
            if(op==3):
                x_[i,:,:,:,j]=ndimage.uniform_filter(x_[i,:,:,:,j], (1,1,3))
            if(op==4):
                x_[i,:,:,:,j] =ndimage.uniform_filter(x_[i,:,:,:,j], (1,1,2))
        return x_

def random_rot(x,y):
    op=np.random.choice(10,1)
    if(op==1):
        x=np.rot90(x,k=2,axes=(1,3))
        y=np.rot90(y,k=2,axes=(1,3))
    elif(op==9):
        x=np.rot90(x,k=1,axes=(1,3))
        y=np.rot90(y,k=1,axes=(1,3))
    elif(op==2):
        x=np.rot90(x,k=3,axes=(1,3))
        y=np.rot90(y,k=3,axes=(1,3))
    elif(op==3):
        x=np.rot90(x,k=2,axes=(2,3))
        y=np.rot90(y,k=2,axes=(2,3))
    elif(op==4):
        x=np.rot90(x,k=1,axes=(2,3))
        y=np.rot90(y,k=1,axes=(2,3))
    elif(op==5):
        x=np.rot90(x,k=3,axes=(2,3))
        y=np.rot90(y,k=3,axes=(2,3))
    elif(op==6):
        x=np.rot90(x,k=2,axes=(1,2))
        y=np.rot90(y,k=2,axes=(1,2))
    elif(op==7):
        x=np.rot90(x,k=1,axes=(1,2))
        y=np.rot90(y,k=1,axes=(1,2))
    elif(op==8):
        x=np.rot90(x,k=3,axes=(1,2))
        y=np.rot90(y,k=3,axes=(1,2))

    op=np.random.choice(4,1)
    if(op==1):
        x=x[:,-1::-1,:,:]
        y=y[:,-1::-1,:,:]
    elif(op==2):
        x=x[:,:,:,-1::-1]
        y=y[:,:,:,-1::-1]
    elif(op==3):
        x=x[:,:,-1::-1,:]
        y=y[:,:,-1::-1,:]
    return x,y

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

def batch_rot90_with_choice(lesion_batch, op, op2):
    for i in range(lesion_batch.shape[0]):
        a=lesion_batch[i]
        if(op==1):
            a=np.rot90(a,k=2,axes=(0,1))
        elif(op==3):
            a=np.rot90(a,k=1,axes=(0,1))
        elif(op==2):
            a=np.rot90(a,k=3,axes=(0,1))
        if(op2==1):
            a=a[-1::-1,:,:]
        elif(op2==2):
            a=a[:,:,-1::-1]
        elif(op2==3):
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
            size_in_x1=x_.shape[4]
            size_in_y1=size_in_x1+y_.shape[4]
            x_=x_train_[:,:,:,:,0:size_in_x1]
            y_=x_train_[:,:,:,:,size_in_x1:size_in_y1]
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

        #seg = seg_majvote(T1,FLAIR,model,nbNN=[5,5,5],ps=[64,64,64])
        seg = seg_majvote(T1,FLAIR,model,nbNN=[5,5,5],ps=[64,64,64],regularized=regularized)
        #x_in, y_in = random_patches_andLAB(T1,FLAIR,seg,number=5)
        x_in, y_in , out_indx= notNull_patches_andLAB(T1,FLAIR,seg,number=numbernotnullpatch)
        for j in range(x_in.shape[0]):
            np.save(datafolder+'x_'+str(numb_data+j+i*numbernotnullpatch)+"_tile_"+str(out_indx[j])+'.npy',x_in[j:j+1])
            np.save(datafolder+'y_'+str(numb_data+j+i*numbernotnullpatch)+"_tile_"+str(out_indx[j])+'.npy',y_in[j:j+1])
    return True

def update_data_folder_flair(model,new_pseudo,listaFLAIR,listaMASK=None,datafolder='data/',numbernotnullpatch=2,regularized=False,nbNN=[5,5,5],ps=[64,64,64]):
    numb_data= len(sorted(glob.glob(datafolder+"x*.npy")))
    for i in range(len(new_pseudo)):
        print(listaFLAIR[new_pseudo[i]])
        try:
            FLAIR=load_flair(listaFLAIR[new_pseudo[i]],listaMASK[new_pseudo[i]])
        except:
            FLAIR=load_flair(listaFLAIR[new_pseudo[i]])

        seg = seg_majvote_flair(FLAIR,model,nbNN=[5,5,5],ps=ps,regularized=regularized)
        #x_in, y_in = random_patches_andLAB(T1,FLAIR,seg,number=5)
        x_in, y_in , out_indx= notNull_flair_andLAB(FLAIR,seg,nbNN=nbNN,ps=ps,number=numbernotnullpatch)
        for j in range(x_in.shape[0]):
            np.save(datafolder+'x_'+str(numb_data+j+i*numbernotnullpatch)+"_tile_"+str(out_indx[j])+'.npy',x_in[j:j+1])
            np.save(datafolder+'y_'+str(numb_data+j+i*numbernotnullpatch)+"_tile_"+str(out_indx[j])+'.npy',y_in[j:j+1])
    return True

def update_data_folder_pytorch(model,new_pseudo,listaT1, listaFLAIR,listaMASK=None,datafolder='data/',numbernotnullpatch=2,nbNN=[5,5,5],ps=[64,64,64]):
    numb_data= len(sorted(glob.glob(datafolder+"x*.npy")))
    for i in range(len(new_pseudo)):
        print(listaFLAIR[new_pseudo[i]])

        if(listaT1 is None):

            try:
                FLAIR=load_flair(listaFLAIR[new_pseudo[i]],listaMASK[new_pseudo[i]])
            except:
                FLAIR=load_flair(listaFLAIR[new_pseudo[i]])

        else:
            try:
                T1,FLAIR=load_modalities(listaT1[i],listaFLAIR[i],listaMASK[i])

            except:
                T1,FLAIR=load_modalities(listaT1[i],listaFLAIR[i])

        if(listaT1 is None):
            seg = seg_majvote_pytorch(None, FLAIR,model,ps=ps)
            x_in, y_in , out_indx= notNull_flair_andLAB(FLAIR,seg,nbNN=nbNN,ps=ps,number=numbernotnullpatch)
        else:
            seg = seg_majvote_pytorch(T1, FLAIR,model,ps=ps)
            x_in, y_in , out_indx= notNull_patches_andLAB(T1,FLAIR,seg,nbNN=nbNN,ps=ps,number=numbernotnullpatch)  

        for j in range(x_in.shape[0]):
            np.save(datafolder+'x_'+str(numb_data+j+i*numbernotnullpatch)+"_tile_"+str(out_indx[j])+'.npy',x_in[j:j+1])
            np.save(datafolder+'y_'+str(numb_data+j+i*numbernotnullpatch)+"_tile_"+str(out_indx[j])+'.npy',y_in[j:j+1])
    return True

def load_seg(path):
    seg_img = nii.load(path)
    seg=seg_img.get_data()
    seg=seg.astype('int')
    return seg

def seg_majvote_pytorch(T1,FLAIR,model,ps=[64,64,64],offset1=32,offset2=32,offset3=32,crop_bg=5):
    MASK = (1-(FLAIR==0).astype('int'))
    ind=np.where(MASK>0)
    indbg=np.where(MASK==0)

    out_shape=(FLAIR.shape[0],FLAIR.shape[1],FLAIR.shape[2],2)
    output=np.zeros(out_shape,FLAIR.dtype)
    acu=np.zeros(out_shape[0:3],FLAIR.dtype)

    ii=0 # Network ID

    for x in range(crop_bg,  FLAIR.shape[0]-crop_bg ,offset1):
        xx = x+ps[0]
        if xx> output.shape[0]:
            xx = output.shape[0]
            x=xx-ps[0]
        for y in range(crop_bg,  FLAIR.shape[1]-crop_bg ,offset2):
            yy = y+ps[1]
            if yy> output.shape[1]:
                yy = output.shape[1]
                y=yy-ps[1]
            for z in range(crop_bg,  FLAIR.shape[2]-crop_bg ,offset3):
                zz = z+ps[2]
                if zz> output.shape[2]:
                    zz = output.shape[2]
                    z=zz-ps[2]
                if(T1 is None):
                    T = np.reshape(   FLAIR[x:xx,y:yy,z:zz] , (1,ps[0],ps[1],ps[2], 1))
                else:
                    T = np.reshape(   T1[x:xx,y:yy,z:zz] , (1,ps[0],ps[1],ps[2], 1))
                    F = np.reshape(    FLAIR[x:xx,y:yy,z:zz] , (1,ps[0],ps[1],ps[2], 1))
                    T=np.concatenate((T,F), axis=4)
                T=torch.from_numpy(T.transpose((0,4,1,2,3))).to(device).float()
                with torch.no_grad():
                    patches = model(T)
                    patches= patches.cpu().numpy()
                    patches= patches.transpose((0,2,3,4,1))
                #store result
                local_patch = np.reshape(patches,(patches.shape[1],patches.shape[2],patches.shape[3],patches.shape[4]))
                output[x:xx,y:yy,z:zz,:]=output[x:xx,y:yy,z:zz,:]+local_patch[0:xx-x,0:yy-y,0:zz-z]
                ii=ii+1

    SEG= np.argmax(output, axis=3)
    SEG_mask= np.reshape(SEG, SEG.shape[0:3])
    return SEG_mask

def keyword_toList(path,keyword):
    search=os.path.join(path,'*'+keyword+'*')
    lista=sorted(glob.glob(search))
    print("list contains: "+str( len(lista))+" elements")
    return lista


def update_labeled_folder(listaT1,listaFLAIR,listaSEG,listaMASK=None,datafolder='data/',numbernotnullpatch=15,nbNN=[5,5,5],ps=[64,64,64]):
    numb_data= len(sorted(glob.glob(os.path.join(datafolder,"x*.npy"))))
    print(numb_data)
    for i in range(len(listaT1)):
        print(listaT1[i])
        try:
            T1,FLAIR=load_modalities(listaT1[i],listaFLAIR[i],listaMASK[i])
        except:
            T1,FLAIR=load_modalities(listaT1[i],listaFLAIR[i])
        seg=load_seg(listaSEG[i])
        x_in, y_in, out_indx = notNull_patches_andLAB(T1,FLAIR,seg,number=numbernotnullpatch,nbNN=nbNN,ps=ps)
        for j in range(x_in.shape[0]):
            np.save(os.path.join(datafolder,'x_'+str(numb_data+j+i*numbernotnullpatch)+"_tile_"+str(out_indx[j])+'.npy'),x_in[j:j+1])
            np.save(os.path.join(datafolder,'y_'+str(numb_data+j+i*numbernotnullpatch)+"_tile_"+str(out_indx[j])+'.npy'),y_in[j:j+1])
    return True

def update_labeled_folder_flair(listaFLAIR,listaSEG,listaMASK=None,datafolder='data/',numbernotnullpatch=15,nbNN=[5,5,5],ps=[64,64,64]):
    numb_data= len(sorted(glob.glob(os.path.join(datafolder,"x*.npy"))))
    print(numb_data)
    for i in range(len(listaFLAIR)):
        print(listaFLAIR[i])
        try:
            FLAIR=load_flair(listaFLAIR[i],listaMASK[i])
        except:
            FLAIR=load_flair(listaFLAIR[i])
        seg=load_seg(listaSEG[i])

        x_in, y_in , out_indx = notNull_flair_andLAB(FLAIR,seg,nbNN=nbNN,ps=ps,number=numbernotnullpatch)
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

def patches(T1,FLAIR,nbNN=[5,5,5],ps=[64,64,64]):
    crop_bg = 4
    [ps1,ps2,ps3]=ps

    overlap1 = np.floor((nbNN[0]*ps1 - (T1.shape[0]-2*crop_bg)) / (nbNN[0]-1))
    offset1 = ps1 - overlap1.astype('int')
    overlap2 = np.floor((nbNN[1]*ps2 - (T1.shape[1]-2*crop_bg)) / (nbNN[1]-1))
    offset2 = ps2 - overlap2.astype('int')
    overlap3 = np.floor((nbNN[2]*ps3 - (T1.shape[2]-2*crop_bg)) / (nbNN[2]-1))
    offset3 = ps3 - overlap3.astype('int')
    pT1=patch_extract_3D_v2(T1,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
    pFLAIR=patch_extract_3D_v2(FLAIR,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
    pT1 = np.expand_dims(pT1.astype('float32'),axis=4)
    pFLAIR =np.expand_dims(pFLAIR.astype('float32'),axis=4)
    x_in=np.concatenate((pT1,pFLAIR),axis=4)
    return x_in

def patches_flair(FLAIR,nbNN=[5,5,5],ps=[64,64,64]):
    crop_bg = 4
    [ps1,ps2,ps3]=ps

    overlap1 = np.floor((nbNN[0]*ps1 - (FLAIR.shape[0]-2*crop_bg)) / (nbNN[0]-1))
    offset1 = ps1 - overlap1.astype('int')
    overlap2 = np.floor((nbNN[1]*ps2 - (FLAIR.shape[1]-2*crop_bg)) / (nbNN[1]-1))
    offset2 = ps2 - overlap2.astype('int')
    overlap3 = np.floor((nbNN[2]*ps3 - (FLAIR.shape[2]-2*crop_bg)) / (nbNN[2]-1))
    offset3 = ps3 - overlap3.astype('int')
    pFLAIR=patch_extract_3D_v2(FLAIR,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
    pFLAIR =np.expand_dims(pFLAIR.astype('float32'),axis=4)
    return pFLAIR

def patch_extract_3D_v2(input,patch_shape,nbNN,offx=1,offy=1,offz=1,crop_bg=0):
    n=0
    numPatches=nbNN[0]*nbNN[1]*nbNN[2]
    local_patch = np.zeros( patch_shape,input.dtype)
    patches_3D=np.zeros([numPatches]+list(patch_shape),input.dtype)
    for x in range(crop_bg,(nbNN[0]-1)*offx+crop_bg+1,offx):
        for y in range(crop_bg,(nbNN[1]-1)*offy+crop_bg+1,offy):
            for z in range(0,(nbNN[2]-1)*offz+1,offz): #spine is touching one side in Z dicrection, so crop has to be asymetric
                xx = x+patch_shape[0]
                if xx> input.shape[0]:
                    xx = input.shape[0]
                yy = y+patch_shape[1]
                if yy> input.shape[1]:
                    yy = input.shape[1]
                zz = z+patch_shape[2]
                if zz> input.shape[2]:
                    zz = input.shape[2]
                # To deal with defferent patch size due to border issue
                local_patch = local_patch*0
                local_patch[0:xx-x,0:yy-y,0:zz-z] = input[x:xx,y:yy,z:zz]
                a=local_patch[np.newaxis,...]
                patches_3D[n,:,:,:]=a
                n=n+1
    patches_3D=patches_3D[0:n,...]
    return patches_3D

def patch_reconstruct_3D_v2(out_shape,patches,nbNN, offx=1,offy=1,offz=1,crop_bg=0):
    n=0
    output=np.zeros(out_shape,patches.dtype)
    acu=np.zeros(out_shape,patches.dtype)
    pesos=np.ones((patches.shape[1],patches.shape[2],patches.shape[3]))
    for x in range(crop_background_border,(nbNN-1)*offx+crop_background_border+1,offx):
        for y in range(crop_background_border,(nbNN-1)*offy+crop_background_border+1,offy):
            for z in range(crop_background_border,(nbNN-1)*offz+crop_background_border+1,offz):

                xx = x+patches.shape[1]
                if xx> input.shape[0]:
                    xx = input.shape[0]
                yy = y+patches.shape[2]
                if yy> input.shape[1]:
                    yy = input.shape[1]
                zz = z+patches.shape[3]
                if zz> input.shape[2]:
                    zz = input.shape[2]

                output[x:xx,y:yy,z:zz]=output[x:xx,y:yy,z:zz]+np.reshape(patches[n,:,:,:],(patches.shape[1],patches.shape[2],patches.shape[3]))
                acu[x:xx,y:yy,z:zz]=acu[x:xx,y:yy,z:zz]+pesos
                n=n+1
    ind=np.where(acu==0)
    acu[ind]=1
    output=output/acu
    return output

def random_patches(T1,FLAIR,nbNN=[5,5,5],number=10):
    x_in= patches(T1,FLAIR,nbNN=[5,5,5])
    random_idxs= np.arange(x_in.shape[0])
    np.random.shuffle(random_idxs)
    return x_in[random_idxs[:number]]

def random_patches_andLAB(T1,FLAIR,LAB,nbNN=[5,5,5],number=10):
    x_in= patches(T1,FLAIR,nbNN=[5,5,5])
    y_in= patches(1-LAB,LAB,nbNN=[5,5,5]).astype('int')
    random_idxs= np.arange(x_in.shape[0])
    np.random.shuffle(random_idxs)
    return x_in[random_idxs[:number]], y_in[random_idxs[:number]]

def notNull_patches_andLAB(T1,FLAIR,LAB,ps=[64,64,64],nbNN=[5,5,5],number=10):
    x_in= patches(T1,FLAIR,nbNN=nbNN)
    y_in= patches(1-LAB,LAB,nbNN=nbNN).astype('int')
    sum_y_in=y_in[:,:,:,:,1].sum(axis=(1,2,3))
    #print(x_in.shape[0])
    #random_idxs= np.arange(x_in.shape[0])

    ###remove border tiles###
    a=np.zeros((nbNN[0],nbNN[1],nbNN[2]))
    n=0
    for x in range(nbNN[0]):
        for y in range(nbNN[1]):
            for z in range(nbNN[2]):
                a[x,y,z]= n
                n=n+1

    random_idxs= []
    for stepx in range(1,nbNN[0]-1):
        for stepy in range(1,nbNN[1]-1):
            for stepz in range(1,nbNN[2]-1):
                tile_num= int(a[stepx,stepy,stepz])
                random_idxs.append(tile_num)
    
    random_idxs= np.array(random_idxs)
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
                indx_out= np.array([i])
            else:
                x=np.concatenate((x,x_in[i:i+1]),axis=0)
                y=np.concatenate((y,y_in[i:i+1]),axis=0)
                indx_out= np.concatenate((indx_out,np.array([i])))
            num=num+1
    return x, y, indx_out

def notNull_flair_andLAB(FLAIR,LAB,nbNN=[5,5,5],ps=[64,64,64],number=10):
    x_in= patches_flair(FLAIR,nbNN=nbNN,ps=ps)
    y_in= patches(1-LAB,LAB,nbNN=nbNN,ps=ps).astype('int')
    sum_y_in=y_in[:,:,:,:,1].sum(axis=(1,2,3))
    #print(x_in.shape[0])
    #random_idxs= np.arange(x_in.shape[0])
    a=np.zeros((nbNN[0],nbNN[1],nbNN[2]))
    n=0
    for x in range(nbNN[0]):
        for y in range(nbNN[1]):
            for z in range(nbNN[2]):
                a[x,y,z]= n
                n=n+1

    random_idxs= []
    for stepx in range(1,nbNN[0]-1):
        for stepy in range(1,nbNN[1]-1):
            for stepz in range(1,nbNN[2]-1):
                tile_num= int(a[stepx,stepy,stepz])
                random_idxs.append(tile_num)
    
    random_idxs= np.array(random_idxs)
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
                indx_out= np.array([i])
            else:
                x=np.concatenate((x,x_in[i:i+1]),axis=0)
                y=np.concatenate((y,y_in[i:i+1]),axis=0)
                indx_out= np.concatenate((indx_out,np.array([i])))
            num=num+1
    return x, y, indx_out

def seg_majvote(T1,FLAIR,model,nbNN=[5,5,5],ps=[64,64,64],multi_out=False,multi_in=True):
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
    pT1=patch_extract_3D_v2(T1,(ps[0],ps[1],ps[2]),nbNN,offset1,offset2,offset3,crop_bg)
    pT1= pT1.astype('float32')
    pFLAIR=patch_extract_3D_v2(FLAIR,(ps[0],ps[1],ps[2]),nbNN,offset1,offset2,offset3,crop_bg)
    pFLAIR= pFLAIR.astype('float32')

    out_shape=(T1.shape[0],T1.shape[1],T1.shape[2],2)
    output=np.zeros(out_shape,T1.dtype)
    acu=np.zeros(out_shape[0:3],T1.dtype)

    ii=0 # Network ID

    for x in range(crop_bg,(nbNN[0]-1)*offset1+crop_bg+1,offset1):
        for y in range(crop_bg,(nbNN[1]-1)*offset2+crop_bg+1,offset2):
            for z in range(0,(nbNN[2]-1)*offset3+1,offset3):

                if(multi_in):
                    T = np.reshape(pT1[ii], (1,pT1.shape[1],pT1.shape[2],pT1.shape[3], 1))
                    F = np.reshape(pFLAIR[ii], (1,pFLAIR.shape[1],pFLAIR.shape[2],pFLAIR.shape[3], 1))
                    T=np.concatenate((T,F), axis=4)
                else:
                    T = np.reshape(pFLAIR[ii], (1,pFLAIR.shape[1],pFLAIR.shape[2],pFLAIR.shape[3], 1))



                lista=np.array([0,1])
                if(multi_out):
                    patches = model.predict(T)[0]
                else:
                    patches = model.predict(T)



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

def seg_majvote_flair(FLAIR,model,nbNN=[5,5,5],ps=[64,64,64],regularized=False):
    MASK = (1-(FLAIR==0).astype('int'))
    ind=np.where(MASK>0)
    indbg=np.where(MASK==0)
    crop_bg = 4
    overlap1 = np.floor((nbNN[0]*ps[0] - (FLAIR.shape[0]-2*crop_bg)) / (nbNN[0]-1))
    offset1 = ps[0] - overlap1.astype('int')
    overlap2 = np.floor((nbNN[1]*ps[1] - (FLAIR.shape[1]-2*crop_bg)) / (nbNN[1]-1))
    offset2 = ps[1] - overlap2.astype('int')
    overlap3 = np.floor((nbNN[2]*ps[2] - (FLAIR.shape[2]-2*crop_bg)) / (nbNN[2]-1))
    offset3 = ps[2]- overlap3.astype('int')

    pFLAIR=patch_extract_3D_v2(FLAIR,(ps[0],ps[1],ps[2]),nbNN,offset1,offset2,offset3,crop_bg)
    pFLAIR= pFLAIR.astype('float32')

    out_shape=(FLAIR.shape[0],FLAIR.shape[1],FLAIR.shape[2],2)
    output=np.zeros(out_shape,FLAIR.dtype)
    acu=np.zeros(out_shape[0:3],FLAIR.dtype)

    ii=0 # Network ID

    for x in range(crop_bg,(nbNN[0]-1)*offset1+crop_bg+1,offset1):
        for y in range(crop_bg,(nbNN[1]-1)*offset2+crop_bg+1,offset2):
            for z in range(0,(nbNN[2]-1)*offset3+1,offset3):

                T = np.reshape(pFLAIR[ii], (1,pFLAIR.shape[1],pFLAIR.shape[2],pFLAIR.shape[3], 1))



                lista=np.array([0,1])
                if(regularized):
                    patches = model.predict(T)[0]
                else:
                    patches = model.predict(T)



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

def seg_majvote_flair_ssl(FLAIR,model,nbNN=[5,5,5],ps=[64,64,64],regularized=False):
    MASK = (1-(FLAIR==0).astype('int'))
    ind=np.where(MASK>0)
    indbg=np.where(MASK==0)
    crop_bg = 4
    overlap1 = np.floor((nbNN[0]*ps[0] - (FLAIR.shape[0]-2*crop_bg)) / (nbNN[0]-1))
    offset1 = ps[0] - overlap1.astype('int')
    overlap2 = np.floor((nbNN[1]*ps[1] - (FLAIR.shape[1]-2*crop_bg)) / (nbNN[1]-1))
    offset2 = ps[1] - overlap2.astype('int')
    overlap3 = np.floor((nbNN[2]*ps[2] - (FLAIR.shape[2]-2*crop_bg)) / (nbNN[2]-1))
    offset3 = ps[2]- overlap3.astype('int')

    pFLAIR=patch_extract_3D_v2(FLAIR,(ps[0],ps[1],ps[2]),nbNN,offset1,offset2,offset3,crop_bg)
    pFLAIR= pFLAIR.astype('float32')

    out_shape=(FLAIR.shape[0],FLAIR.shape[1],FLAIR.shape[2],2)
    output=np.zeros(out_shape,FLAIR.dtype)
    acu=np.zeros(out_shape[0:3],FLAIR.dtype)

    ii=0 # Network ID

    for x in range(crop_bg,(nbNN[0]-1)*offset1+crop_bg+1,offset1):
        for y in range(crop_bg,(nbNN[1]-1)*offset2+crop_bg+1,offset2):
            for z in range(0,(nbNN[2]-1)*offset3+1,offset3):

                T = np.reshape(pFLAIR[ii], (1,pFLAIR.shape[1],pFLAIR.shape[2],pFLAIR.shape[3], 1))



                lista=np.array([0,1])
                if(regularized):
                    patches = model.predict(T)[:,:,:,:,0:2]
                else:
                    patches = model.predict(T)



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


################ for FLARE
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchio as tio
from PIL import Image


def update_labeled_folder_flare(listaIM,listaSEG,datafolder='data/',numbernotnullpatch=15,nbNN=[5,5,5],ps=[64,64,64], augment_times=0, bg=20):
    numb_data= len(sorted(glob.glob(os.path.join(datafolder,"x*.npy"))))
    clamp = tio.Clamp(-1000,1000)
    if(augment_times>0):
        data_transform= da_v3_single() 

    print(numb_data)
    for i in range(len(listaIM)):
        print(listaIM[i])
        print(listaSEG[i])
        IM= nii.load(listaIM[i]).get_data()
        seg=load_seg(listaSEG[i])
        print("normalization....")
        IM= clamp(IM[np.newaxis,:,:,:])[0]
        seg= seg[bg:seg.shape[0]-bg, bg:seg.shape[1]-bg, :]
        IM= IM[bg:IM.shape[0]-bg, bg:IM.shape[1]-bg, :]

        ratiox= 236/IM.shape[0]
        ratioy= 236/IM.shape[1]
        ratioz= 236/IM.shape[2]

        
        print("Rescaling....")
        IM= zoom(IM, (ratiox, ratioy, ratioz))
        seg= zoom(seg, (ratiox, ratioy, ratioz), mode='nearest', order=0)
        #seg= one_hot_encode(seg)
        IM= np.expand_dims(IM, -1)
        seg= np.expand_dims(seg, -1)

        if(augment_times>0):
            for k in range(augment_times):
                numb_data= len(sorted(glob.glob(os.path.join(datafolder,"x*.npy"))))
                IM_, seg_= data_transform.call(np.copy(IM), np.copy(seg))
                seg_= one_hot_encode(seg_[...,0])
        
                x_in, y_in , out_indx = notNull_flare(IM_,seg_,nbNN=nbNN,ps=ps,number=numbernotnullpatch)
                for j in range(x_in.shape[0]):
                    np.save(os.path.join(datafolder,'x_'+str(numb_data+j)+"_tile_"+str(out_indx[j])+'.npy'),x_in[j:j+1])
                    np.save(os.path.join(datafolder,'y_'+str(numb_data+j)+"_tile_"+str(out_indx[j])+'.npy'),y_in[j:j+1])
        else:
            seg= one_hot_encode(seg[...,0])     
            x_in, y_in , out_indx = notNull_flare(IM,seg,nbNN=nbNN,ps=ps,number=numbernotnullpatch)
            for j in range(x_in.shape[0]):
                np.save(os.path.join(datafolder,'x_'+str(numb_data+j+i*numbernotnullpatch)+"_tile_"+str(out_indx[j])+'.npy'),x_in[j:j+1])
                np.save(os.path.join(datafolder,'y_'+str(numb_data+j+i*numbernotnullpatch)+"_tile_"+str(out_indx[j])+'.npy'),y_in[j:j+1])
    return True


def patches_flare(IM,nbNN=[5,5,5],ps=[64,64,64]):
    crop_bg = 4
    ps1,ps2,ps3=ps[0], ps[1], ps[2]

    overlap1 = np.floor((nbNN[0]*ps1 - (IM.shape[0]-2*crop_bg)) / (nbNN[0]-1))
    offset1 = ps1 - overlap1.astype('int')
    overlap2 = np.floor((nbNN[1]*ps2 - (IM.shape[1]-2*crop_bg)) / (nbNN[1]-1))
    offset2 = ps2 - overlap2.astype('int')
    overlap3 = np.floor((nbNN[2]*ps3 - (IM.shape[2]-2*crop_bg)) / (nbNN[2]-1))
    offset3 = ps3 - overlap3.astype('int')
    pIM=patch_extract_3D_v2(IM,ps,nbNN,offset1,offset2,offset3,crop_bg)
    #pIM =np.expand_dims(pIM.astype('float32'),axis=4)
    return pIM

def notNull_flare(FLAIR,LAB,nbNN=[5,5,5],ps=[64,64,64],number=10):
    x_in= patches_flare(FLAIR,nbNN=nbNN,ps=ps+[1])
    y_in= patches_flare(LAB,nbNN=nbNN,ps=ps+[13]).astype('int')
    #sum_y_in=y_in[:,:,:,:,1:].sum(axis=(1,2,3,4))
    
    a=np.zeros((nbNN[0],nbNN[1],nbNN[2]))
    n=0
    for x in range(nbNN[0]):
        for y in range(nbNN[1]):
            for z in range(nbNN[2]):
                a[x,y,z]= n
                n=n+1

    random_idxs= []
    for stepx in range(1,nbNN[0]-1):
        for stepy in range(1,nbNN[1]-1):
            for stepz in range(1,nbNN[2]-1):
                tile_num= int(a[stepx,stepy,stepz])
                random_idxs.append(tile_num)
    
    random_idxs= np.array(random_idxs)
    np.random.shuffle(random_idxs)
    num=0
    x=x_in[random_idxs[0:1]]
    y=y_in[random_idxs[0:1]]
    indx_out= random_idxs[0:1]
    for i in random_idxs:
        if(num==number):
            break
        #suum=sum_y_in[i]
        #if suum>100:
        if(num==0):
            x=x_in[i:i+1]
            y=y_in[i:i+1]
            indx_out= np.array([i])
        else:
            x=np.concatenate((x,x_in[i:i+1]),axis=0)
            y=np.concatenate((y,y_in[i:i+1]),axis=0)
            indx_out= np.concatenate((indx_out,np.array([i])))
        num=num+1
    return x, y, indx_out

def one_hot_encode(seg, n_classes=13):
    seg= np.expand_dims(seg, -1)
    out= np.zeros((seg.shape[0], seg.shape[1], seg.shape[2], n_classes) )
    out[...,0]=1
    for i in range(1, n_classes):
        zone= np.where(seg==i)
        out[...,i:i+1][zone]=1
        out[...,0:1][zone]=0
    return out

def seg_region(flair, overlap=32):

    ax0= flair.sum(axis=(1,2))
    ax0=np.where(ax0>0)
    #ax0min=ax0[0][0]
    #ax0max=ax0[0][-1]
    ax0min=min(ax0[0])
    ax0max=max(ax0[0])
    ax1= flair.sum(axis=(0,2))
    ax1=np.where(ax1>0)
    #ax1min=ax1[0][0]
    #ax1max=ax1[0][-1]
    ax1min=min(ax1[0])
    ax1max=max(ax1[0])
    ax2= flair.sum(axis=(0,1))
    ax2=np.where(ax2>0)
    #ax2min=ax2[0][0]
    #ax2max=ax2[0][-1]
    ax2min=min(ax2[0])
    ax2max=max(ax2[0])

    if(overlap>0):
        ax0min=max([ax0min-overlap,0])
        ax0max=min([ax0max+overlap, flair.shape[0] ])
        ax1min=max([ax1min-overlap,0])
        ax1max=min([ax1max+overlap, flair.shape[1] ])
        ax2min=max([ax2min-overlap,0])
        ax2max=min([ax2max+overlap, flair.shape[2] ])

    return ax0min,ax0max,ax1min,ax1max,ax2min,ax2max

def save_slices(IM, seg, datafolder, axes=[0,1,2], step=1, all_slices=False):
    numb_data= len(sorted(glob.glob(os.path.join(datafolder,"x*.npy"))))
    transpose = [(0, 1, 2, 3 ), (1, 0, 2, 3), (2, 0, 1, 3)]
    print(IM.shape)
    count=0
    for axis_to_take in axes:
        SEG=np.transpose(np.copy(seg), transpose[axis_to_take])
        IM=np.transpose(np.copy(IM), transpose[axis_to_take])
        indices= np.arange(SEG.shape[0])
        np.random.shuffle(indices)
        for sl_num in indices[::step]:
            condition= (SEG[sl_num].sum()>5) or all_slices
            if(condition):
                x=IM[sl_num:sl_num+1]
                y= SEG[sl_num:sl_num+1]
                """
                save_img(x[0,:,:,0], "x1.png")
                save_img(y[0,:,:,0], "y1.png")
                end
                """
                np.save(os.path.join(datafolder,'x_'+str(numb_data+count)+'.npy'),x)
                np.save(os.path.join(datafolder,'y_'+str(numb_data+count)+'.npy'),y)
                count=count+1


class da_v3_abstract(object):
    def __init__(self):

        gaussian_blur= tio.RandomBlur(std= (0.25, 1))
        edge_enhancement= tio.Lambda(lambda x: x+ np.random.uniform(1,5)*(x- gaussian_blur(x) ))
        rot_flip= tio.Lambda(lambda x: torch.from_numpy( rot_90(x.numpy().transpose(1,2,3,0) ).transpose(3,0,1,2).copy()   )  )
        distortion= tio.Lambda(lambda x: torch.from_numpy( ndimage.uniform_filter(x.numpy(), (1,1,int(np.random.uniform(1,6)),1)).copy()))
        
        self.spatial_transform = tio.Compose([
                    tio.OneOf({                                # either
                        tio.RandomAffine(scales=(1,1.3),degrees=15, default_pad_value='otsu' ): 0.4,               # random affine
                        tio.RandomElasticDeformation(max_displacement=4): 0.6,   # or random elastic deformation
                    }, p=1),                                 # applied to 80% of images

                    ])
        #self.spatial_transform =tio.RandomElasticDeformation(max_displacement=4, p=0.8)

        
        #"""
        self.other_transforms = tio.Compose([

                    tio.RandomBiasField(p=0.5),                # magnetic field inhomogeneity 30% of times
                    
                    tio.OneOf({ 
                        tio.RandomAnisotropy(downsampling = (1.5, 4)):1,              # make images look anisotropic 25% of times
                        distortion:1,
                        tio.RandomBlur(std= (0.25, 1, 0.25, 1, 0.25, 1)):1 ,                    # blur 25% of times
                        edge_enhancement:1,
                    }, p=0.5 ),
                    
                    tio.OneOf({                                # either
                        tio.RandomMotion(degrees=5 , translation= 4): 1,                 # random motion artifact
                        tio.RandomSpike(intensity=(0.1,0.15) ): 1,                  # or spikes
                        tio.RandomGhosting(num_ghosts=(4,10), intensity=(0.25,0.75)): 1,               # or ghosts
                    }, p=0.5),                                 # applied to 50% of images            
                    tio.RandomNoise( std=(0, 0.1), p=0.5),                   # Gaussian noise 25% of times
                    
                    ])

def rot_90(a):
    op=np.random.choice(10,1)
    #print(op)
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
    return a

from torchio import Subject, ScalarImage, LabelMap
class da_v3_single(da_v3_abstract):
    def __init__(self):
        super().__init__()

    def call(self,inputs, label):
        inputs, label = inputs.transpose((3,0,1,2)), label.transpose((3,0,1,2))
        #X=np.concatenate((inputs,label), axis=0)
        #X= self.spatial_transform(X)
        X={'t1':inputs , 'seg':label }
        subject = Subject(
                                t1=ScalarImage(tensor=inputs),  # this class is new
                                label=LabelMap(tensor=label),
                            )
        X= self.spatial_transform(subject)
        inputs= self.other_transforms(X.t1)
        return inputs.numpy().transpose((1,2,3,0)), X.label.numpy().transpose((1,2,3,0))

def update_labeled_folder2D(listaIM,listaSEG, datafolder='data/',augment_times=0, bg=0, step=1, all_slices=False, axis=2):
    clamp = tio.Clamp(-1000,1000)
    if(augment_times>0):
        data_transform= da_v3_single() 

    for i in range(len(listaIM)):
        print(listaIM[i])
        print(listaSEG[i])
        IM= nii.load(listaIM[i]).get_data()
        seg=load_seg(listaSEG[i])
        print("normalization....")
        IM= clamp(IM[np.newaxis,:,:,:])[0]
        seg= seg[bg:seg.shape[0]-bg, bg:seg.shape[1]-bg, :]
        IM= IM[bg:IM.shape[0]-bg, bg:IM.shape[1]-bg, :]

        ratiox= 236/IM.shape[0]
        ratioy= 236/IM.shape[1]
        ratioz= 236/IM.shape[2]

        
        print("Rescaling....")
        IM= zoom(IM, (ratiox, ratioy, ratioz))
        seg= zoom(seg, (ratiox, ratioy, ratioz), mode='nearest', order=0)
        
        #seg= one_hot_encode(seg)
        IM= np.expand_dims(IM, -1)
        seg= np.expand_dims(seg, -1)
        
        if(augment_times>0):
            for k in range(augment_times):
                IM_, seg_= data_transform.call(IM, seg)
                seg_= one_hot_encode(seg_[...,0])
                save_slices(IM_,seg_,datafolder, axes=[axis], step=3, all_slices=all_slices)
        else:
            seg= one_hot_encode(seg[...,0])
            save_slices(IM,seg,datafolder, axes=[axis], all_slices=all_slices) 
    return True

def IQDA2D(x_):
    for i in range(0,x_.shape[0]):
        op=np.random.choice(3,1) # 0:nada, 1:sharp, 2:blur, 3: axial blur 3, 4: axial blur 5, 5: axial blur 2
        for j in range(x_.shape[-1]):
            if(op==1):
                x_[i,:,:,j] = 2*x_[i,:,:,j]-ndimage.uniform_filter(x_[i,:,:,j], (3,3))
            if(op==2):
                x_[i,:,:,j] = ndimage.uniform_filter(x_[i,:,:,j], (3,3))

        return x_

def IQDA2D_v2(x_, op=None):
    if(op==None):
        op=np.random.choice(3,1 ,p=[0.1,0.45,0.45]) # 0:nada, 1:sharp, 2:blur, 3: axial blur 3, 4: axial blur 5, 5: axial blur 2
    std=[np.random.uniform(0.25,0.6),np.random.uniform(0.25,0.6)]
    alpha= np.random.uniform(1,3)

    for j in range(x_.shape[-1]):
        if(op==1):
            x_[...,j] =  x_[...,j]+ alpha*(x_[...,j]-ndimage.gaussian_filter(x_[...,j], std))
        if(op==2):
            x_[...,j] = ndimage.gaussian_filter(x_[...,j], std)
    return x_

def batch_rot90_2D(lesion_batch):
    for i in range(lesion_batch.shape[0]):
        a=lesion_batch[i]
        op=np.random.choice(4,1)
        if(op==1):
            a=np.rot90(a,k=2,axes=(1,0))
        elif(op==3):
            a=np.rot90(a,k=1,axes=(1,0))
        elif(op==2):
            a=np.rot90(a,k=3,axes=(1,0))

        op=np.random.choice(3,1)
        if(op==1):
            a=a[-1::-1,:]
        elif(op==2):
            a=a[:,-1::-1]
        lesion_batch[i]=a
    return lesion_batch

class TileDataset2D(Dataset):
    """Face Landmarks dataset."""
    def __init__(self,files_dir,transform=None, da=False, img_size=[128,128]):
        self.x_list= sorted(glob.glob(files_dir+"x*.npy"))
        self.random_idx= np.arange(len(self.x_list))
        np.random.shuffle(self.random_idx)
        self.da=da
        self.img_size=img_size

    def __len__(self):
        return len(self.x_list)

    def Mixup(self,x1,x2,y1,y2,alfa=0.3):
        a=np.random.beta(alfa,alfa)
        x=a*x1+(1-a)*x2
        y=a*y1+(1-a)*y2
        return x,y

    def iqda(self, x, y):
        x=IQDA2D(x)
        #return self.rot(x,y)
        return x,y

    def iqda_v2(self, x, y):
        x[0,:,:,:]=IQDA2D_v2(x[0,:,:,:])
        #return self.rot(x,y)
        return x,y

    def rot(self, x, y):
        X=np.concatenate((x,y), axis=-1)
        X=batch_rot90_2D(X)
        x=X[:,:,:,0:x.shape[-1]]
        y=X[:,:,:,x.shape[-1]:x.shape[-1]+y.shape[-1]]
        return x,y
    
    def rot4(self, x, y, z, h):
        X=np.concatenate((x,y,z,h), axis=-1)
        X=batch_rot90_2D(X)
        x=X[:,:,:,0:x.shape[-1]]
        y=X[:,:,:,x.shape[-1]:x.shape[-1]+y.shape[-1]]
        z=X[:,:,:,x.shape[-1]+y.shape[-1]:x.shape[-1]+y.shape[-1]+z.shape[-1]]
        h=X[:,:,:,x.shape[-1]+y.shape[-1]+z.shape[-1]:]
        return x,y,z,h

    def get_random_patch(self, x1,y1, size=[128,128]):
        x_ran= np.random.randint( size[0]//2, x1.shape[1]- (size[0]-size[0]//2) +1 )
        y_ran= np.random.randint( size[0]//2, x1.shape[2]- (size[1]-size[1]//2) +1 )
        xo=x_ran- size[0]//2
        yo=y_ran- size[1]//2
        return x1[:,xo: xo+size[0], yo: yo+size[1],:],y1[:,xo: xo+size[0], yo: yo+size[1],:]

    def get_random_patch_v2(self, x1, y1, size=[128,128]):
        map_lab=np.where(np.sum(y1[0,:,:,:],2)>0)
        
        choice=np.random.randint(0,len(map_lab[0]))
        x_ran= map_lab[0][choice]
        y_ran= map_lab[1][choice]

        xo=max([0,x_ran- (size[0]//2)])
        yo=max([0,y_ran- (size[1]//2)])
        xt=xo+size[0]
        yt=yo+size[1]

        if(xt>x1.shape[1]):
            xt=x1.shape[1]
            xo=xt-size[0]
        if(yt>x1.shape[2]):
            yt=x1.shape[2]
            yo=yt-size[1]

        return x1[:,xo: xt, yo: yt,:],y1[:,xo: xt, yo: yt,:]

    def normalize_input_std(self,input_ ):
        m1=np.mean(input_)
        s1=np.std(input_)+0.00001
        input_=(input_-m1)/s1
        return input_
    
    def normalize_input_minmax(self,input_ ):
        mi=np.min(input_)
        ma=np.max(input_)
        #print(mi, ma)
        if((ma-mi)==0):
            ma=1
            mi=0
        input_=(input_-mi)/(ma-mi)
        return input_

    def get_pair(self, idx):
        x1_name=self.x_list[idx]
        y1_name=x1_name.replace('x_','y_')
        x1=np.load(x1_name).astype('float')
        y1=np.load(y1_name).astype('float')
        
        #x1[...,0]= normalize_image(x1[...,0], 't1')
        
        x1= self.normalize_input_minmax(x1 )

        #end
        
        if(not x1.shape==self.img_size):
            x1,y1=self.get_random_patch(x1,y1, size=self.img_size)
        return x1,y1

    def __getitem__(self, idx):
        #print(idx)
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        x1,y1= self.get_pair(idx)

        if(self.da=="rot"):
            inputs,label= self.rot( x1, y1)

        if(self.da=="iqda"):
            inputs,label= self.iqda( x1, y1)

        if(self.da=="iqda_v2"):
            inputs,label= self.iqda_v2( x1, y1)

        elif(self.da=="mixup"):
            ind_ran=self.random_idx[idx]
            x2,y2= self.get_pair(ind_ran)
            x2=np.load(x2_name)
            y2=np.load(y2_name)
            inputs,label=self.Mixup(x1,x2,y1,y2)

        else:
            inputs=x1
            label=y1
        """
        print(label.max())
        save_img(label[0,:,:,0], 'y.png')
        save_img(inputs[0,:,:,0], 'x1.png')
        save_img(inputs[0,:,:,1], 'x2.png')
        end
        #"""
        inputs= torch.from_numpy(  inputs[0].transpose((2,0,1))  ).float()
        label=torch.from_numpy(  label[0].transpose((2,0,1))  ).float()
        sample = {'inputs': inputs, 'label': label}
        return sample

class TileDataset_with_reg_2D(TileDataset2D):
    """Face Landmarks dataset."""
    def __init__(self,files_dir, transform=None, da=False, img_size=[128,128]):
        super().__init__(files_dir, transform, da, img_size)
        
        self.x_list= sorted(glob.glob(files_dir+"x*.npy"))    
        self.transform = transform
        self.random_idx= np.arange(len(self.x_list))
        np.random.shuffle(self.random_idx)
        self.da=da

    def Mixup_it(self,x1,x2,y1,y2, x1_,x2_,y1_,y2_, sim,sim_, alfa=0.3 ):
        a=np.random.beta(alfa,alfa)
        x1=a*x1+(1-a)*x1_
        y1=a*y1+(1-a)*y1_
        x2=a*x2+(1-a)*x2_
        y2=a*y2+(1-a)*y2_
        sim= a*sim+(1-a)*sim_
        return x1,x2,y1,y2,sim
        
    def gen_iqda_2it(self,idx,same_ratio=0.8,sim='output_diff'):
        op=np.random.choice(2,1,p=[same_ratio,1-same_ratio]) # 0 same 1 different
        x1, y1= self.get_pair(idx)
        if(op==0):        
            inter_sim=0.0
            y2, x2 = np.copy(y1), np.copy(x1) 
        else:
            x1, y1= self.get_pair(idx)
            ind_ran=self.random_idx[idx]
            x2,y2= self.get_pair(ind_ran)
            inter_sim= np.exp(-np.mean(np.square(y1-y2)))    
        return x1, x2, y1, y2 ,inter_sim

    def __getitem__(self, idx):
        #print(idx)
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x1, x2, y1, y2 ,inter_sim= self.gen_iqda_2it(idx)
        #print(y2.shape)
        #print(y1.shape)
        
        """
        save_img(x1[0,:,:,0], "x1.png")
        save_img(x2[0,:,:,0], "x2.png")
        save_img(y1[0].argmax(-1), "y1.png")
        save_img(y2[0].argmax(-1), "y2.png")
        end
        """
        if(self.da=="rot"):
            x1, y1, x2, y2= self.rot4( x1, y1, x2, y2)

        elif(self.da=="iqda"):
            x1, y1= self.iqda( x1, y1)
            x2, y2= self.iqda( x2, y2)
            
            x1, y1, x2, y2= self.rot4( x1, y1, x2, y2)


        elif(self.da=="iqda_v2"):
            x1, y1= self.iqda_v2( x1, y1)
            x2, y2= self.iqda_v2( x2, y2)
            x1, y1, x2, y2= self.rot4( x1, y1, x2, y2)
        
        elif(self.da=="mixup"):
            ind_ran=self.random_idx[idx]
            x1_, x2_, y1_, y2_ ,inter_sim_= self.gen_iqda_2it(ind_ran)
            x1,x2,y1,y2,inter_sim =self.Mixup_it(x1,x2,y1,y2, x1_,x2_,y1_,y2_, inter_sim,inter_sim_)
            x1, y1, x2, y2= self.rot4( x1, y1, x2, y2)

        
        sample1 = {'inputs': x1, 'label': y1}
        sample2 = {'inputs': x2, 'label': y2}

        """
        if(np.isnan(x1+y1+x2+y2).sum()):
            print(np.isnan(x1).sum())
            print(np.isnan(y1).sum())
            print(np.isnan(x2).sum())
            print(np.isnan(y2).sum())
        """
        
        inter_sim= torch.tensor(inter_sim).float().to(device)
        #return sample1, sample2, inter_sim
        inputs1, inputs2 = torch.from_numpy(  x1[0].transpose((2,0,1))  ).float(), torch.from_numpy(  x2[0].transpose((2,0,1))  ).float() 
        label1, label2=torch.from_numpy(  y1[0].transpose((2,0,1))  ).float(), torch.from_numpy(  y2[0].transpose((2,0,1))  ).float()
        return {'inputs1': inputs1, 'label1': label1,
                'inputs2': inputs2, 'label2': label2,
                'inter_sim': inter_sim}

def save_img(img, name):
    Image.fromarray(((img-img.min())/(img.max()-img.min())*255).astype(np.uint8), 'L').save(name)

def train_model_2D(model,optimizer,criterion,val_criterion, Epoch,dataset_loader,
                dataset_loader_val,eval_strategy,out_PATH,best_val_loss=100,early_stop=False,regularized=True,  loss_weights=[1,0.01],
                early_stop_treshold=100):
    count=0
    for epoch in range(Epoch):
        running_loss = 0.0
        model.train()
        k=0
        with tqdm(dataset_loader) as tepoch:
            for sample in tepoch:
                k=k+1
                optimizer.zero_grad()
                if(regularized):
                    
                    inputs1, labels1 =sample['inputs1'].to(device) , sample['label1'].to(device)
                    inputs2, labels2 =sample['inputs2'].to(device) , sample['label2'].to(device)

                    
                    bottleneck1 = model.encoder(inputs1)
                    pred1=model.segmentation_head(model.decoder(*bottleneck1))
                    bottleneck2 = model.encoder(inputs2)
                    pred2=model.segmentation_head(model.decoder(*bottleneck2))

                    #print(bottleneck1[-1].size())

                    latent_distance= 2*torch.sum(torch.square(bottleneck1[-1]-bottleneck2[-1]), dim=(1,2,3))/(torch.mean(torch.square(bottleneck1[-1]), dim=(1,2,3))+torch.mean(torch.square(bottleneck2[-1]), dim=(1,2,3)))
                    consistency=  torch.mean(latent_distance* torch.exp(-sample['inter_sim']))
                    loss_supervised= (criterion(pred1, labels1)+criterion(pred2, labels2))/2
                    loss = loss_supervised *loss_weights[0]+ consistency*loss_weights[1]
                    tepoch.set_postfix(supervised_loss=loss_supervised.item(), regularization=consistency.item())

                else:
                    inputs, labels =sample['inputs'].to(device) , sample['label'].to(device)
                    outputs= model(inputs)
                    loss = criterion(outputs, labels)
                    tepoch.set_postfix(loss=loss.item(), Dice=100. * (1-loss.item()))

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                #break

            running_loss_val =0.0
            model.eval()
            j=0
        if(not dataset_loader_val is None):
            for sample in tqdm(dataset_loader_val):
                j=j+1
                inputs, labels =sample['inputs'].to(device) , sample['label'].to(device)
                with torch.no_grad():
                    outputs = model(inputs)
                loss = val_criterion(outputs, labels)
                running_loss_val += loss.item()
            #res
            val_loss = running_loss_val/(j+1)
            
            train_loss= running_loss/(k+1)
            print('Epoch: '+str(epoch)+'   Loss: '+str(train_loss)+'   Val_loss: '+str(val_loss))

            if(val_loss<best_val_loss):
                #savingmodel
                count=0
                torch.save(model, out_PATH)
                best_val_loss=val_loss
                print('saving weights. new best loss: '+str(best_val_loss))
            else:
                count=count+1
                if(early_stop):
                    if(count>early_stop_treshold):
                        print('val loss did not improve for:'+str(early_stop_treshold)+' epochs. Stopping training.')
                        break
    return best_val_loss
    
def features_from_names_pytorch2D(listaT1=None, idices=None, Models=None, image_size=[128, 128]):
    [model.eval() for model in Models]
    if(not idices is None):
        listaT1=listaT1[idices]
        
    file_names=[]
    transpose = [(0, 1, 2), (1, 0, 2), (2, 0, 1)]

    for i in range(len(listaT1)):
        T1= nii.load(listaT1[i]).get_data()
        
        print("normalization....")
        IM= clamp(IM[np.newaxis,:,:,:])[0]
        IM= IM[bg:IM.shape[0]-bg, bg:IM.shape[1]-bg, :]

        ratiox= 236/IM.shape[0]
        ratioy= 236/IM.shape[1]
        ratioz= 236/IM.shape[2]

        
        print("Rescaling....")
        IM= zoom(IM, (ratiox, ratioy, ratioz))

        T1_cropped=crop_center(IM,image_size[0],image_size[0],image_size[0])
        
        for axe in [0,1,2]:
            x_in=np.transpose(np.copy(T1_cropped), transpose[axe])
            x_in= crop_center(x_in,32,image_size[0],image_size[0])
            x_in=np.expand_dims(x_in,axis=1)
            mi=np.min(x_in)
            ma=np.max(x_in)
            x_in=(x_in-mi)/(ma-mi)  
            x_in= torch.from_numpy(x_in).to(device).float()
            
            with torch.no_grad():
                bottleneck_out = Models[0].encoder(x_in)[-1]

            if(axe==0):
                bottleneck_out_mean=bottleneck_out.cpu().numpy().mean(axis=(0,2,3))
            else:
                bottleneck_out_mean=np.concatenate((bottleneck_out_mean, bottleneck_out.cpu().numpy().mean(axis=(0,2,3))))

        
        if(i==0):
            bottleneck_features=np.expand_dims(bottleneck_out_mean,axis=0)
        else:
            bottleneck_features=np.concatenate((bottleneck_features,np.expand_dims(bottleneck_out_mean,axis=0)),axis=0)
        file_names.append(listaT1[i])
    bottleneck_features= np.expand_dims(bottleneck_features,axis=1)
    return bottleneck_features,file_names

def update_data_folder_pytorch2D(Models,new_pseudo,listaT1,datafolder='data',img_size=[128,128]):
    [model.eval() for model in Models]
    numb_data= len(sorted(glob.glob(datafolder+"x*.npy")))
    clamp = tio.Clamp(-1000,1000)
    
    for i in range(len(new_pseudo)):
        print(listaT1[new_pseudo[i]])
        IM= nii.load(listaT1[i]).get_data()
        IM= clamp(IM[np.newaxis,:,:,:])[0]
        

        IM= IM[bg:IM.shape[0]-bg, bg:IM.shape[1]-bg, IM.shape[2]//5:IM.shape[2]-IM.shape[2]//5]

        ratiox= 236/IM.shape[0]
        ratioy= 236/IM.shape[1]
        ratioz= 236/IM.shape[2]

        
        print("Rescaling....")
        IM= zoom(IM, (ratiox, ratioy, ratioz))
        
        seg = seg_soft_majvote_times_2D(IM,Models,ps=img_size)
        seg= np.argmax(seg, axis=0)

        transpose = [(0, 1, 2), (1, 0, 2), (2, 0, 1)]
        #print(seg.shape)
        for axis in [0,1,2]:
            IMG=np.transpose(np.copy(IM), transpose[axis])
            SEG=np.transpose(np.copy(seg), transpose[axis])
            label_distribution= (SEG>0).astype('int')  
            max_sli= np.sum(label_distribution, axis=(1,2) )
            quantile= np.quantile(max_sli, 0.7)
            to_take= np.where(max_sli>quantile)
            SEG= one_hot_encode(SEG)
            IMG= np.expand_dims(IMG, -1)
            save_slices(IMG[to_take[0],:,:,:],SEG[to_take[0],:,:,:],datafolder+str(axis), axes=[0], step=1)
    return True


def seg_soft_majvote_times_2D(img,MODELS,ps=[224,224], out_dim=5,
        offset1=220,offset2=220,mini_batch=16,crop_bg=0,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    out_shape=( out_dim,img.shape[0], img.shape[1],img.shape[2])
    OUTPUT=np.zeros(out_shape,img.dtype)
    SEG=np.zeros(out_shape,img.dtype)

    #model axe0, axe1, axe2
    for axis_to_take, model in enumerate(MODELS):
        transpose = [(0, 1, 2), (1, 0, 2), (2, 0, 1)]
        transpose_4dim = [(1, 0, 2, 3), (2, 0, 1, 3), (3, 0, 1, 2)]
        transpose_inv = [(1, 0, 2, 3), (1, 2, 0, 3), (1, 2, 3, 0)]
        weights=[1,1,1]
        output=np.transpose(np.copy(OUTPUT), transpose_4dim[axis_to_take])
        IMG=np.transpose(np.copy(img), transpose[axis_to_take])

        for x in range(crop_bg,  IMG.shape[0] ,mini_batch):
            xx = x+mini_batch
            if xx> IMG.shape[0]:
                xx = IMG.shape[0]
            for y in range(crop_bg,  IMG.shape[1] ,offset1):
                yy = y+ps[0]
                if yy> IMG.shape[1]:
                    yy = IMG.shape[1]
                    y=yy-ps[0]
                for z in range(crop_bg,  IMG.shape[2] ,offset2):
                    zz = z+ps[1]
                    if zz> IMG.shape[2]:
                        zz = IMG.shape[2]
                        z=zz-ps[1]
                    T = np.reshape(   IMG[x:xx,y:yy,z:zz] , (xx-x,1,ps[0],ps[1]))
                    mi=np.min(T)
                    ma=np.max(T)
                    T=(T-mi)/(ma-mi)
                    T=torch.from_numpy(T).to(device).float()
                    with torch.no_grad():
                        patches = model(T)
                        patches= patches.cpu().numpy()
                    output[x:xx,:,y:yy,z:zz]=output[x:xx,:,y:yy,z:zz]+patches[0:xx-x,:,0:yy-y,0:zz-z]
        SEG=SEG+np.transpose(output, transpose_inv[axis_to_take])*weights[axis_to_take]

    return SEG
