from sklearn.datasets import load_digits
import os, glob, random, math, operator, umap
import numpy as np
import nibabel as nii
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import patch_extraction
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,DBSCAN

class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self,filepath,validation_data=(), monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        #super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        super(CustomModelCheckpoint, self).__init__(filepath,monitor=monitor, verbose=verbose,
                     save_best_only=save_best_only, save_weights_only=save_weights_only,
                     mode=mode, period=period)
        self.X_val, self.y_val = validation_data
        self.y_val =np.argmax(self.y_val, axis=-1)
    def pred_val_score(self):
        y_pred = self.model.predict(self.X_val, batch_size=1,verbose=0)
        y_pred = np.argmax(y_pred, axis=-1)
        return score_batch( y_pred,self.y_val)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:

                current =self.pred_val_score()
                print('score of validation:'+str(current))
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)

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

        #seg = seg_majvote(T1,FLAIR,model,nbNN=[5,5,5],ps=[96,96,96])
        seg = seg_majvote(T1,FLAIR,model,nbNN=[3,3,3],ps=[96,96,96],regularized=regularized)
        #x_in, y_in = random_patches_andLAB(T1,FLAIR,seg,number=5)
        x_in, y_in , out_indx= notNull_patches_andLAB(T1,FLAIR,seg,number=numbernotnullpatch)
        for j in range(x_in.shape[0]):
            np.save(datafolder+'x_'+str(numb_data+j+i*numbernotnullpatch)+"_tile_"+str(out_indx[j])+'.npy',x_in[j:j+1])
            np.save(datafolder+'y_'+str(numb_data+j+i*numbernotnullpatch)+"_tile_"+str(out_indx[j])+'.npy',y_in[j:j+1])
    return True

def update_data_folder_flair(model,new_pseudo,listaFLAIR,listaMASK=None,datafolder='data/',numbernotnullpatch=2,regularized=False,nbNN=[3,3,3],ps=[64,64,64]):
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

def update_labeled_folder_flair(listaFLAIR,listaSEG,listaMASK=None,datafolder='data/',numbernotnullpatch=15,nbNN=[3,3,3],ps=[64,64,64]):
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

def patches(T1,FLAIR,nbNN=[3,3,3],ps=[96,96,96]):
    crop_bg = 4
    [ps1,ps2,ps3]=ps

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

def patches_flair(FLAIR,nbNN=[3,3,3],ps=[96,96,96]):
    crop_bg = 4
    [ps1,ps2,ps3]=ps

    overlap1 = np.floor((nbNN[0]*ps1 - (FLAIR.shape[0]-2*crop_bg)) / (nbNN[0]-1))
    offset1 = ps1 - overlap1.astype('int')
    overlap2 = np.floor((nbNN[1]*ps2 - (FLAIR.shape[1]-2*crop_bg)) / (nbNN[1]-1))
    offset2 = ps2 - overlap2.astype('int')
    overlap3 = np.floor((nbNN[2]*ps3 - (FLAIR.shape[2]-2*crop_bg)) / (nbNN[2]-1))
    offset3 = ps3 - overlap3.astype('int')
    pFLAIR=patch_extraction.patch_extract_3D_v2(FLAIR,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
    pFLAIR =np.expand_dims(pFLAIR.astype('float32'),axis=4)
    return pFLAIR

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
                indx_out= np.array([i])
            else:
                x=np.concatenate((x,x_in[i:i+1]),axis=0)
                y=np.concatenate((y,y_in[i:i+1]),axis=0)
                indx_out= np.concatenate((indx_out,np.array([i])))
            num=num+1
    return x, y, indx_out

def notNull_flair_andLAB(FLAIR,LAB,nbNN=[3,3,3],ps=[64,64,64],number=10):
    x_in= patches_flair(FLAIR,nbNN=nbNN,ps=ps)
    y_in= patches(1-LAB,LAB,nbNN=nbNN,ps=ps).astype('int')
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
                indx_out= np.array([i])
            else:
                x=np.concatenate((x,x_in[i:i+1]),axis=0)
                y=np.concatenate((y,y_in[i:i+1]),axis=0)
                indx_out= np.concatenate((indx_out,np.array([i])))
            num=num+1
    return x, y, indx_out

def seg_majvote(T1,FLAIR,model,nbNN=[5,5,5],ps=[96,96,96],multi_out=False,multi_in=True):
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

def seg_majvote_flair(FLAIR,model,nbNN=[5,5,5],ps=[96,96,96],regularized=False):
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

    pFLAIR=patch_extraction.patch_extract_3D_v2(FLAIR,(ps[0],ps[1],ps[2]),nbNN,offset1,offset2,offset3,crop_bg)
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

def seg_majvote_flair_ssl(FLAIR,model,nbNN=[5,5,5],ps=[96,96,96],regularized=False):
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

    pFLAIR=patch_extraction.patch_extract_3D_v2(FLAIR,(ps[0],ps[1],ps[2]),nbNN,offset1,offset2,offset3,crop_bg)
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
