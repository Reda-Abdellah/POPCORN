import umap
from sklearn.datasets import load_digits
import os
import glob
import numpy as np
import nibabel as nii
import math
import operator
from scipy.ndimage.interpolation import zoom
from keras.models import load_model
from scipy import ndimage
import scipy.io as sio
#import matplotlib.pyplot as plt
import modelos
from utils import *
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
import Data_fast
import losses,metrics

os.environ["CUDA_VISIBLE_DEVICES"]='2'
Rootpath=os.getcwd()
nbNN=[5,5,5]
ps=[96,96,96]
Epoch_per_step=2
increment_new_data=100
datafolder='data3/'
load_precomputed_features=True
#model=modelos.load_UNET3D_SLANT27_v2_groupNorm(96,96,96,2,2,24,0.5)
#model.load_weights(filepath)
#model.compile(optimizer=optimizers.Adam(0.0001), loss='mse')
#model.compile(optimizer=optimizers.Adam(0.0001), loss=losses.norm_mse)

model=modelos.load_UNET3D_bottleneck_regularized(ps[0],ps[1],ps[2],2,2,20,0.5,groups=4)
model.compile(optimizer=optimizers.Adam(0.0001), loss=[losses.GJLsmooth,losses.BottleneckRegularized],loss_weights=[1,0.2])
#model.compile(optimizer=optimizers.Adam(0.0001), loss=losses.GJLsmooth, metrics=[metrics.mdice])
#savemodel=ModelCheckpoint('curruc.h5', monitor='val_mdice', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)

fun=get_bottleneck_features_func(model)
listaT1 = sorted(glob.glob("../lib/volbrain_qc/n_mfmni*t1*.nii*"))
listaFLAIR = sorted(glob.glob("../lib/volbrain_qc/n_mfmni*flair*.nii*"))
listaMASK = sorted(glob.glob("../lib/volbrain_qc/mask*.nii*"))
listaMASK=np.array(listaMASK)

#listaT1 =listaT1[:5]
#listaFLAIR =listaFLAIR[:5]
"""
listaT1 = sorted(glob.glob("../lib/ISBI_preprocess/test*mprage*.nii*"))
listaFLAIR = sorted(glob.glob("../lib/ISBI_preprocess/test*flair*.nii*"))
"""

listaT1=np.array(listaT1)
listaFLAIR=np.array(listaFLAIR)

listaT1_isbi = sorted(glob.glob("../lib/isbi_final_train_preprocessed/training*mprage*.nii*"))
listaFLAIR_isbi = sorted(glob.glob("../lib/isbi_final_train_preprocessed/training*flair*.nii*"))

listaT1_isbi=np.array(listaT1_isbi)
listaFLAIR_isbi=np.array(listaFLAIR_isbi)

x_train,y_train,x_val,y_val=load_isbi(one_out=False)

labeld_to_data(x_train,y_train,datafolder=datafolder)

#indexs are used to index from listaT1
pseudolabeled_indxs=[]
unlabeled_indxs= range(len(listaT1))
unlabeled_num=len(unlabeled_indxs)
pseudolabeled_num=len(pseudolabeled_indxs)
labeled_num=len(listaT1_isbi)


step=0

#filepath="One_Tile_96_2mods.h5"
#filepath="One_2mods_96_MSO_andISBI_gen_IQDA.h5"
filepath="One_2mods_2it033same_loss1[1_02]_96_MSO_andISBI_gen_IQDA.h5"
model.load_weights(filepath)

#bottleneck_features_labeled,file_names= features_from_names(listaT1_isbi,listaFLAIR_isbi,fun)
#bottleneck_features_unlabeled,file_names= features_from_names(listaT1[unlabeled_indxs],listaFLAIR[unlabeled_indxs],fun,listaMASK[unlabeled_indxs])
#bottleneck_features_unlabeled,file_names= features_from_names(listaT1[unlabeled_indxs],listaFLAIR[unlabeled_indxs],fun)
#bottleneck_features_labeled=np.load('bottleneck_features_labeled.npy')
#bottleneck_features_unlabeled=np.load('bottleneck_features_unlabeled_volbrain.npy')

#rank_distance= pca_rank(bottleneck_features_unlabeled,bottleneck_features_labeled,n_components=16)
#np.save('rank_distance_volbrain_pca16.npy',rank_distance)
#print(rank_distance.shape)
#rank_distance= tsne_rank(bottleneck_features_unlabeled,bottleneck_features_labeled,n_components=3)
#rank_distance= tsne_rank(bottleneck_features_unlabeled,bottleneck_features_labeled,n_components=3)
#np.save('rank_distance_volbrain_tsne3.npy',rank_distance)
if(load_precomputed_features):
    ##rank_distance=np.load('rank_distance_volbrain.npy')
    rank_distance=np.load('rank_distance_volbrain_tsne3.npy')

while(unlabeled_num>increment_new_data):
    step=step+1
    print('step: '+str(step))
    print('loading new data...')
    #new_pos_in_features = give_n_closest(rank_distance,n_indxs=increment_new_data)
    new_pos_in_features = give_dist_for_Kclosest(rank_distance,n_indxs=increment_new_data,k=5)
    print(new_pos_in_features)
    not_new_pos_in_features = [x for x in range(unlabeled_num) if x not in new_pos_in_features]
    #new_pseudo=np.array(unlabeled_indxs)[new_pos_in_features]
    #new_pseudo=new_pseudo.tolist()
    #update indexes
    pseudolabeled_indxs= pseudolabeled_indxs+ new_pos_in_features
    unlabeled_indxs =  [x for x in unlabeled_indxs if x not in pseudolabeled_indxs]
    #update num
    unlabeled_num=len(unlabeled_indxs)
    #pseudolabeled_num=len(pseudolabeled_indxs)

    #x_train,y_train=update_with_new_pseudo(model,x_train,y_train,new_pseudo,listaT1,listaFLAIR,listaMASK)

    #update_data_folder(model,new_pos_in_features,listaT1,listaFLAIR,listaMASK,datafolder=datafolder)
    update_data_folder(model,new_pos_in_features,listaT1,listaFLAIR,listaMASK,datafolder=datafolder,regularized=True)

    print('training with new data...')
    """
    result=model.fit_generator(data_augmentation.DA_2Ddegradation(x_train,y_train),#data_augmentation.MixUp_Generator(x_train,y_train,0.3),DA_2Ddegradation, #model.fit(x_train, y_train,batch_size=1
                    steps_per_epoch=x_train.shape[0],
                    epochs=Epoch_per_step,
                    validation_data=(x_val, y_val),
                    callbacks=[savemodel])
    """
    numb_data= len(sorted(glob.glob(datafolder+"x*.npy")))
    result=model.fit_generator(data_gen_iqda_2it(datafolder=datafolder,sim='DICE'),#data_gen(), data_gen_iqda
                steps_per_epoch=numb_data,
                epochs=Epoch_per_step)#,  validation_data=(x_val, y_val))
    model.save_weights('weights/data_gen_iqda_2it_volbrain_TSNE3_bottleneckRegulirized_loss1[1_02]_Kclosest_'+str(step)+'.h5')
