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

os.environ["CUDA_VISIBLE_DEVICES"]='1'
Rootpath=os.getcwd()
dataset_path="/data1/rkamraoui/DeepvolBrain/Segmentation/DeepLesionBrain/lib"
nbNN=[5,5,5]
ps=[96,96,96]
Epoch_per_step=5
increment_new_data=10
#filepath="One_Tile_2mods_one_100ep.h5"
#filepath="../networks/One_Tile_96_2mods.h5"
#model=modelos.load_UNET3D_SLANT27_v2_groupNorm(96,96,96,2,1,24,0.5,final_act='sigmoid')
model=modelos.load_UNET3D_SLANT27_v2_groupNorm(96,96,96,2,2,24,0.5)
#model.load_weights(filepath)
#model.compile(optimizer=optimizers.Adam(0.0001), loss='mse')
model.compile(optimizer=optimizers.Adam(0.0001), loss=losses.GJLsmooth, metrics=[metrics.mdice])
savemodel=ModelCheckpoint('curruc_far.h5', monitor='val_mdice', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)

fun=get_bottleneck_features_func(model)
if(unlabeled_dataset=="volbrain"):
    listaT1 = sorted(glob.glob(dataset_path+"/volbrain_qc/n_mfmni*t1*.nii*"))
    listaFLAIR = sorted(glob.glob(dataset_path+"/volbrain_qc/n_mfmni*flair*.nii*"))
    listaMASK = sorted(glob.glob(dataset_path+"/volbrain_qc/mask*.nii*"))
    listaMASK = np.array(listaMASK)
elif(unlabeled_dataset=="isbi_test"):
    listaT1 = sorted(glob.glob(dataset_path+"/ISBI_preprocess/test*mprage*.nii*"))
    listaFLAIR = sorted(glob.glob(dataset_path+"/ISBI_preprocess/test*flair*.nii*"))

#listaT1 =listaT1[:5]
#listaFLAIR =listaFLAIR[:5]

listaT1=np.array(listaT1)
listaFLAIR=np.array(listaFLAIR)

#indexing labeled data
lib_path_1 = os.path.join(dataset_path,"lib","MS_O")
lib_path_2 = os.path.join(dataset_path,"lib","msseg")
lib_path_3 = os.path.join(dataset_path,"lib","isbi_final_train_preprocessed")


listaT1_isbi=np.array(listaT1_isbi)
listaFLAIR_isbi=np.array(listaFLAIR_isbi)

x_train,y_train,x_val,y_val=load_isbi(one_out=False)

#indexs are used to index from listaT1
pseudolabeled_indxs=[]
unlabeled_indxs= range(len(listaT1))
unlabeled_num=len(unlabeled_indxs)
pseudolabeled_num=len(pseudolabeled_indxs)
labeled_num=len(listaT1_isbi)


step=0
"""
result=model.fit_generator(data_augmentation.rot90_Generator_2mods(x_train,y_train), #model.fit(x_train, y_train,batch_size=1
                steps_per_epoch=x_train.shape[0],
                epochs=10,
                validation_data=(x_val, y_val),
                callbacks=[savemodel])

model.save_weights('step_'+str(step)+'.h5')
"""
filepath="../networks/One_Tile_96_2mods.h5"
model.load_weights(filepath)

while(unlabeled_num>increment_new_data):
    step=step+1
    print('step: '+str(step))
    print('loading new data...')
    bottleneck_features_labeled,file_names= features_from_names(listaT1_isbi,listaFLAIR_isbi,fun)
    #bottleneck_features_unlabeled,file_names= features_from_names(listaT1[unlabeled_indxs],listaFLAIR[unlabeled_indxs],fun,listaMASK[unlabeled_indxs])
    bottleneck_features_unlabeled,file_names= features_from_names(listaT1[unlabeled_indxs],listaFLAIR[unlabeled_indxs],fun)
    if(pseudolabeled_num>0):
        #bottleneck_features_pseudolabeled,file_names= features_from_names(listaT1[pseudolabeled_indxs],listaFLAIR[pseudolabeled_indxs],fun,listaMASK[pseudolabeled_indxs])
        bottleneck_features_pseudolabeled,file_names= features_from_names(listaT1[pseudolabeled_indxs],listaFLAIR[pseudolabeled_indxs],fun)
        #rank_distance= brute_force_rank(bottleneck_features_unlabeled, np.concatenate((bottleneck_features_labeled,bottleneck_features_pseudolabeled),axis=0))
        rank_distance= pca_rank(bottleneck_features_unlabeled, np.concatenate((bottleneck_features_labeled,bottleneck_features_pseudolabeled),axis=0))
    else:
        #rank_distance= brute_force_rank(bottleneck_features_unlabeled,bottleneck_features_labeled)
        rank_distance= pca_rank(bottleneck_features_unlabeled,bottleneck_features_labeled)
    #new_pos_in_features = give_n_closest(rank_distance,n_indxs=increment_new_data)
    new_pos_in_features = give_n_farest(rank_distance,n_indxs=increment_new_data)
    not_new_pos_in_features = [x for x in range(unlabeled_num) if x not in new_pos_in_features]
    new_pseudo=np.array(unlabeled_indxs)[new_pos_in_features]
    new_pseudo=new_pseudo.tolist()
    #update indexes
    pseudolabeled_indxs= pseudolabeled_indxs+ new_pseudo
    unlabeled_indxs =  [x for x in unlabeled_indxs if x not in pseudolabeled_indxs]

    if(pseudolabeled_num>0):
        reducer= reducer_umap(  get_x([bottleneck_features_labeled,bottleneck_features_pseudolabeled,bottleneck_features_unlabeled[new_pos_in_features ],bottleneck_features_unlabeled[not_new_pos_in_features]])  )
    else:
        reducer= reducer_umap(  get_x([bottleneck_features_labeled,bottleneck_features_unlabeled[new_pos_in_features ],bottleneck_features_unlabeled[not_new_pos_in_features]])  )
    #update num
    unlabeled_num=len(unlabeled_indxs)
    pseudolabeled_num=len(pseudolabeled_indxs)
    print('saving plot...')
    save_plot(reducer,plotname='plot_step'+str(step)+'.eps',labeled_num=labeled_num,pseudolabeled_num=pseudolabeled_num,unlabeled_num=unlabeled_num)
    print('updating new data...')
    #x_train,y_train=update_with_new_pseudo(model,x_train,y_train,new_pseudo,listaT1,listaFLAIR,listaMASK)
    #x_train,y_train=update_with_new_pseudo(model,x_train,y_train,new_pseudo,listaT1,listaFLAIR)
    x_train__,y_train__=update_with_new_pseudo(model,x_train,y_train,pseudolabeled_indxs,listaT1,listaFLAIR)
    print('training with new data...')
    result=model.fit_generator(data_augmentation.DA_2Ddegradation(x_train__,y_train__),#data_augmentation.MixUp_Generator(x_train,y_train,0.3),DA_2Ddegradation, #model.fit(x_train, y_train,batch_size=1
                    steps_per_epoch=x_train__.shape[0],
                    epochs=Epoch_per_step,
                    validation_data=(x_val, y_val),
                    callbacks=[savemodel])
    model.save_weights('step_IQDA_farest_'+str(step)+'.h5')
