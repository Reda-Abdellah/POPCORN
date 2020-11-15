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
from keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"]='1'
Rootpath=os.getcwd()
nbNN=[5,5,5]
dataset_path="/data1/rkamraoui/DeepvolBrain/Segmentation/DeepLesionBrain/lib"
ps=[96,96,96]
Epoch_per_step=2
increment_new_data=100
datafolder='data_fares/'
resume=False
resume_after_adding_pseudo_of_step=10
load_precomputed_features=False
load_labeled_dataset=False
unlabeled_dataset="volbrain"
#unlabeled_dataset="isbi_test"
regularized=True
train_by_loading_alldata_to_RAM=False
regularized_loss='loss3'
loss_weights=[1,100]
#in_filepath="One_Tile_96_2mods.h5"
#filepath="One_2mods_96_MSO_andISBI_gen_IQDA.h5"
in_filepath="One_2mods_2it033same_loss1[1_02]_96_MSO_andISBI_gen_IQDA.h5"
#out_filepath= lambda x: 'weights/data_gen_iqda_2it_volbrain_TSNE3_bottleneckRegulirized_'+regularized_loss+'__'+str(loss_weights[0])+'_'+str(loss_weights[1])+'__Kfarest_'+"%02d" % (x)+'.h5'
out_filepath= lambda x: 'weights/data_gen_iqda_volbrain_TSNE3_Kfarest_'+"%02d" % (x)+'.h5'


#model=modelos.load_UNET3D_SLANT27_v2_groupNorm(96,96,96,2,2,24,0.5)
#model.load_weights(filepath)
#model.compile(optimizer=optimizers.Adam(0.0001), loss='mse')
#model.compile(optimizer=optimizers.Adam(0.0001), loss=losses.norm_mse)
if(regularized):
    model = modelos.load_UNET3D_bottleneck_regularized(ps[0],ps[1],ps[2],2,2,20,0.5,groups=4)
    model.compile(optimizer=optimizers.Adam(0.0001), loss=[losses.GJLsmooth,losses.BottleneckRegularized],loss_weights=loss_weights)
    fun = K.function([model.input, K.learning_phase()],[model.output[0]])
else:
    model=modelos.load_UNET3D_SLANT27_v2_groupNorm(96,96,96,2,2,24,0.5)
    model.compile(optimizer=optimizers.Adam(0.0001), loss=losses.GJLsmooth, metrics=[metrics.mdice])
    #savemodel=ModelCheckpoint('curruc.h5', monitor='val_mdice', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)
    fun = get_bottleneck_features_func(model)

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
lib_path_1 = os.path.join(dataset_path,"MS_O")
lib_path_2 = os.path.join(dataset_path,"msseg")
lib_path_3 = os.path.join(dataset_path,"isbi_final_train_preprocessed")


listaT1_1=keyword_toList(path=lib_path_1,keyword="t1")
listaFLAIR_1=keyword_toList(path=lib_path_1,keyword="flair")
#listaSEG_1=keyword_toList(path=lib_path_1,keyword="lesion")
listaMASK_1=keyword_toList(path=lib_path_1,keyword="mask")

listaT1_2=keyword_toList(path=lib_path_2,keyword="t1")
listaFLAIR_2=keyword_toList(path=lib_path_2,keyword="flair")
listaSEG_2=keyword_toList(path=lib_path_2,keyword="mask1")

listaT1_3=keyword_toList(path=lib_path_3,keyword="mprage")
listaFLAIR_3=keyword_toList(path=lib_path_3,keyword="flair")
listaSEG1_3=keyword_toList(path=lib_path_3,keyword="mask1")
listaSEG2_3=keyword_toList(path=lib_path_3,keyword="mask2")

#listaT1_labeled= np.array(listaT1_1+listaT1_3)
#listaFLAIR_labeled= np.array(listaFLAIR_1+listaFLAIR_3)

listaT1_labeled= np.array(listaT1_3)
listaFLAIR_labeled= np.array(listaFLAIR_3)

unlabeled_indxs= range(len(listaT1))
pseudolabeled_indxs=[]
unlabeled_num=len(unlabeled_indxs)
pseudolabeled_num=len(pseudolabeled_indxs)
labeled_num=len(listaT1_labeled)

if(load_labeled_dataset):
    classic_loading=False
    if(classic_loading):
        x_train,y_train,x_val,y_val=load_isbi(one_out=False)
    elif(not resume):
        #add data to datafolder
        #update_labeled_folder(listaT1_1,listaFLAIR_1,listaMASK_1,listaMASK=None,datafolder=datafolder,numbernotnullpatch=10)
        #update_labeled_folder(listaT1_2,listaFLAIR_2,listaSEG_2,listaMASK=None,datafolder=datafolder_val,numbernotnullpatch=10)
        update_labeled_folder(listaT1_3,listaFLAIR_3,listaSEG1_3,listaMASK=None,datafolder=datafolder,numbernotnullpatch=15)
        update_labeled_folder(listaT1_3,listaFLAIR_3,listaSEG2_3,listaMASK=None,datafolder=datafolder,numbernotnullpatch=15)



step=0
model.load_weights(in_filepath)


if(load_precomputed_features):
    print('loading precomputed features...')
    #rank_distance=np.load('rank_distance_volbrain_tsne3.npy')
    rank_distance=np.load('rank_distance_volbrain_tsne3_regularized.npy')
else:
    print('computing bottleneck_features...')
    bottleneck_features_labeled,file_names= features_from_names(listaT1_labeled,listaFLAIR_labeled,fun)
    bottleneck_features_unlabeled,file_names= features_from_names(listaT1[unlabeled_indxs],listaFLAIR[unlabeled_indxs],fun,listaMASK[unlabeled_indxs])
    np.save('bottleneck_features_labeled_withreg.npy',bottleneck_features_labeled)
    np.save('bottleneck_features_unlabeled_withreg.npy',bottleneck_features_unlabeled)
    #bottleneck_features_unlabeled,file_names= features_from_names(listaT1[unlabeled_indxs],listaFLAIR[unlabeled_indxs],fun)
    #bottleneck_features_labeled=np.load('bottleneck_features_labeled.npy')
    #bottleneck_features_unlabeled=np.load('bottleneck_features_unlabeled_volbrain.npy')
    #rank_distance= pca_rank(bottleneck_features_unlabeled,bottleneck_features_labeled,n_components=16)
    #np.save('rank_distance_volbrain_pca16.npy',rank_distance)
    #print(rank_distance.shape)
    print('computing projection to simplar dataplane...')
    rank_distance= tsne_rank(bottleneck_features_unlabeled,bottleneck_features_labeled,n_components=3)
    ###np.save('rank_distance_volbrain_tsne3.npy',rank_distance)
    np.save('rank_distance_volbrain_tsne3_regularized_far.npy',rank_distance)

if(resume):
    for it in range(resume_after_adding_pseudo_of_step):
        print('resuming training...')
        new_pos_in_features = give_dist_for_Kfarest(rank_distance,n_indxs=increment_new_data,k=5)
        not_new_pos_in_features = [x for x in range(unlabeled_num) if x not in new_pos_in_features]
        #update indexes
        pseudolabeled_indxs= pseudolabeled_indxs+ new_pos_in_features
        unlabeled_indxs =  [x for x in unlabeled_indxs if x not in pseudolabeled_indxs]
        #update num
        unlabeled_num=len(unlabeled_indxs)
    step=resume_after_adding_pseudo_of_step-1
    if(not step==0):
        model.load_weights(out_filepath(step))

#labeld_to_data(x_train,y_train,datafolder=datafolder)

#Training
while(unlabeled_num>increment_new_data):
    step=step+1
    print('step: '+str(step))
    print('loading new data...')
    if( resume and step==resume_after_adding_pseudo_of_step):
        print('resuming..')
    else:
        #new_pos_in_features = give_n_closest(rank_distance,n_indxs=increment_new_data)
        new_pos_in_features = give_dist_for_Kfarest(rank_distance,n_indxs=increment_new_data,k=5)
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
        update_data_folder(model,new_pos_in_features,listaT1,listaFLAIR,listaMASK,datafolder=datafolder,regularized=regularized)

    train_files_bytiles=[]
    for i in range(27):
    	train_files_bytiles.append(keyword_toList(datafolder,"tile_"+str(i)+".npy") )

    print('training with new data...')
    if(train_by_loading_alldata_to_RAM):
        result=model.fit_generator(data_augmentation.DA_2Ddegradation(x_train,y_train),#data_augmentation.MixUp_Generator(x_train,y_train,0.3),DA_2Ddegradation, #model.fit(x_train, y_train,batch_size=1
                    steps_per_epoch=x_train.shape[0],
                    epochs=Epoch_per_step,
                    validation_data=(x_val, y_val),
                    callbacks=[savemodel])
    else:
        numb_data= len(sorted(glob.glob(datafolder+"x*.npy")))
        if(regularized):
            result=model.fit_generator(data_gen_iqda_2it(datafolder,train_files_bytiles,sim='output_diff'), #loss1->sim='DICE' loss3->sim='output_diff' loss4->sim='input_diff'
                steps_per_epoch=numb_data,
                epochs=Epoch_per_step)
        else:
            result=model.fit_generator(data_gen_iqda(datafolder=datafolder),#data_gen(), data_gen_iqda
                steps_per_epoch=numb_data,
                epochs=Epoch_per_step)

    model.save_weights(out_filepath(step))
