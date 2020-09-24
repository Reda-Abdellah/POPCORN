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
nbNN=[5,5,5]
ps=[96,96,96]
Epoch_per_step=2
increment_new_data=100
datafolder='data2/'
#filepath="One_Tile_2mods_one_100ep.h5"
#filepath="../networks/One_Tile_96_2mods.h5"
#model=modelos.load_UNET3D_SLANT27_v2_groupNorm(96,96,96,2,1,24,0.5,final_act='sigmoid')
model=modelos.load_UNET3D_SLANT27_v2_groupNorm(96,96,96,2,2,24,0.5)
#model.load_weights(filepath)
#model.compile(optimizer=optimizers.Adam(0.0001), loss='mse')
model.compile(optimizer=optimizers.Adam(0.0001), loss=losses.GJLsmooth, metrics=[metrics.mdice])
savemodel=ModelCheckpoint('curruc.h5', monitor='val_mdice', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)

fun=get_bottleneck_features_func(model)
listaT1 = sorted(glob.glob("../lib/volbrain_qc/n_mfmni*t1*.nii*"))
listaFLAIR = sorted(glob.glob("../lib/volbrain_qc/n_mfmni*flair*.nii*"))
listaMASK = sorted(glob.glob("../lib/volbrain_qc/mask*.nii*"))
listaMASK=np.array(listaMASK)

#listaT1 =listaT1[:5]
#listaFLAIR =listaFLAIR[:5]
"""
"""
listaT1_isbitest = sorted(glob.glob("../lib/ISBI_preprocess/test*mprage*.nii*"))
listaFLAIR_isbitest = sorted(glob.glob("../lib/ISBI_preprocess/test*flair*.nii*"))

listaT1=np.array(listaT1)
listaFLAIR=np.array(listaFLAIR)
listaFLAIR_isbitest =np.array(listaFLAIR_isbitest )
listaT1_isbitest =np.array(listaT1_isbitest )

listaT1_isbi = sorted(glob.glob("../lib/isbi_final_train_preprocessed/training*mprage*.nii*"))
listaFLAIR_isbi = sorted(glob.glob("../lib/isbi_final_train_preprocessed/training*flair*.nii*"))

listaT1_isbi=np.array(listaT1_isbi)
listaFLAIR_isbi=np.array(listaFLAIR_isbi)

x_train,y_train,x_val,y_val=load_isbi(one_out=False)

labeld_to_data(x_train,y_train,datafolder=datafolder)

#indexs are used to index from listaT1
pseudolabeled_indxs=[]
unlabeled_indxs= range(len(listaT1))
unlabeled_indxs_isbitest= range(len(listaT1_isbitest))
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
filepath="One_Tile_96_2mods.h5"
model.load_weights(filepath)

bottleneck_features_labeled,file_names= features_from_names(listaT1_isbi,listaFLAIR_isbi,fun)
np.save('bottleneck_features_labeled.npy',bottleneck_features_labeled)
bottleneck_features_unlabeled_volbrain,file_names= features_from_names(listaT1[unlabeled_indxs],listaFLAIR[unlabeled_indxs],fun,listaMASK[unlabeled_indxs])
np.save('bottleneck_features_unlabeled_volbrain.npy',bottleneck_features_unlabeled_volbrain)
bottleneck_features_unlabeled_isbitest,file_names= features_from_names(listaT1_isbitest[unlabeled_indxs_isbitest],listaFLAIR_isbitest[unlabeled_indxs_isbitest],fun)
np.save('bottleneck_features_unlabeled_isbitest.npy',bottleneck_features_unlabeled_isbitest)
#rank_distance= pca_rank(bottleneck_features_unlabeled,bottleneck_features_labeled,n_components=2)
#rank_distance= tsne_rank(bottleneck_features_unlabeled,bottleneck_features_labeled,n_components=3)
#np.save('rank_distance_volbrain_tsne3.npy',rank_distance)

##rank_distance=np.load('rank_distance_volbrain.npy')
#rank_distance=np.load('rank_distance_volbrain_tsne3.npy')
#new_pos_in_features = give_n_closest(rank_distance,n_indxs=increment_new_data)
