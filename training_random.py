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
increment_new_data=10
#filepath="One_Tile_2mods_one_100ep.h5"
#filepath="../networks/One_Tile_96_2mods.h5"
#model=modelos.load_UNET3D_SLANT27_v2_groupNorm(96,96,96,2,1,24,0.5,final_act='sigmoid')
model=modelos.load_UNET3D_SLANT27_v2_groupNorm(96,96,96,2,2,24,0.5)
#model.compile(optimizer=optimizers.Adam(0.0001), loss='mse')
model.compile(optimizer=optimizers.Adam(0.0001), loss=losses.GJLsmooth, metrics=[metrics.mdice])
savemodel=ModelCheckpoint('_random_curruc.h5', monitor='val_mdice', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)

fun=get_bottleneck_features_func(model)
"""
listaT1 = sorted(glob.glob("../lib/volbrain_qc/n_mfmni*t1*.nii*"))
listaFLAIR = sorted(glob.glob("../lib/volbrain_qc/n_mfmni*flair*.nii*"))
listaMASK = sorted(glob.glob("../lib/volbrain_qc/mask*.nii*"))
listaMASK=np.array(listaMASK)
"""
listaT1 = sorted(glob.glob("../lib/ISBI_preprocess/test*mprage*.nii*"))
listaFLAIR = sorted(glob.glob("../lib/ISBI_preprocess/test*flair*.nii*"))

listaT1=np.array(listaT1)
listaFLAIR=np.array(listaFLAIR)

listaT1_isbi = sorted(glob.glob("../lib/isbi_final_train_preprocessed/training*mprage*.nii*"))
listaFLAIR_isbi = sorted(glob.glob("../lib/isbi_final_train_preprocessed/training*flair*.nii*"))

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

filepath="../networks/One_Tile_96_2mods.h5"
model.load_weights(filepath)

while(unlabeled_num>increment_new_data):
    step=step+1
    print('step: '+str(step))
    print('loading new data...')

    new_pseudo = np.array(unlabeled_indxs)
    np.random.shuffle(new_pseudo)
    new_pseudo =new_pseudo[:increment_new_data]
    new_pseudo=new_pseudo.tolist()
    #update indexes
    pseudolabeled_indxs= pseudolabeled_indxs+ new_pseudo
    unlabeled_indxs =  [x for x in unlabeled_indxs if x not in pseudolabeled_indxs]

    x_train__,y_train__=update_with_new_pseudo(model,x_train,y_train,pseudolabeled_indxs,listaT1,listaFLAIR)
    print('training with new data...')
    result=model.fit_generator(data_augmentation.DA_2Ddegradation(x_train__,y_train__), #model.fit(x_train, y_train,batch_size=1
                    steps_per_epoch=x_train__.shape[0],
                    epochs=Epoch_per_step,
                    validation_data=(x_val, y_val),
                    callbacks=[savemodel])
    model.save_weights('step_random_'+str(step)+'.h5')
