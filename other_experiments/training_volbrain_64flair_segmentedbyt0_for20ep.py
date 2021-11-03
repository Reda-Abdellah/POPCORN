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
#nbNN=[5,5,5]
ps=[64,64,64]
dataset_path="/data1/rkamraoui/DeepvolBrain/Segmentation/DeepLesionBrain/lib"
Epochs=20
increment_new_data=200
#datafolder_tmp='data_temp/'
#datafolder='data_at_t0/'
#datafolder='data_nearest/'
datafolder='data_nearest_recompute/'
resume=False
resume_after_adding_pseudo_of_step=1
load_precomputed_features=False
load_labeled_dataset=False
unlabeled_dataset="volbrain"
train_by_loading_alldata_to_RAM=False
regularized_loss='loss3'
loss_weights=[1,0.01]


in_filepath="One_2mods_2it02same_loss3_1_001_64_flair_only_64_ISBI_gen_IQDA_.h5"
#out_filepath= lambda x: 'weights/flair64_volbrain_bottleneckRegulirized_'+regularized_loss+'__'+str(loss_weights[0])+'_'+str(loss_weights[1])+'_allfor20ep_'+"%02d" % (x)+'.h5'
#out_filepath= lambda x: 'weights/flair64_volbrain_bottleneckRegulirized_'+regularized_loss+'__'+str(loss_weights[0])+'_'+str(loss_weights[1])+'_nearest_for20ep_'+"%02d" % (x)+'.h5'
out_filepath= lambda x: 'weights/flair64_volbrain_bottleneckRegulirized_'+regularized_loss+'__'+str(loss_weights[0])+'_'+str(loss_weights[1])+'_nearest_recompute_for20ep_'+"%02d" % (x)+'.h5'

model = modelos.load_UNET3D_bottleneck_regularized(ps[0],ps[1],ps[2],1,2,24,0.5,groups=8)
model.compile(optimizer=optimizers.Adam(0.0001), loss=[losses.mdice_loss,losses.BottleneckRegularized],loss_weights=loss_weights)
fun = K.function([model.input, K.learning_phase()],[model.output[0]])


if(unlabeled_dataset=="volbrain"):
    #listaT1 = sorted(glob.glob(dataset_path+"/volbrain_qc/n_mfmni*t1*.nii*"))
    listaFLAIR = sorted(glob.glob(dataset_path+"/volbrain_qc/n_mfmni*flair*.nii*"))
    listaMASK = sorted(glob.glob(dataset_path+"/volbrain_qc/mask*.nii*"))
    listaMASK = np.array(listaMASK)
elif(unlabeled_dataset=="isbi_test"):
    #listaT1 = sorted(glob.glob(dataset_path+"/ISBI_preprocess/test*mprage*.nii*"))
    listaFLAIR = sorted(glob.glob(dataset_path+"/ISBI_preprocess/test*flair*.nii*"))

#listaT1 =listaT1[:5]
#listaFLAIR =listaFLAIR[:5]

#listaT1=np.array(listaT1)
listaFLAIR=np.array(listaFLAIR)

#indexing labeled data
#lib_path_1 = os.path.join(dataset_path,"MS_O")
#lib_path_2 = os.path.join(dataset_path,"msseg")
lib_path_3 = os.path.join(dataset_path,"isbi_final_train_preprocessed")


listaT1_3=keyword_toList(path=lib_path_3,keyword="mprage")
listaFLAIR_3=keyword_toList(path=lib_path_3,keyword="flair")
listaSEG1_3=keyword_toList(path=lib_path_3,keyword="mask1")
listaSEG2_3=keyword_toList(path=lib_path_3,keyword="mask2")

#listaT1_labeled= np.array(listaT1_1+listaT1_3)
#listaFLAIR_labeled= np.array(listaFLAIR_1+listaFLAIR_3)

listaFLAIR_labeled= np.array(listaFLAIR_3)

unlabeled_indxs= range(len(listaFLAIR))
pseudolabeled_indxs=[]
unlabeled_num=len(unlabeled_indxs)
pseudolabeled_num=len(pseudolabeled_indxs)
labeled_num=len(listaFLAIR_labeled)

if(load_labeled_dataset):
    update_labeled_folder_flair(listaFLAIR_3,listaSEG1_3,listaMASK=None,datafolder=datafolder,numbernotnullpatch=15)
    update_labeled_folder_flair(listaFLAIR_3,listaSEG2_3,listaMASK=None,datafolder=datafolder,numbernotnullpatch=15)

step=0
model.load_weights(in_filepath)


train_files_bytiles=[]
for i in range(27):
    train_files_bytiles.append(keyword_toList(datafolder,"x*tile_"+str(i)+".npy") )


numb_data= len(sorted(glob.glob(datafolder+"x*.npy")))
result=model.fit_generator(data_gen_iqda_2it(datafolder,train_files_bytiles,sim='output_diff'), #loss1->sim='DICE' loss3->sim='output_diff' loss4->sim='input_diff'
            steps_per_epoch=numb_data,
            epochs=Epochs)

model.save_weights(out_filepath(0))
