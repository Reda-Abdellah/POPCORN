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

os.environ["CUDA_VISIBLE_DEVICES"]='0'
Rootpath=os.getcwd()
#nbNN=[5,5,5]
ps=[64,64,64]
Epoch=100
datafolder='data/'
datafolder_ssl='ssl_data/'

in_filepath="One_noreg_64_flair_only_64_ISBI_gen_IQDA_.h5"
out_filepath= lambda x: 'weights/flair64_volbrain_noreg_'+regularized_loss+'__'+str(loss_weights[0])+'_'+str(loss_weights[1])+'__random_'+"%02d" % (x)+'.h5'

model=modelos.load_UNET3D_SLANT27_v2_groupNorm(ps[0],ps[1],ps[2],1,2,24,0.5)
model.compile(optimizer=optimizers.Adam(0.0001), loss=losses.uncertainty_pseudo_lab)
list_x_ssl=sorted(glob.glob(datafolder_ssl+"x*.npy"))


result=model.fit_generator(data_gen_uncertainty_pseudolab(datafolder,datafolder_ssl),
                steps_per_epoch=818,
                epochs=Epoch)

model.save_weights('weights/uncertainty_pseudo_lab.h5')
