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
Epoch=100
datafolder='data/'
datafolder_ssl='ssl_data/'

model=modelos.load_UNET3D_SLANT27_v2_groupNorm(ps[0],ps[1],ps[2],1,2,24,0.5)
model.summary()
model.compile(optimizer=optimizers.Adam(0.0001), loss=losses.consistency_reg )

result=model.fit_generator(data_gen_consistency_reg(datafolder,datafolder_ssl),
                steps_per_epoch=818,
                epochs=Epoch)

model.save_weights('weights/consistency_reg_rot.h5')
