import modelos, os, glob, losses, metrics,gc
from utils import seg_majvote,load_modalities
import nibabel as nii
from keras import optimizers
import numpy as np
from keras import backend as K


os.environ["CUDA_VISIBLE_DEVICES"]='1'
#img_path='../lib/msseg/'
img_path='../lib/MS_O/'
#WEIGHTS=''
#pred_path=''
#regularized=False
loss_weights=[1,100]

listaT1 = sorted(glob.glob(img_path+"*t1*.nii*"))
listaFLAIR = sorted(glob.glob(img_path+"*flair*.nii*"))

#print(listaT1)


def seg_to_folder_with_Weightlist(listaWeights,model,listaT1,listaFLAIR):
    for WEIGHTS in listaWeights:
        weight_folder_name= WEIGHTS.split('/')[-1].split('.h5')[0]
        pred_path='SEG/'+weight_folder_name+'/'
        try:
            os.mkdir(pred_path)
        except OSError:
            print ("Creation of the directory %s failed" % pred_path)
        else:
            print ("Successfully created the directory %s " % pred_path)
        seg_to_folder(pred_path,model,WEIGHTS,listaT1,listaFLAIR)

def seg_to_folder(pred_path,regularized,WEIGHTS,listaT1,listaFLAIR):
    if(regularized):
        model = modelos.load_UNET3D_bottleneck_regularized(96,96,96,2,2,20,0.5,groups=4)
    else:
        model=modelos.load_UNET3D_SLANT27_v2_groupNorm(96,96,96,2,2,24,0.5)
    model.load_weights(WEIGHTS)
    if(regularized):
        model.compile(optimizer=optimizers.Adam(0.0001), loss=[losses.GJLsmooth,losses.BottleneckRegularized],loss_weights=loss_weights)
    else:
        model.compile(optimizer=optimizers.Adam(0.0001), loss=losses.GJLsmooth, metrics=[metrics.mdice])


    print('Total images to process: '+str(len(listaT1)))

    for T1_name,FLAIR_name in zip(listaT1,listaFLAIR):
        print('processing:')
        print(T1_name)
        print(FLAIR_name)
        T1,FLAIR=load_modalities(T1_name,FLAIR_name)
        SEG_mask=seg_majvote(T1,FLAIR,model,nbNN=[5,5,5],ps=[96,96,96],regularized=regularized)
        name=pred_path+T1_name.split('/')[-1].split('.nii')[0]+'_SEG.nii.gz'
        T1_img = nii.load(T1_name)
        img = nii.Nifti1Image(SEG_mask.astype(np.uint8), T1_img.affine )
        img.to_filename(name)
    K.clear_session()
    gc.collect() #free memory

tsne1=sorted(glob.glob("weights/tsne1/*.h5"))
random_noreg=sorted(glob.glob("weights/data_gen_iqda_volbrain_TSNE3_random_without_reg/*.h5"))
random=sorted(glob.glob("weights/data_gen_iqda_2it_volbrain_TSNE3_bottleneckRegulirized_loss3__1_100__random/*.h5"))
nearest=sorted(glob.glob("weights/data_gen_iqda_2it_volbrain_TSNE3_bottleneckRegulirized_loss3__1_100__Kclosest_v3/*.h5"))
nearest_noreg=sorted(glob.glob("weights/data_gen_idqa_TSNE3_nearest_without_reg/*.h5"))
all_noreg=sorted(glob.glob("weights/noreg_volbrain_all_increase/*.h5"))
all=sorted(glob.glob("weights/all_at_once_increment/*.h5"))

"""
seg_to_folder_with_Weightlist(random,True,listaT1,listaFLAIR)
seg_to_folder_with_Weightlist(nearest,True,listaT1,listaFLAIR)
seg_to_folder_with_Weightlist(all,True,listaT1,listaFLAIR)
seg_to_folder_with_Weightlist(tsne1,True,listaT1,listaFLAIR)
seg_to_folder_with_Weightlist(random_noreg,False,listaT1,listaFLAIR)
seg_to_folder_with_Weightlist(nearest_noreg,False,listaT1,listaFLAIR)
"""
seg_to_folder_with_Weightlist(all_noreg,False,listaT1,listaFLAIR)
