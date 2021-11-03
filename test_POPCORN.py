import modelos, os, glob, losses, metrics,gc
from utils import seg_majvote,load_modalities,load_flair,seg_majvote_flair
import nibabel as nii
from keras import optimizers
import numpy as np
from keras import backend as K


os.environ["CUDA_VISIBLE_DEVICES"]='0'
img_path_msseg='../lib/msseg/'
img_path_mso='../lib/MS_O/'
img_path_isbitest='../lib/ISBI_preprocess/'
#WEIGHTS=''
#pred_path=''
#regularized=False
loss_weights=[1,0.001]


listaFLAIR_msseg = sorted(glob.glob(img_path_msseg+"*flair*.nii*"))
listaFLAIR_isbitest = sorted(glob.glob(img_path_isbitest+"*flair*.nii*"))
listaFLAIR_mso = sorted(glob.glob(img_path_mso+"*flair*.nii*"))


def seg_to_folder_with_Weightlist(listaWeights,model,listaFLAIR,extension_name=''):
    for WEIGHTS in listaWeights:
        weight_folder_name= WEIGHTS.split('/')[-1].split('.h5')[0]
        pred_path='SEG/'+weight_folder_name+extension_name+'/'
        try:
            os.mkdir(pred_path)
        except OSError:
            print ("Creation of the directory %s failed" % pred_path)
        else:
            print ("Successfully created the directory %s " % pred_path)
        seg_to_folder(pred_path,model,WEIGHTS,listaFLAIR)

def seg_to_folder(pred_path,regularized,WEIGHTS,listaFLAIR):
    if(regularized):
        model = modelos.load_UNET3D_bottleneck_regularized(64,64,64,1,2,24,0.5,groups=8)
    else:
        model=modelos.load_UNET3D_SLANT27_v2_groupNorm(64,64,64,1,2,24,0.5)
    model.load_weights(WEIGHTS)
    if(regularized):
        model.compile(optimizer=optimizers.Adam(0.0001), loss=[losses.GJLsmooth,losses.BottleneckRegularized],loss_weights=loss_weights)
    else:
        model.compile(optimizer=optimizers.Adam(0.0001), loss=losses.GJLsmooth, metrics=[metrics.mdice])


    print('Total images to process: '+str(len(listaFLAIR)))

    for FLAIR_name in listaFLAIR:
        print('processing:')
        print(FLAIR_name)
        FLAIR=load_flair(FLAIR_name)
        SEG_mask=seg_majvote_flair(FLAIR,model,nbNN=[5,5,5],ps=[64,64,64],regularized=regularized)
        name=pred_path+FLAIR_name.split('/')[-1].split('.nii')[0]+'_SEG.nii.gz'
        T1_img = nii.load(FLAIR_name)
        img = nii.Nifti1Image(SEG_mask.astype(np.uint8), T1_img.affine )
        img.to_filename(name)
    K.clear_session()
    gc.collect() #free memory

#tsne1=sorted(glob.glob("weights/tsne1/*.h5"))
#random_noreg=sorted(glob.glob("weights/data_gen_iqda_volbrain_TSNE3_random_without_reg/*.h5"))
#random=sorted(glob.glob("weights/flair64_noreg_random/*.h5"))
nearest=sorted(glob.glob("weights/test_/*.h5"))
#nearest_noreg=sorted(glob.glob("weights/data_gen_idqa_TSNE3_nearest_without_reg/*.h5"))
#all_noreg=sorted(glob.glob("weights/noreg_volbrain_all_increase/*.h5"))
#all=sorted(glob.glob("weights/flair64_segmentedbyt0_Kclosest_reg/*.h5"))
print(nearest)
#seg_to_folder_with_Weightlist(nearest,True,listaFLAIR_mso,'mso')
#seg_to_folder_with_Weightlist(nearest,True,listaFLAIR_msseg,'msseg')
seg_to_folder_with_Weightlist(nearest,True,listaFLAIR_isbitest,'isbi_test')
"""
seg_to_folder_with_Weightlist(nearest,True,listaFLAIR,'msseg')
seg_to_folder_with_Weightlist(nearest,True,listaFLAIR_mso,'mso')
seg_to_folder_with_Weightlist(all,True,listaT1,listaFLAIR)
seg_to_folder_with_Weightlist(tsne1,True,listaT1,listaFLAIR)
seg_to_folder_with_Weightlist(random_noreg,False,listaT1,listaFLAIR)
seg_to_folder_with_Weightlist(nearest_noreg,False,listaT1,listaFLAIR)
seg_to_folder_with_Weightlist(all_noreg,False,listaT1,listaFLAIR)
"""
