import modelos, os, glob, losses, metrics,gc
from utils import seg_majvote,load_modalities,load_flair,seg_majvote_flair_ssl
import nibabel as nii
from keras import optimizers
import numpy as np
from keras import backend as K


os.environ["CUDA_VISIBLE_DEVICES"]='1'
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
        model = modelos.load_UNET3D_MULTITASK(64,64,64,1,2,24,0.5)
    else:
        model=modelos.load_UNET3D_SLANT27_v2_groupNorm(64,64,64,1,2,24,0.5)
    model.load_weights(WEIGHTS)
    model.compile(optimizer=optimizers.Adam(0.0001), loss=losses.GJLsmooth, metrics=[metrics.mdice])


    print('Total images to process: '+str(len(listaFLAIR)))

    for FLAIR_name in listaFLAIR:
        print('processing:')
        print(FLAIR_name)
        FLAIR=load_flair(FLAIR_name)
        SEG_mask=seg_majvote_flair_ssl(FLAIR,model,nbNN=[5,5,5],ps=[64,64,64],regularized=regularized)
        name=pred_path+FLAIR_name.split('/')[-1].split('.nii')[0]+'_SEG.nii.gz'
        T1_img = nii.load(FLAIR_name)
        img = nii.Nifti1Image(SEG_mask.astype(np.uint8), T1_img.affine )
        img.to_filename(name)
    K.clear_session()
    gc.collect() #free memory



con_reg=sorted(glob.glob("weights/consistency_reg_rot.h5"))
recon=sorted(glob.glob("weights/recon_and_seg.h5"))
uncertainty=sorted(glob.glob("weights/uncertainty_pseudo_lab.h5"))

seg_to_folder_with_Weightlist(con_reg,False,listaFLAIR_mso)
#seg_to_folder_with_Weightlist(uncertainty,False,listaFLAIR_mso)
#seg_to_folder_with_Weightlist(recon,True,listaFLAIR_mso)
