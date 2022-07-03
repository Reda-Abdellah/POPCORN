import  os, glob, losses, gc
from utils import seg_soft_majvote_times_2D, normalize_image, keyword_toList
import nibabel as nii
import numpy as np
from scipy.ndimage import zoom 
import torch
import torchio as tio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lib_path = os.path.join("..","dataset/TestImage")
listaT1=keyword_toList(path=lib_path,keyword="*.gz")



def seg_to_folder_with_Weightlist_times_2D(weight_folder_name, listaWeights,listaT1, ps= [224,224]):
    WEIGHTS= listaWeights
    #weight_folder_name= WEIGHTS[0].split('/')[-2]
    pred_path='SEG/'+weight_folder_name+'/'
    try:
        os.mkdir(pred_path)
    except OSError:
        print ("Creation of the directory %s failed" % pred_path)
    else:
        print ("Successfully created the directory %s " % pred_path)
    seg_to_folder_times_2D(pred_path,WEIGHTS,listaT1, ps=ps)


def seg_to_folder_times_2D(pred_path,WEIGHTS,listaT1, ps= [224,224], bg=20):
    clamp = tio.Clamp(-1000,1000)
    MODELS=[]
    for weight in WEIGHTS:
        MODELS.append(torch.load(weight).to(device).eval())
    print('Total images to process: '+str(len(listaT1)))
    for i in range(len(listaT1)):
        print('processing:')
        T1_name=listaT1[i]
        print(T1_name)
        IM= nii.load(T1_name).get_data()

        IM= IM[bg:IM.shape[0]-bg, bg:IM.shape[1]-bg, :]
        
        ratiox= 236/IM.shape[0]
        ratioy= 236/IM.shape[1]
        ratioz= 236/IM.shape[2]

        print("Rescaling....")
        IM= zoom(IM, (ratiox, ratioy, ratioz))
        print("normalization....")
        IM= clamp(IM[np.newaxis,:,:,:])[0]
        
        print("Seg....")
        out = seg_soft_majvote_times_2D(IM,MODELS,ps=ps)
        print("Rescale up....")
        
        out_rescaled= np.zeros( (out.shape[0], int(np.ceil(out.shape[1]/ratiox)), int(np.ceil(out.shape[2]/ratioy)), int(np.ceil(out.shape[3]/ratioz))) )
        for channel in range(out.shape[0]):
            out_rescaled[channel]= zoom(out[channel], (1/ratiox, 1/ratioy, 1/ratioz) , order=1)#, mode='nearest', order=0)
        
        out_rescaled=np.argmax(out_rescaled, axis=0)
        
        out_rescaled= np.pad( out_rescaled, pad_width= ( (bg,bg), (bg,bg), (0,0)  ))
            
        name=pred_path+T1_name.split('/')[-2]+T1_name.split('/')[-1].split('.nii')[0]+'_SEG.nii.gz'
        img = nii.Nifti1Image(out_rescaled.astype(np.uint8), nii.load(T1_name).affine )
        img.to_filename(name)
    gc.collect() #free memory


#listaWeights= keyword_toList(path="weights/",keyword="SUPERVISED_2D_T1_FLAIR_regularized224_mixupresnet18dice_0.pt")

listaWeights=[  "weights/SUPERVISED_2D_T1_FLAIR_regularized224_mixupresnet18dice_axis_0_k1_.pt",
                "weights/SUPERVISED_2D_T1_FLAIR_regularized224_mixupresnet18dice_axis_1_k0.pt",
                "weights/SUPERVISED_2D_T1_FLAIR_regularized224_mixupresnet18dice_0.pt"]

seg_to_folder_with_Weightlist_times_2D('supervised_25D',listaWeights,listaT1[3:4])