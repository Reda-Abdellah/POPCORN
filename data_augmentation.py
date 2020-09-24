#################################################################
#
# AssemblyNET: Deep learning for Brain segmentation
#
# Authors: Jose Vicente Manjon Herrera
#          Pierrick Coupe
#
#    Date: 12/02/2019
#
#################################################################

from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import numpy as np
import nibabel as nii
import time
import gc
import modelos
import scipy.misc
import random
import keras
#import losses
from scipy.ndimage import gaussian_filter
from scipy.special import erfinv
from scipy import ndimage


def data_augment(lesion_batch):
    for i in range(lesion_batch.shape[0]):
        a=lesion_batch[i]
        op=np.random.choice(10,1)
        if(op==1):
            a=np.rot90(a,k=2,axes=(1,0))
        elif(op==9):
            a=np.rot90(a,k=1,axes=(1,0))

        elif(op==2):
            a=np.rot90(a,k=3,axes=(1,0))
        elif(op==3):
            a=np.rot90(a,k=2,axes=(2,0))
        elif(op==4):
            a=np.rot90(a,k=1,axes=(2,0))
        elif(op==5):
            a=np.rot90(a,k=3,axes=(2,0))
        elif(op==6):
            a=np.rot90(a,k=2,axes=(1,2))
        elif(op==7):
            a=np.rot90(a,k=1,axes=(1,2))
        elif(op==8):
            a=np.rot90(a,k=3,axes=(1,2))

        op=np.random.choice(4,1)
        if(op==1):
            a=a[-1::-1,:,:]
        elif(op==2):
            a=a[:,:,-1::-1]
        elif(op==3):
            a=a[:,-1::-1,:]
        lesion_batch[i]=a
    return lesion_batch

def small_Lesion_Generator(x,y,Heatmap,lesions_min400,lesions_min200,lesions_90_200,lesions):
    """
    Heatmap[0:8,:,:]=0
    Heatmap[:,:,0:8]=0
    Heatmap[:,0:8,:]=0
    Heatmap[:,:,80-7:80]=0
    Heatmap[:,96-7:96,:]=0
    Heatmap[80-7:80,:,:]=0
    """
    Heatmap[0:13,:,:]=0
    Heatmap[:,:,0:13]=0
    Heatmap[:,0:13,:]=0
    Heatmap[:,:,80-12:80]=0
    Heatmap[:,96-12:96,:]=0
    Heatmap[80-12:80,:,:]=0
    #print(heatmap.shape)
    while(1):
        ind1=np.random.permutation(x.shape[0])
        ind2=np.random.permutation(lesions_min200.shape[0])

        ind3=np.random.permutation(lesions_90_200.shape[0])
        #lesions=np.concatenate((lesions,lesions),axis=0)
        ind4=np.random.permutation(lesions.shape[0])
        ind11=np.random.permutation(lesions_min400.shape[0])
        #print('heree')
        indd11=0
        indd2=0
        indd4=0
        indd3=0
        ind11_len=len(ind11)
        ind2_len=len(ind2)
        ind3_len=len(ind3)
        ind4_len=len(ind4)
        for n in range(0,x.shape[0]):
                heatmap=Heatmap
                #heatmap_one=Heatmap_one
                x_=  np.copy(x[ind1[n]])
                y_=  np.copy(y[ind1[n]])
                #c=0
                y_=y_.astype('uint8')

                #super_threshold_indices = x_[...,0] < 1.2
                #heatmap[super_threshold_indices]=0

                #heatmap_one[super_threshold_indices]=0
                heatmap=heatmap/heatmap.sum()
                #heatmap=heatmap.reshape((-1,))
                #heatmap_one=heatmap_one/heatmap_one.sum()
                #heatmap_one=heatmap_one.reshape((-1,))
                #max_flair=((y_[...,1]==1)*x_[...,1]).max()
                #max_t1=((y_[...,1]==1)*x_[...,0]).max()
                #min_flair=((y_[...,1]==1)*x_[...,1]).min()
                #min_t1=((y_[...,1]==1)*x_[...,0]).min()


                #t1_underLAB=(y_[...,1]==1)*x_[...,0]
                #flair_underLAB=(y_[...,1]==1)*x_[...,1]

                #print( np.where(np.isnan(heatmap)))
                #print(y_[...,1].shape)

                for j in range(0,15):





                    if(np.isnan(heatmap.reshape((-1,)).sum())):
                            break

                    #print(heatmap.shape)
                    op=np.random.choice(heatmap.reshape((-1,)).shape[0], 1, p=heatmap.reshape((-1,)))

                    position_x,position_y,position_z=np.unravel_index(op, (80,96,80))
                    position_z=int(position_z)
                    position_y=int(position_y)
                    position_x=int(position_x)
                    #print(position_x)
                    #print(position_y)
                    #print(position_z)

                    #sum_pos=y_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,1]
                    #print(lesions_min400[ind11[indd11],:,:,:,0].shape)

                    if((y_[position_x-12:position_x+13,position_y-12:position_y+13,position_z-12:position_z+13,1]*lesions_min400[ind11[indd11],:,:,:,0]).sum()==0):
                        #sigma = np.exp(np.random.uniform(np.log(0.4), np.log(0.6)))
                        #mask= gaussian_filter(lesions[ind2[n],:,:,:,0][0], sigma)
                        mask_d= gaussian_filter(lesions_min400[ind11[indd11],:,:,:,0], np.exp(np.random.uniform(np.log(1), np.log(1.3))))
                        mask_d=mask_d/mask_d.max()
                        mask=lesions_min400[ind11[indd11],:,:,:,0]
                        y_[position_x-12:position_x+13,position_y-12:position_y+13,position_z-12:position_z+13,1]= mask + (1-mask)*y_[position_x-12:position_x+13,position_y-12:position_y+13,position_z-12:position_z+13,1]
                        x_[position_x-12:position_x+13,position_y-12:position_y+13,position_z-12:position_z+13,0]= mask_d *(lesions_min400[ind11[indd11],:,:,:,1])  + (1-mask_d)*x_[position_x-12:position_x+13,position_y-12:position_y+13,position_z-12:position_z+13,0]
                        x_[position_x-12:position_x+13,position_y-12:position_y+13,position_z-12:position_z+13,1]= mask_d *(lesions_min400[ind11[indd11],:,:,:,2])+ (1-mask_d)*x_[position_x-12:position_x+13,position_y-12:position_y+13,position_z-12:position_z+13,1]
                        heatmap[position_x-12:position_x+13,position_y-12:position_y+13,position_z-12:position_z+13]=0
                        heatmap=heatmap/heatmap.sum()
                    if((indd11+1)%ind11_len==0):
                        indd11=0
                    else:
                        indd11=indd11+1


                for j in range(0,15):





                    if(np.isnan(heatmap.reshape((-1,)).sum())):
                            break

                    #print(heatmap.shape)
                    op=np.random.choice(heatmap.reshape((-1,)).shape[0], 1, p=heatmap.reshape((-1,)))

                    position_x,position_y,position_z=np.unravel_index(op, (80,96,80))
                    position_z=int(position_z)
                    position_y=int(position_y)
                    position_x=int(position_x)
                    #print(position_x)
                    #print(position_y)
                    #print(position_z)

                    #sum_pos=y_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,1]

                    if((y_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,1]*lesions_min200[ind2[indd2],:,:,:,0]).sum()==0):
                        #sigma = np.exp(np.random.uniform(np.log(0.4), np.log(0.6)))
                        #mask= gaussian_filter(lesions[ind2[n],:,:,:,0][0], sigma)
                        mask_d= gaussian_filter(lesions_min200[ind2[indd2],:,:,:,0], np.exp(np.random.uniform(np.log(1), np.log(1.3))))
                        mask_d=mask_d/mask_d.max()
                        mask=lesions_min200[ind2[indd2],:,:,:,0]

                        y_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,1]= mask + (1-mask)*y_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,1]
                        x_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,0]= mask_d *(lesions_min200[ind2[indd2],:,:,:,1])  + (1-mask_d)*x_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,0]
                        x_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,1]= mask_d *(lesions_min200[ind2[indd2],:,:,:,2])+ (1-mask_d)*x_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,1]
                        heatmap[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8]=0
                        heatmap=heatmap/heatmap.sum()
                    if((indd2+1)%ind2_len==0):
                        indd2=0
                    else:
                        indd2=indd2+1


                for j in range(0,15):

                    if(np.isnan(heatmap.reshape((-1,)).sum())):
                            break

                    op=np.random.choice(heatmap.reshape((-1,)).shape[0], 1, p=heatmap.reshape((-1,)))

                    position_x,position_y,position_z=np.unravel_index(op, (80,96,80))
                    position_z=int(position_z)
                    position_y=int(position_y)
                    position_x=int(position_x)
                    #print(position_x)
                    #print(position_y)
                    #print(position_z)

                    #sum_pos=y_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,1]

                    if((y_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,1]*lesions_90_200[ind3[indd3],:,:,:,0]).sum()==0):
                        #sigma = np.exp(np.random.uniform(np.log(0.4), np.log(0.6)))
                        #mask= gaussian_filter(lesions[ind2[n],:,:,:,0][0], sigma)
                        mask_d= gaussian_filter(lesions_90_200[ind3[indd3],:,:,:,0], np.exp(np.random.uniform(np.log(1), np.log(1.3))))
                        mask_d=mask_d/mask_d.max()
                        mask=lesions_90_200[ind3[indd3],:,:,:,0]

                        y_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,1]= mask + (1-mask)*y_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,1]
                        x_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,0]= mask_d *(lesions_90_200[ind3[indd3],:,:,:,1] ) + (1-mask_d)*x_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,0]
                        x_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,1]=mask_d *lesions_90_200[ind3[indd3],:,:,:,2]+ (1-mask_d)*x_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,1]
                        heatmap[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8]=0
                        heatmap=heatmap/heatmap.sum()
                    if((indd3+1)%ind3_len==0):
                        indd3=0
                    else:
                        indd3=indd3+1


                for j in range(0,15):

                    if(np.isnan(heatmap.reshape((-1,)).sum())):
                            break

                    op=np.random.choice(heatmap.reshape((-1,)).shape[0], 1, p=heatmap.reshape((-1,)))

                    position_x,position_y,position_z=np.unravel_index(op, (80,96,80))
                    position_z=int(position_z)
                    position_y=int(position_y)
                    position_x=int(position_x)
                    #print(position_x)
                    #print(position_y)
                    #print(position_z)

                    #sum_pos=y_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,1]

                    if((y_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,1]*lesions[ind4[indd4],:,:,:,0]).sum()==0):
                        #sigma = np.exp(np.random.uniform(np.log(0.4), np.log(0.6)))
                        #mask= gaussian_filter(lesions[ind2[n],:,:,:,0][0], sigma)
                        mask_d= gaussian_filter(lesions[ind4[indd4],:,:,:,0], np.exp(np.random.uniform(np.log(1), np.log(1.3))))
                        mask_d=mask_d/mask_d.max()
                        mask=lesions[ind4[indd4],:,:,:,0]

                        y_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,1]= mask + (1-mask)*y_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,1]
                        x_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,0]= mask_d *(lesions[ind4[indd4],:,:,:,1])+(1-mask_d) *x_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,0]
                        x_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,1]= mask_d *(lesions[ind4[indd4],:,:,:,2])+ (1-mask_d)*x_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,1]

                        heatmap[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8]=0
                        heatmap=heatmap/heatmap.sum()
                    if((indd4+1)%ind4_len==0):
                        indd4=0
                    else:
                        indd4=indd4+1


                y_0 = y_[...,0]
                y_1 = y_[...,1]
                y_0[y_1 ==1 ]=0
                y_=np.concatenate((np.expand_dims(y_0,axis=-1),np.expand_dims(y_1,axis=-1)),axis=-1)
                #t1_underLAB_add=(LAB_==1)*T1_
                #flair_underLAB_add=(y_[...,1]==1)*x_[...,1]
                #t1_underLAB_add=t1_underLAB_add-t1_underLAB
                #flair_underLAB_add=flair_underLAB_add-flair_underLAB


                #where_zero_t1_add= np.where(t1_underLAB_add.reshape((-1,))==0)
                #where_zero_flair_add= np.where(flair_underLAB_add.reshape((-1,))==0)
                #where_notzero_t1_add= np.where(np.logical_not(t1_underLAB_add.reshape((-1,))==0))
                #where_notzero_flair_add= np.where(np.logical_not(flair_underLAB_add.reshape((-1,))==0))


                #t1_underLAB_add_vect=t1_underLAB_add.reshape((-1,))
                #flair_underLAB_add_vect= flair_underLAB_add.reshape((-1,))

                #t1_underLAB_add_nozero= np.delete( t1_underLAB_add_vect , where_zero_t1_add  )
                #t1_underLAB_nozero= np.delete(t1_underLAB.reshape((-1,)), np.where(t1_underLAB.reshape((-1,))==0))
                #flair_underLAB_add_nozero= np.delete( flair_underLAB_add_vect , where_zero_flair_add  )
                #flair_underLAB_nozero= np.delete(flair_underLAB.reshape((-1,)), np.where(flair_underLAB.reshape((-1,))==0))


                #t1_underLAB_add_nozero=hist_match(t1_underLAB_add_nozero, t1_underLAB_nozero)
                #flair_underLAB_add_nozero=hist_match(flair_underLAB_add_nozero, flair_underLAB_nozero)

                #t1_underLAB_add_vect[where_notzero_t1_add]= t1_underLAB_add_nozero
                #flair_underLAB_add_vect[where_notzero_flair_add]= flair_underLAB_add_nozero


                #t1_underLAB_add=t1_underLAB_add_vect.reshape((80,96,80))
                #flair_underLAB_add=flair_underLAB_add_vect.reshape((80,96,80))



                #T1_=np.logical_not(LAB_==1)*T1_+t1_underLAB + t1_underLAB_add
                #FLAIR_=np.logical_not(y_[...,1]==1)*x_[...,1]+flair_underLAB + flair_underLAB_add

                #print(y_[...,-1].sum())

                x_=np.reshape(x_,(1,x.shape[1],x.shape[2],x.shape[3],x.shape[4]))
                y_=np.reshape(y_,(1,y.shape[1],y.shape[2],y.shape[3],y.shape[4]))
                yield x_,y_


def rot90_Generator(x,y):
    while(1):
        ind=np.random.permutation(x.shape[0])
        for n in range(0,x.shape[0]):
            xi=x[ind[n]:ind[n]+1,:,:,:,:]
            yi=y[ind[n]:ind[n]+1,:,:,:,:]
            xi=np.concatenate((xi,yi),axis=4)
            xi=data_augment(xi)
            yi=xi[:,:,:,:,4:6]
            xi=xi[:,:,:,:,0:4]
            yield xi,yi

def rot90_Generator_2mods(x,y):
    while(1):
        ind=np.random.permutation(x.shape[0])
        for n in range(0,x.shape[0]):
            xi=x[ind[n]:ind[n]+1,:,:,:,:]
            yi=y[ind[n]:ind[n]+1,:,:,:,:]
            xi=np.concatenate((xi,yi),axis=4)
            xi=data_augment(xi)
            yi=xi[:,:,:,:,2:4]
            xi=xi[:,:,:,:,0:2]
            yield xi,yi



def small_Lesion(x,y,i):

    Heatmap=np.load("../small_label.npy")[i,:,:,:]
    lesions=np.load("../lib/small_lesions_MSOPXX_30_90.npy")
    lesions_90_200=np.load("../lib/small_lesions_volbrain_90_200.npy")
    lesions_min200=np.load("../lib/small_lesions_volbrain_min200.npy")

    Heatmap[0:8,:,:]=0
    Heatmap[:,:,0:8]=0
    Heatmap[:,0:8,:]=0
    Heatmap[:,:,80-7:80]=0
    Heatmap[:,96-7:96,:]=0
    Heatmap[80-7:80,:,:]=0
    while(1):
        #ind2=np.random.permutation(lesions_min200.shape[0])
        ind3=np.random.permutation(lesions_90_200.shape[0])
        ind4=np.random.permutation(lesions.shape[0])


        for n in range(0,ind4.shape[0]):
                heatmap=Heatmap
                x_=  np.copy(x)
                y_=  np.copy(y)
                y_=y_.astype('uint8')
                super_threshold_indices = x_[...,0] < 1.2
                heatmap[super_threshold_indices]=0
                heatmap=heatmap/heatmap.sum()
                t1_underLAB=(y_[...,1]==1)*x_[...,0]
                flair_underLAB=(y_[...,1]==1)*x_[...,1]

                for j in range(0,25):

                    if(np.isnan(heatmap.reshape((-1,)).sum())):
                            break

                    op=np.random.choice(heatmap.reshape((-1,)).shape[0], 1, p=heatmap.reshape((-1,)))

                    position_x,position_y,position_z=np.unravel_index(op, (80,96,80))
                    position_z=int(position_z)
                    position_y=int(position_y)
                    position_x=int(position_x)

                    if((y_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,1]*lesions_90_200[ind3[int(25*n+j)],:,:,:,0]).sum()==0):

                        mask_d= gaussian_filter(lesions_90_200[ind3[int(25*n+j)],:,:,:,0], np.exp(np.random.uniform(np.log(1), np.log(1.3))))
                        mask_d=mask_d/mask_d.max()
                        mask=lesions_90_200[ind3[int(25*n+j)],:,:,:,0]

                        y_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,1]= mask + (1-mask)*y_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,1]
                        x_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,0]=  -np.abs( mask *(lesions_90_200[ind3[int(25*n+j)],:,:,:,1]-x_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,0]) )    + (1-mask)*x_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,0]
                        x_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,1]=mask *lesions_90_200[ind3[int(25*n+j)],:,:,:,2]+ (1-mask)*x_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,1]
                        #c=c+1
                    heatmap[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8]=0
                    heatmap=heatmap/heatmap.sum()

                for j in range(0,25):

                    if(np.isnan(heatmap.reshape((-1,)).sum())):
                            break

                    op=np.random.choice(heatmap.reshape((-1,)).shape[0], 1, p=heatmap.reshape((-1,)))

                    position_x,position_y,position_z=np.unravel_index(op, (80,96,80))
                    position_z=int(position_z)
                    position_y=int(position_y)
                    position_x=int(position_x)

                    if((y_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,1]*lesions_min200[ind4[int(25*n+j)],:,:,:,0]).sum()==0):
                        mask_d= gaussian_filter(lesions[ind4[int(25*n+j)],:,:,:,0], np.exp(np.random.uniform(np.log(1), np.log(1.3))))
                        mask_d=mask_d/mask_d.max()
                        mask=lesions[ind4[int(25*n+j)],:,:,:,0]

                        y_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,1]= mask + (1-mask)*y_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,1]
                        x_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,0]=  -np.abs( mask *(lesions[ind4[int(25*n+j)],:,:,:,1]-x_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,0]) )    + (1-mask)*x_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,0]
                        x_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,1]=mask *lesions[ind4[int(25*n+j)],:,:,:,2]+ (1-mask)*x_[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8,1]
                    heatmap[position_x-7:position_x+8,position_y-7:position_y+8,position_z-7:position_z+8]=0
                    heatmap=heatmap/heatmap.sum()

                x_=np.expand_dims(x_,0)
                y_=np.expand_dims(y_,0)
                yield x_,y_





def MixUp_small_lesion_Generator(x,y,i):

    while(1):
        alfa=0.3
        ind1=np.random.permutation(x.shape[0])
        ind2=np.random.permutation(x.shape[0])


        for n in range(0,x.shape[0]):
            x1,y1=next(small_Lesion(x[ind1[n]],y[ind1[n]],i))
            x2,y2=next(small_Lesion(x[ind2[n]],y[ind2[n]],i))
            #mixup
            a=np.random.beta(alfa,alfa)
            x3=a*x1+(1-a)*x2
            y3=a*y1+(1-a)*y2
            x3=np.reshape(x3,(1,x.shape[1],x.shape[2],x.shape[3],x.shape[4]))
            y3=np.reshape(y3,(1,y.shape[1],y.shape[2],y.shape[3],y.shape[4]))
            yield x3,y3


def cow_mixing_mask(img_size, sigma_min, sigma_max, p_min, p_max):
    #call:   generate_mixing_mask((80,90,80), (5/128)*80, (7/128)*80, 0.4, 0.6)
    #p_min and p_max gives good result around ~0.5
    #sigma_min and max gives good result depending on img_size
    # Randomly draw sigma from log-uniform distribution
    sigma = np.exp(np.random.uniform(np.log(sigma_min), np.log(sigma_max)))
    p = np.random.uniform(p_min, p_max)
    # Randomly draw proportion p
    N = np.random.normal(size=img_size)
    # Generate noise image
    Ns = gaussian_filter(N, sigma)
    # Smooth with a Gaussian
    # Compute threshold
    t = erfinv(p*2 - 1) * (2**0.5) * Ns.std() + Ns.mean()
    return (Ns > t).astype(float) # Apply threshold and return

def CowMix_Generator(x,y,imsize=(80,90,80)):
    while(1):
        ind1=np.random.permutation(x.shape[0])
        ind2=np.random.permutation(x.shape[0])
        for n in range(0,x.shape[0]):
            #mixup

            a=cow_mixing_mask(imsize, (5/128)*80, (7/128)*80, 0.4, 0.6)
            a=np.expand_dims(a,axis=3)
            a=np.concatenate((a,a),axis=3)
            #a=np.concatenate((a,a,a,a),axis=3)
            y2=a*y[ind1[n]]+(1-a)*y[ind2[n]]
            a=np.concatenate((a,a),axis=3)
            x2=a*x[ind1[n]]+(1-a)*x[ind2[n]]
            x2=np.reshape(x2,(1,x.shape[1],x.shape[2],x.shape[3],x.shape[4]))
            y2=np.reshape(y2,(1,y.shape[1],y.shape[2],y.shape[3],y.shape[4]))
            yield x2,y2


def DA_2Ddegradation(x,y):
    while(1):
        ind1=np.random.permutation(x.shape[0])
        mods=x.shape[-1]
        for i in range(0,x.shape[0]):

            x_=  np.copy(x[ind1[i]])
            y_=  np.copy(y[ind1[i]])

            op=np.random.choice(6,1,p=[0.1,0.15,0.15,0.2,0.2,0.2]) # 0:nada, 1:sharp, 2:blur, 3: axial blur 3, 4: axial blur 5, 5: axial blur 2

            if(op==1):
                for j in range(x_.shape[3]):
                    x_[:,:,:,j] = 2*x_[:,:,:,j]-ndimage.uniform_filter(x_[:,:,:,j], (3,3,3))

            if(op==2):
                for j in range(x_.shape[3]):
                    x_[:,:,:,j] = ndimage.uniform_filter(x_[:,:,:,j], (3,3,3))

            if(op==3):
                for j in range(x_.shape[3]):
                    #x_[:,:,:,j]=ndimage.uniform_filter(x_[:,:,:,j], (1,1,3))
                    x_[:,:,:,j]=ndimage.uniform_filter(x_[:,:,:,j], (1,2,1))

            if(op==4):
                for j in range(x_.shape[3]):
                    #x_[:,:,:,j] = ndimage.uniform_filter(x_[:,:,:,j], (1,1,5))
                    x_[:,:,:,j] = ndimage.uniform_filter(x_[:,:,:,j], (2,1,1))

            if(op==5):
                for j in range(x_.shape[3]):
                    x_[:,:,:,j] =ndimage.uniform_filter(x_[:,:,:,j], (1,1,2))

            x_=np.expand_dims(x_, axis=0)
            y_=np.expand_dims(y_, axis=0)


            yield x_,y_





def MixUp_Generator(x,y,alfa,use_sometimes=False, use_2Ddegradation=False):
    while(1):
        ind1=np.random.permutation(x.shape[0])
        ind2=np.random.permutation(x.shape[0])
        if(use_2Ddegradation):
            for i in range(x.shape[0]):
                x_=np.zeros((x.shape))
                op=np.random.choice(5,1,p=[0.3,0.05,0.05,0.4,0.2]) # 0:nada, 1:sharp, 2:blur

                if(op==1):
                    x_[i,:,:,:,0] = 2*x[i,:,:,:,0]-ndimage.uniform_filter(x[i,:,:,:,0], (3,3,3))
                    x_[i,:,:,:,1] = 2*x[i,:,:,:,1]-ndimage.uniform_filter(x[i,:,:,:,1], (3,3,3))

                if(op==2):
                    x_[i,:,:,:,0] = ndimage.uniform_filter(x[i,:,:,:,0], (3,3,3))
                    x_[i,:,:,:,1] = ndimage.uniform_filter(x[i,:,:,:,1], (3,3,3))

                if(op==3):
                    x_[i,:,:,:,0] =ndimage.uniform_filter(x[i,:,:,:,0], (3,3,3))
                    x_[i,:,:,:,1] =ndimage.uniform_filter(x[i,:,:,:,1], (3,3,3))

                if(op==4):
                    x_[i,:,:,:,0] =ndimage.uniform_filter(x[i,:,:,:,0], (1,1,3))
                    x_[i,:,:,:,1] =ndimage.uniform_filter(x[i,:,:,:,1], (1,1,5))

                x=x_

        for n in range(0,x.shape[0]):
            #mixup
            if(use_sometimes):
                if(random.random()<0.5):
                    a=np.random.beta(alfa,alfa)
                else:
                    a=1
            else:
                a=np.random.beta(alfa,alfa)
            x2=a*x[ind1[n]]+(1-a)*x[ind2[n]]
            y2=a*y[ind1[n]]+(1-a)*y[ind2[n]]
            x2=np.reshape(x2,(1,x.shape[1],x.shape[2],x.shape[3],x.shape[4]))
            y2=np.reshape(y2,(1,y.shape[1],y.shape[2],y.shape[3],y.shape[4]))
            yield x2,y2


def nMixUp_Generator(x,y,n=5):
    while(1):
        ind=np.zeros((n,x.shape[0]))
        for i in range(n):
            ind[i,:]=(np.random.permutation(x.shape[0]))
        ind=ind.astype('int')
        for i in range(0,x.shape[0]):
            a=np.random.normal(loc=0.5, scale=0.5, size=(n,1,1,1,1))
            a[0]=0
            a[0]=a.sum()*(0.7/0.3)
            a=a/a.sum()

            x2=a*x[ind[:,i].reshape(-1)]
            x2=x2.sum(axis=0)
            y2=a*y[ind[:,i].reshape(-1)]
            y2=y2.sum(axis=0)
            #print(y2.shape)
            x2=np.reshape(x2,(1,x.shape[1],x.shape[2],x.shape[3],x.shape[4]))
            y2=np.reshape(y2,(1,y.shape[1],y.shape[2],y.shape[3],y.shape[4]))
            yield x2,y2





def Lesion_Generator(lesion):
    while(1):
        ind=np.random.permutation(lesion.shape[0])
        for n in range(0,lesion.shape[0]):

            x1_lesion=lesion[ind[n],:,:,:,0]
            x1_t1=lesion[ind[n],:,:,:,1]
            x1_flair=lesion[ind[n],:,:,:,2]
            y_t1=x1_t1
            y_flair=x1_flair
            #print('hi')
            """
            mean_t1= x1_t1[np.where(x1_lesion==1)].mean()
            mean_flair= x1_flair[np.where(x1_lesion==1)].mean()
            std_t1= x1_t1[np.where(x1_lesion==1)].std()
            std_flair= x1_flair[np.where(x1_lesion==1)].std()
            #print(std_flair.size)
            if(std_flair<0 or np.isnan(std_flair)):
                std_flair=1
            if(std_t1<0 or np.isnan(std_t1)):
                std_t1=1

            x1_t1 = np.random.normal(loc=mean_t1,scale=std_t1, size=(x1_t1.shape[0],x1_t1.shape[1],x1_t1.shape[2]) )* (x1_lesion) + (1-x1_lesion)* x1_t1
            x1_flair = np.random.normal(loc= mean_flair,scale=std_flair, size=(x1_t1.shape[0],x1_t1.shape[1],x1_t1.shape[2]) )* (x1_lesion) + (1-x1_lesion)* x1_flair
            """
            noise = np.random.normal(0, 1, ( 25,25,25,2))
            x1_t1 = noise[...,0] *  (x1_lesion) + (1-x1_lesion)* x1_t1
            x1_flair = noise[...,1] *  (x1_lesion) + (1-x1_lesion)* x1_flair

            x1_lesion=np.expand_dims(x1_lesion,axis=0)
            x1_t1=np.expand_dims(x1_t1,axis=0)
            x1_flair=np.expand_dims(x1_flair,axis=0)
            y_t1=np.expand_dims(y_t1,axis=0)
            y_flair=np.expand_dims(y_flair,axis=0)

            x1_lesion=np.expand_dims(x1_lesion,axis=4)
            x1_t1=np.expand_dims(x1_t1,axis=4)
            x1_flair=np.expand_dims(x1_flair,axis=4)
            y_t1=np.expand_dims(y_t1,axis=4)
            y_flair=np.expand_dims(y_flair,axis=4)

            x3=np.concatenate((x1_lesion,x1_t1,x1_flair),axis=4)
            y3=np.concatenate((y_t1,y_flair),axis=4)
            yield x3,y3


def MixUp_Generator_ssl(x,y,x_vb,y_vb,alfa=0.3,use_sometimes=False):
    while(1):
        ind1=np.random.permutation(x.shape[0])
        ind2=np.random.permutation(x_vb.shape[0])
        ind3=np.random.permutation(x_vb.shape[0])
        for n in range(0,x.shape[0]):
            #mixup
            if(use_sometimes):
                if(random.random()<0.5):
                    a=np.random.beta(alfa,alfa)
                else:
                    a=1
            else:
                a=np.random.beta(alfa,alfa)
            x2vb=a*x_vb[ind3[n]]+(1-a)*x_vb[ind2[n]]
            y2vb=a*y_vb[ind3[n]]+(1-a)*y_vb[ind2[n]]
            x2=x[ind1[n]]
            y2=y[ind1[n]]
            x2=np.reshape(x2,(1,x.shape[1],x.shape[2],x.shape[3],x.shape[4]))
            y2=np.reshape(y2,(1,y.shape[1],y.shape[2],y.shape[3],y.shape[4]))
            x2vb=np.reshape(x2vb,(1,x.shape[1],x.shape[2],x.shape[3],x.shape[4]))
            y2vb=np.reshape(y2vb,(1,y.shape[1],y.shape[2],y.shape[3],y.shape[4]))
            x3=np.concatenate((x2,x2vb),axis=0)
            y3=np.concatenate((y2,y2vb),axis=0)
            yield x3,y3



class Generator_feature_consistancy_AE(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x,y,filepath_teacher ,session,alfa=0.3,use_sometimes=False,epoch_size=300,consistency='MixUp'):
        'Initialization'
        self.x=x
        self.y=y
        self.session=session
        self.alfa=alfa
        self.use_sometimes=use_sometimes
        self.epoch_size=epoch_size
        self.batch_size=1
        self.consistency=consistency
        self.filepath_teacher=filepath_teacher
        model_AE=modelos.load_UNET_AE(80,90,80,2,2,24,0)
        model_AE.summary()
        self.f = K.function([model_AE.layers[0].input, K.learning_phase()],[K.pool3d(model_AE.get_layer('bottleneck').output, (2,2,2),strides=(2,2,2))])
        #self.f = K.function([model_AE.layers[0].input, K.learning_phase()],[model_AE.get_layer('bottleneck_reduced').output])
        self.on_epoch_end()



    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        #indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        ind1=self.ind1[index*self.batch_size:(index+1)*self.batch_size]
        ind2=self.ind2[index*self.batch_size:(index+1)*self.batch_size]

        a=self.a[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = ind1, ind2,a

        # Generate data
        X, y1,y2 = self.__data_generation(list_IDs_temp)
        #print('shape X: '+str(X.shape))
        #print('shape y1: '+str(y1.shape))
        #print('shape y2: '+str(y2.shape))
        return X,[y1,y2]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.ind1=np.random.permutation(self.x.shape[0])
        self.ind2=np.random.permutation(self.x.shape[0])

        self.a=np.random.beta(self.alfa,self.alfa,(self.x.shape[0]))


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        ind1,ind2,a=list_IDs_temp

        # Generate data
        for n in range(len(ind1)) :
            aa=a[n]
            if(self.consistency=='CowMix'):
                aa=cow_mixing_mask((80,90,80), (5/128)*80, (7/128)*80, 0.4, 0.6)
                aa=np.expand_dims(aa,axis=3)
                aa=np.concatenate((aa,aa),axis=3)
            #x2=aa*self.x[ind1[n]]+(1-aa)*self.x[ind2[n]]
            #y2=aa*self.y[ind1[n]]+(1-aa)*self.y[ind2[n]]
            x2=self.x[ind1[n]]
            y2=self.y[ind1[n]]
            #print(y2vb.shape)
            #print(y2.shape)
            x2=np.expand_dims(x2,axis=0)
            with self.session.as_default():
                with self.session.graph.as_default():
                    y_bottleneck=self.f((x2,0))
                    y_bottleneck=np.asarray(y_bottleneck)
                    y_bottleneck=np.reshape(y_bottleneck, (y_bottleneck.shape[1:6]))
                    #print(y_bottleneck.shape)
            #x2vb=np.expand_dims(x2vb,axis=0)

            y2=np.expand_dims(y2,axis=0)

        return x2, y2, y_bottleneck


class Generator_feature_consistancy(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x,y,x_vb,y_vb,model,alpha=None,session='',alfa=0.3,use_sometimes=False,epoch_size=300,consistency='MixUp'):
        'Initialization'
        self.x=x
        self.y=y
        self.x_vb=x_vb
        self.y_vb=y_vb
        self.alfa=alfa
        self.use_sometimes=use_sometimes
        self.epoch_size=epoch_size
        self.batch_size=2
        self.model=model
        self.consistency=consistency
        #self.alpha=alpha
        self.session = session
        #K.set_session(self.sess)
        #self.model_mean_teacher=model_mean_teacher
        #self.graph = tf.get_default_graph()
        self.first_time=True
        #self.beta=0
        #self.mini_epoch=16
        self.on_epoch_end()
        self.first_time=False


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        #indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        ind1=self.ind1[index*self.batch_size:(index+1)*self.batch_size]
        ind2=self.ind2[index*self.batch_size:(index+1)*self.batch_size]
        ind3=self.ind3[index*self.batch_size:(index+1)*self.batch_size]
        a=self.a[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = ind1, ind2, ind3,a

        # Generate data
        X, y1,y2 = self.__data_generation(list_IDs_temp)
        #print('shape X: '+str(X.shape))
        #print('shape y1: '+str(y1.shape))
        #print('shape y2: '+str(y2.shape))
        return X,[y1,y2]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.ind1=np.random.permutation(self.x.shape[0])
        self.ind2=np.random.permutation(self.x_vb.shape[0])
        self.ind3=np.random.permutation(self.x_vb.shape[0])
        self.a=np.random.beta(self.alfa,self.alfa,(self.x_vb.shape[0]))


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        ind1,ind2,ind3,a=list_IDs_temp

        # Generate data
        for n in range(len(ind1)) :
            aa=a[n]
            if(self.consistency=='CowMix'):
                aa=cow_mixing_mask((80,90,80), (5/128)*80, (7/128)*80, 0.4, 0.6)
                aa=np.expand_dims(aa,axis=3)
                aa=np.concatenate((aa,aa),axis=3)
            x2vb=aa*self.x_vb[ind3[n]]+(1-aa)*self.x_vb[ind2[n]]


            ##x2vb=self.x_vb[ind3[n]] ##
            #y2vb_fake=a[n]*self.y_vb[ind3[n]]+(1-a[n])*self.y_vb[ind2[n]]

            #layer_output=self.model.get_layer('bottleneck').output
            #intermediate_model=tf.keras.models.Model(inputs=model.input,outputs=layer_output)
            with self.session.as_default():
                with self.session.graph.as_default():
                    intermediate_prediction1=self.model.predict(np.expand_dims(self.x_vb[ind3[n]],axis=0))[1]
                    intermediate_prediction2=self.model.predict(np.expand_dims(self.x_vb[ind2[n]],axis=0))[1]

            y2vb=aa*intermediate_prediction1+(1-aa)*intermediate_prediction2
            ##y2vb=intermediate_prediction1
            #y2vb=self.beta*y2vb+(1-self.beta)*y2vb_fake
            #y2vb=y2vb_fake
            x2=self.x[ind1[n]]
            y2=self.y[ind1[n]]
            #print(y2vb.shape)
            #print(y2.shape)
            x2=np.expand_dims(x2,axis=0)
            x2vb=np.expand_dims(x2vb,axis=0)

            y2=np.expand_dims(y2,axis=0)

            y2=np.concatenate((y2,y2),axis=0)
            y2vb=np.concatenate((y2vb,y2vb),axis=0)

            #y2vb=np.expand_dims(y2vb,axis=0)
            x3=np.concatenate((x2,x2vb),axis=0)
            #y3=np.concatenate((y2,y2vb),axis=0)
        return x3, y2,y2vb



class Generator_seg_recon(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x,y,alpha=None,sess='',alfa=0.3,use_sometimes=False,epoch_size=300,consistency='MixUp'):
        'Initialization'
        self.x=x
        self.y=y
        self.x_vb=x_vb
        self.y_vb=y_vb
        self.alfa=alfa
        #self.use_sometimes=use_sometimes
        self.epoch_size=epoch_size
        self.batch_size=1
        self.model=model
        #self.consistency=consistency
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        #indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        ind1=self.ind1[index*self.batch_size:(index+1)*self.batch_size]
        #ind2=self.ind2[index*self.batch_size:(index+1)*self.batch_size]
        #ind3=self.ind3[index*self.batch_size:(index+1)*self.batch_size]
        #a=self.a[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = ind

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        #print(y)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.ind1=np.random.permutation(self.x.shape[0])
        #self.ind2=np.random.permutation(self.x_vb.shape[0])
        #self.ind3=np.random.permutation(self.x_vb.shape[0])



    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization


        # Generate data
        for n in range(len(list_IDs_temp)) :


            #y2vb=self.beta*y2vb+(1-self.beta)*y2vb_fake
            #y2vb=y2vb_fake
            x2=self.x[ind1[n]]
            y2=self.y[ind1[n]]
            y2vb=y2*x2[:,:,:,2]

            #print(y2vb.shape)
            #print(y2.shape)
            x2=np.expand_dims(x2,axis=0)
            y2=np.expand_dims(y2,axis=0)
            #x2vb=np.expand_dims(x2vb,axis=0)
            y2vb=np.expand_dims(y2vb,axis=0)
            x3=np.concatenate((x2,x2vb),axis=0)
            y3=np.concatenate((y2,y2vb),axis=4)
        return x3, y3

class AE_Recon(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x,y):
        'Initialization'
        self.x = x
        self.y = y
        self.batch_size=1
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, index):

        ind1=self.ind1[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(ind1)
        #print(y)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.ind1=np.random.permutation(self.x.shape[0])


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Generate data
        for n in range(len(list_IDs_temp)) :
            x2=self.x[list_IDs_temp[n]]
            y=self.y[list_IDs_temp[n]]
            #y2=y[:,:,:,1:2]*x2[:,:,:,1:2] #flair lesion
            #y2_= y[:,:,:,0:1]  *x2[:,:,:,0:1] #T1 BG
            y3=np.concatenate((x2,y),axis=3)
            x2=np.expand_dims(x2,axis=0)
            y3=np.expand_dims(y3,axis=0)
            #print(y3.shape)
        return x2, y3

class AE_Recon_2(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x,y):
        'Initialization'
        self.x = x
        self.y = y
        self.batch_size=1
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, index):

        ind1=self.ind1[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(ind1)
        #print(y)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.ind1=np.random.permutation(self.x.shape[0])


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Generate data
        for n in range(len(list_IDs_temp)) :
            x2=self.x[list_IDs_temp[n]]
            y2_1=self.y[list_IDs_temp[n]]*x2[:,:,:,1:2]
            y2_2=self.y[list_IDs_temp[n]]*x2[:,:,:,0:1]
            y2= np.concatenate((y2_1,y2_2),axis=-1)
            x2=np.expand_dims(x2,axis=0)
            y2=np.expand_dims(y2,axis=0)

        return x2, y2

class Generator_ict(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x,y,x_vb,y_vb,model,model_mean_teacher,alpha=None,sess='',alfa=0.3,use_sometimes=False,epoch_size=300,consistency='MixUp'):
        'Initialization'
        self.x=x
        self.y=y
        self.x_vb=x_vb
        self.y_vb=y_vb
        self.alfa=alfa
        self.use_sometimes=use_sometimes
        self.epoch_size=epoch_size
        self.batch_size=2
        self.model=model
        self.consistency=consistency
        #self.alpha=alpha
        #self.sess = sess
        #K.set_session(self.sess)
        self.model_mean_teacher=model_mean_teacher
        #self.graph = tf.get_default_graph()
        self.first_time=True
        #self.beta=0
        self.mini_epoch=16
        self.on_epoch_end()
        self.first_time=False


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        #indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        ind1=self.ind1[index*self.batch_size:(index+1)*self.batch_size]
        ind2=self.ind2[index*self.batch_size:(index+1)*self.batch_size]
        ind3=self.ind3[index*self.batch_size:(index+1)*self.batch_size]
        a=self.a[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = ind1, ind2, ind3,a

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        #print(y)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.ind1=np.random.permutation(self.x.shape[0])
        self.ind2=np.random.permutation(self.x_vb.shape[0])
        self.ind3=np.random.permutation(self.x_vb.shape[0])

        self.a=np.random.beta(self.alfa,self.alfa,(self.x_vb.shape[0]))
        #self.ind3=np.concatenate((self.ind3,self.ind3,self.ind3,self.ind3,self.ind3))
        #self.ind2=np.concatenate((self.ind2,self.ind2,self.ind2,self.ind2,self.ind2))
        #self.a=np.concatenate((self.a,self.a,self.a,self.a,self.a))

        if(not self.first_time):
            #self.model_mean_teacher=self.model
            """
            new_w = self.model.get_weights()
            old_w= self.model_mean_teacher.get_weights()
            for layer in range(len(new_w)):
                #running mean
                old_w[layer] = new_w[layer]
            self.model_mean_teacher.set_weights(old_w)
            if(self.beta<1):
                self.beta=self.beta+0.1
            """


            #self.alpha=self.alpha+0.05
            #self.model.compile(optimizer='adam', loss=losses.GJLsmooth_ssl(alpha), metrics=[metrics.mdice_ssl_gt,metrics.mdice_ssl_vb])
            #self.indexes = np.arange(len(self.x_list))
            #np.random.shuffle(self.indexes)
        else:
            print("first epoch")
            new_w = self.model.get_weights()
            old_w= self.model_mean_teacher.get_weights()
            for layer in range(len(new_w)):
                #running mean
                old_w[layer] = new_w[layer]
            self.model_mean_teacher.set_weights(old_w)

            #self.model_mean_teacher=keras.models.clone_model(self.model)
            #self.model_mean_teacher.compile(optimizer='adam', loss=losses.GJLsmooth)
            #self.model_mean_teacher._make_predict_function()


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        ind1,ind2,ind3,a=list_IDs_temp

        # Generate data
        for n in range(len(ind1)) :
            aa=a[n]
            if(self.consistency=='CowMix'):
                aa=cow_mixing_mask((80,90,80), (5/128)*80, (7/128)*80, 0.4, 0.6)
                aa=np.expand_dims(aa,axis=3)
                aa=np.concatenate((aa,aa),axis=3)
            x2vb=aa*self.x_vb[ind3[n]]+(1-aa)*self.x_vb[ind2[n]]
            #y2vb_fake=a[n]*self.y_vb[ind3[n]]+(1-a[n])*self.y_vb[ind2[n]]


            if(self.mini_epoch==0):
                self.mini_epoch=16
                new_w = self.model.get_weights()
                mean_w= self.model_mean_teacher.get_weights()
                for layer in range(len(new_w)):
                    #running mean
                    mean_w[layer] = 0.33*new_w[layer] + 0.67*mean_w[layer]
                self.model_mean_teacher.set_weights(mean_w)

            else:
                self.mini_epoch=self.mini_epoch-1
            y2vb=aa*self.model_mean_teacher.predict(np.expand_dims(self.x_vb[ind3[n]],axis=0))+(1-aa)*self.model_mean_teacher.predict(np.expand_dims(self.x_vb[ind2[n]],axis=0))

            #y2vb=self.beta*y2vb+(1-self.beta)*y2vb_fake
            #y2vb=y2vb_fake
            x2=self.x[ind1[n]]
            y2=self.y[ind1[n]]
            #print(y2vb.shape)
            #print(y2.shape)
            x2=np.expand_dims(x2,axis=0)
            y2=np.expand_dims(y2,axis=0)
            x2vb=np.expand_dims(x2vb,axis=0)
            #y2vb=np.expand_dims(y2vb,axis=0)
            x3=np.concatenate((x2,x2vb),axis=0)
            y3=np.concatenate((y2,y2vb),axis=0)
        return x3, y3





class Generator_mean_teacher_ssl(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x,y,x_vb,y_vb,model,model_mean_teacher,alpha=None,sess='',alfa=0.3,use_sometimes=False,epoch_size=300,consistency='MixUp'):
        'Initialization'
        self.x=x
        self.y=y
        self.x_vb=x_vb
        self.y_vb=y_vb
        self.alfa=alfa
        self.use_sometimes=use_sometimes
        self.epoch_size=epoch_size
        self.batch_size=2
        self.model=model
        self.consistency=consistency
        #self.alpha=alpha
        #self.sess = sess
        #K.set_session(self.sess)
        self.model_mean_teacher=model_mean_teacher
        #self.graph = tf.get_default_graph()
        self.first_time=True
        #self.beta=0
        self.mini_epoch=16
        self.on_epoch_end()
        self.first_time=False


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        #indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        ind1=self.ind1[index*self.batch_size:(index+1)*self.batch_size]
        ind2=self.ind2[index*self.batch_size:(index+1)*self.batch_size]
        #ind3=self.ind3[index*self.batch_size:(index+1)*self.batch_size]
        #a=self.a[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = ind1, ind2   #,ind3,a

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        #print(y)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.ind1=np.random.permutation(self.x.shape[0])
        self.ind2=np.random.permutation(self.x_vb.shape[0])
        #self.ind3=np.random.permutation(self.x.shape[0])

        #self.a=np.random.beta(self.alfa,self.alfa,(self.x.shape[0]))
        #self.ind3=np.concatenate((self.ind3,self.ind3,self.ind3,self.ind3,self.ind3))
        #self.ind2=np.concatenate((self.ind2,self.ind2,self.ind2,self.ind2,self.ind2))
        #self.a=np.concatenate((self.a,self.a,self.a,self.a,self.a))

        if(not self.first_time):
            #self.model_mean_teacher=self.model
            """
            new_w = self.model.get_weights()
            old_w= self.model_mean_teacher.get_weights()
            for layer in range(len(new_w)):
                #running mean
                old_w[layer] = new_w[layer]
            self.model_mean_teacher.set_weights(old_w)
            if(self.beta<1):
                self.beta=self.beta+0.1
            """


            #self.alpha=self.alpha+0.05
            #self.model.compile(optimizer='adam', loss=losses.GJLsmooth_ssl(alpha), metrics=[metrics.mdice_ssl_gt,metrics.mdice_ssl_vb])
            #self.indexes = np.arange(len(self.x_list))
            #np.random.shuffle(self.indexes)
        else:
            print("first epoch")
            new_w = self.model.get_weights()
            old_w= self.model_mean_teacher.get_weights()
            for layer in range(len(new_w)):
                #running mean
                old_w[layer] = new_w[layer]
            self.model_mean_teacher.set_weights(old_w)

            #self.model_mean_teacher=keras.models.clone_model(self.model)
            #self.model_mean_teacher.compile(optimizer='adam', loss=losses.GJLsmooth)
            #self.model_mean_teacher._make_predict_function()


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        #ind1,ind2,ind3,a=list_IDs_temp
        ind1,ind2=list_IDs_temp
        # Generate data
        for n in range(len(ind1)) :

            x2vb=self.x_vb[ind2[n]]
            #y2vb_fake=a[n]*self.y_vb[ind3[n]]+(1-a[n])*self.y_vb[ind2[n]]


            if(self.mini_epoch==0):
                self.mini_epoch=16
                new_w = self.model.get_weights()
                mean_w= self.model_mean_teacher.get_weights()
                for layer in range(len(new_w)):
                    #running mean
                    mean_w[layer] = 0.33*new_w[layer] + 0.67*mean_w[layer]
                self.model_mean_teacher.set_weights(mean_w)

            else:
                self.mini_epoch=self.mini_epoch-1
            y2vb=self.model_mean_teacher.predict(np.expand_dims(self.x_vb[ind2[n]],axis=0))

            #y2vb=self.beta*y2vb+(1-self.beta)*y2vb_fake
            #y2vb=y2vb_fake
            x2=self.x[ind1[n]] #*a[n]+(1-a[n])* self.x[ind3[n]]
            y2=self.y[ind1[n]] #*a[n]+(1-a[n])* self.y[ind3[n]]
            #print(y2vb.shape)
            #print(y2.shape)
            x2=np.expand_dims(x2,axis=0)
            y2=np.expand_dims(y2,axis=0)
            x2vb=np.expand_dims(x2vb,axis=0)
            #y2vb=np.expand_dims(y2vb,axis=0)
            x3=np.concatenate((x2,x2vb),axis=0)
            y3=np.concatenate((y2,y2vb),axis=0)
        return x3, y3





def MixUp_realdata_Generator_ssl(x,y,x_vb,y_vb,alfa=0.3,use_sometimes=False,epoch_size=300):
    while(1):
        ind1=np.random.permutation(x.shape[0])
        ind2=np.random.permutation(x.shape[0])
        ind3=np.random.permutation(x_vb.shape[0])
        for n in range(0,x.shape[0]):
            #mixup
            if(use_sometimes):
                if(random.random()<0.5):
                    a=np.random.beta(alfa,alfa)
                else:
                    a=1
            else:
                a=np.random.beta(alfa,alfa)
            x2=a*x[ind1[n]]+(1-a)*x[ind2[n]]
            y2=a*y[ind1[n]]+(1-a)*y[ind2[n]]
            x2vb=x_vb[ind3[n]]
            y2vb=y_vb[ind3[n]]
            x2=np.reshape(x2,(1,x.shape[1],x.shape[2],x.shape[3],x.shape[4]))
            y2=np.reshape(y2,(1,y.shape[1],y.shape[2],y.shape[3],y.shape[4]))
            x2vb=np.reshape(x2vb,(1,x.shape[1],x.shape[2],x.shape[3],x.shape[4]))
            y2vb=np.reshape(y2vb,(1,y.shape[1],y.shape[2],y.shape[3],y.shape[4]))
            x3=np.concatenate((x2,x2vb),axis=0)
            y3=np.concatenate((y2,y2vb),axis=0)
            yield x3,y3



def NO_MixUp_Generator_ssl(x,y,x_vb,y_vb):
    while(1):
        ind1=np.random.permutation(x.shape[0])
        #ind2=np.random.permutation(x.shape[0])
        ind2=np.random.permutation(x_vb.shape[0])     #[:x.shape[0]]
        for n in range(0,x.shape[0]):
            x2=x[ind1[n]]
            x2vb=x_vb[ind2[n]]
            y2=y[ind1[n]]
            y2vb=y_vb[ind2[n]]

            x2=np.reshape(x2,(1,x.shape[1],x.shape[2],x.shape[3],x.shape[4]))
            y2=np.reshape(y2,(1,y.shape[1],y.shape[2],y.shape[3],y.shape[4]))
            x2vb=np.reshape(x2vb,(1,x.shape[1],x.shape[2],x.shape[3],x.shape[4]))
            y2vb=np.reshape(y2vb,(1,y.shape[1],y.shape[2],y.shape[3],y.shape[4]))

            x3=np.concatenate((x2,x2vb),axis=0)
            y3=np.concatenate((y2,y2vb),axis=0)
            yield x3,y3




def Hybrid_MixUp_Generator(x,y,alfa):
    while(1):
        ind1=np.random.permutation(x.shape[0])
        ind2=np.random.permutation(x.shape[0])
        for n in range(0,x.shape[0]):
            #mixup
            noise=np.random.uniform(0,1,x.shape[1:4])
            #noise = scipy.ndimage.filters.gaussian_filter(noise, 0.05)
            noise_label = np.zeros(y.shape[1:5])
            noise_image = np.zeros(x.shape[1:5])
            for nc in range(0,y.shape[4]):
                noise_label[:,:,:,nc] = noise
            for ch in range(0,x.shape[4]):
                noise_image[:,:,:,ch] = noise

            x2=noise_image*x[ind1[n]]+(1-noise_image)*x[ind2[n]]
            y2=noise_label*y[ind1[n]]+(1-noise_label)*y[ind2[n]]
            x2=np.reshape(x2,(1,x.shape[1],x.shape[2],x.shape[3],x.shape[4]))
            y2=np.reshape(y2,(1,y.shape[1],y.shape[2],y.shape[3],y.shape[4]))
            yield x2,y2

def TTDO(X,model,n_iter=2):
    #test time Data Augmentation
    f = K.function([model.layers[0].input, K.learning_phase()],[model.layers[-1].output])
    out_shape=model.layers[-1].output.get_shape()
    #result = np.zeros(out_shape)
    result = np.zeros([X.shape[0], X.shape[1], X.shape[2], X.shape[3], out_shape[4]])
    for ii in range(n_iter):
        a=f((X,1))
        result_iter = np.asarray(a)
        result_iter = np.reshape(result_iter, (result_iter.shape[1:6]))
        result += result_iter
    result /= n_iter
    return result

def TTDAO(X,model,n_iter=2):
    #test time Data Augmentation
    f = K.function([model.layers[0].input, K.learning_phase()],[model.layers[-1].output])
    out_shape=model.layers[-1].output.get_shape()
    #result = np.zeros(out_shape)
    result = np.zeros([X.shape[0], X.shape[1], X.shape[2], X.shape[3], out_shape[4]])
    for ii in range(n_iter):
        Xn=X+np.random.normal(0,0.05,X.shape)
        a=f((Xn,0))
        result_iter = np.asarray(a)
        result_iter = np.reshape(result_iter, (result_iter.shape[1:6]))
        result += result_iter
    result /= n_iter
    return result



def TTDAO_with_uncertainty(X,model,n_iter=10,TASK='0'):
    #test time Data Augmentation
    if(TASK=='9'):
        f = K.function([model.layers[0].input, K.learning_phase()],[model.layers[-2].output])
        out_shape=model.layers[-2].output.get_shape()
    else:
        f = K.function([model.layers[0].input, K.learning_phase()],[model.layers[-1].output])
        out_shape=model.layers[-1].output.get_shape()
    #result = np.zeros(out_shape)
    result = np.zeros([X.shape[0], X.shape[1], X.shape[2], X.shape[3], out_shape[4]])
    for ii in range(n_iter):
        Xn=X+np.random.normal(0,0.1,X.shape)
        #a=f((X,0))
        #a=f((Xn,0))
        a=f((Xn,1))
        result_iter = np.asarray(a)
        #result_iter = np.reshape(result_iter, (result_iter.shape[1:6]))
        result_iter = result_iter[0,...]
        #print(type(result_iter))
        #print(result_iter.shape)
        result += result_iter
    result /= n_iter
    u= - result* np.log(result+ 1e-5)
    uncertainty_map= np.sum(u, axis= 4)
    uncertainty_map= np.expand_dims(uncertainty_map,axis=4)
    uncertainty_map= np.exp(-1.5*uncertainty_map)
    #print(uncertainty_map.max())
    #print(uncertainty_map.min())
    #print(np.where(uncertainty_map<0.9)[0].shape)
    #uncertainty_map= np.exp(-5*uncertainty_map)
    return result * uncertainty_map
