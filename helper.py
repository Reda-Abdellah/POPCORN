import os, glob, copy, json, shutil, random, subprocess, csv
import numpy as np
import nibabel as nii
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from scipy import signal, stats, ndimage
import statsmodels.api as sm
from scipy.signal import argrelextrema
from collections import OrderedDict
from tqdm import tqdm
import losses
from scipy.ndimage.morphology import binary_erosion,  binary_dilation, binary_opening, binary_closing
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.augmentations.spatial_transformations import augment_spatial
from PIL import Image
from skimage import measure
#from test_utils import seg_majvote_times_decoder_with_FMs, seg_majvote_times
import torchio as tio


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_img(img, name):
    Image.fromarray(((img-img.min())/(img.max()-img.min())*255).astype(np.uint8), 'L').save(name)

def IQDA_generator(x,y):
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
                    x_[:,:,:,j]=ndimage.uniform_filter(x_[:,:,:,j], (1,1,3))

            if(op==4):
                for j in range(x_.shape[3]):
                    x_[:,:,:,j] = ndimage.uniform_filter(x_[:,:,:,j], (3,3,3))

            if(op==5):
                for j in range(x_.shape[3]):
                    x_[:,:,:,j] =ndimage.uniform_filter(x_[:,:,:,j], (1,1,2))

            x_=np.expand_dims(x_, axis=0)
            y_=np.expand_dims(y_, axis=0)


            yield x_,y_

def IQDA(x_):
    for i in range(0,x_.shape[0]):
        op=np.random.choice(5,1,p=[0.15,0.25,0.2,0.2,0.2]) # 0:nada, 1:sharp, 2:blur, 3: axial blur 3, 4: axial blur 5, 5: axial blur 2
        for j in range(x_.shape[-1]):
            if(op==1):
                x_[i,:,:,:,j] = 2*x_[i,:,:,:,j]-ndimage.uniform_filter(x_[i,:,:,:,j], (3,3,3))
            if(op==2):
                x_[i,:,:,:,j] = ndimage.uniform_filter(x_[i,:,:,:,j], (3,3,3))
            if(op==3):
                x_[i,:,:,:,j]=ndimage.uniform_filter(x_[i,:,:,:,j], (1,1,3))
            if(op==4):
                x_[i,:,:,:,j] =ndimage.uniform_filter(x_[i,:,:,:,j], (1,1,2))
        return x_

def IQDA2D(x_):
    for i in range(0,x_.shape[0]):
        op=np.random.choice(3,1) # 0:nada, 1:sharp, 2:blur, 3: axial blur 3, 4: axial blur 5, 5: axial blur 2
        for j in range(x_.shape[-1]):
            if(op==1):
                x_[i,:,:,j] = 2*x_[i,:,:,j]-ndimage.uniform_filter(x_[i,:,:,j], (3,3))
            if(op==2):
                x_[i,:,:,j] = ndimage.uniform_filter(x_[i,:,:,j], (3,3))

        return x_

def IQDA2D_v2(x_, op=None):
    if(op==None):
        op=np.random.choice(3,1 ,p=[0.1,0.45,0.45]) # 0:nada, 1:sharp, 2:blur, 3: axial blur 3, 4: axial blur 5, 5: axial blur 2
    std=[np.random.uniform(0.25,0.6),np.random.uniform(0.25,0.6)]
    alpha= np.random.uniform(1,3)

    for j in range(x_.shape[-1]):
        if(op==1):
            x_[...,j] =  x_[...,j]+ alpha*(x_[...,j]-ndimage.gaussian_filter(x_[...,j], std))
        if(op==2):
            x_[...,j] = ndimage.gaussian_filter(x_[...,j], std)
    return x_

def IQDA_v2(x):
    x_=np.copy(x)
    op=np.random.choice(4,1,p=[0.1,0.3,0.3,0.3])
    for j in range(x_.shape[-1]):
        if(op==1):
            std=[np.random.uniform(0.25,1),np.random.uniform(0.25,1),np.random.uniform(0.25,1)]
            alpha= np.random.uniform(1,5)
            x_[:,:,:,j] = x_[:,:,:,j]+ alpha*(x_[:,:,:,j]-ndimage.gaussian_filter(x_[:,:,:,j], std))
        if(op==2):
            std=[np.random.uniform(0.25,1),np.random.uniform(0.25,1),np.random.uniform(0.25,1)]
            x_[:,:,:,j] = ndimage.gaussian_filter(x_[:,:,:,j], std)
        if(op==3):
            ax=int(np.random.uniform(1,6))
            x_[:,:,:,j]=ndimage.uniform_filter(x_[:,:,:,j], (1,1,ax))
    return x_

def rot_90(a):
    op=np.random.choice(10,1)
    #print(op)
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
    return a

def batch_rot90(lesion_batch):
    for i in range(lesion_batch.shape[0]):
        lesion_batch[i]= rot_90(lesion_batch[i]) 
    return lesion_batch

#dataset_class
class TileDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self,files_dir,tiles_to_consider=None, transform=None, da=False, nbNN=[5,5,5], image_size=[128,128,128]):
        self.img_size=image_size
        if(tiles_to_consider is None):
            self.x_list= sorted(glob.glob(files_dir+"x*.npy"))
        else:
            self.x_list=[]
            for tile in tiles_to_consider:
                [self.x_list.append(name) for name in sorted(glob.glob(files_dir+"x*tile"+str(tile)+".npy"))]
            
        self.transform = transform
        self.random_idx= np.arange(len(self.x_list))
        np.random.shuffle(self.random_idx)
        self.da=da

        n=0
        a=np.zeros((nbNN[0],nbNN[1],nbNN[2]))
        for x in range(nbNN[0]):
            for y in range(nbNN[1]):
                for z in range(nbNN[2]):
                    a[x,y,z]= n
                    n=n+1

        self.train_files_bytiles={}
        for stepx in range(1,nbNN[0]-1):
            for stepy in range(1,nbNN[1]-1):
                for stepz in range(1,nbNN[2]-1):
                    tile_num= str(int(a[stepx,stepy,stepz]))
                    self.train_files_bytiles[tile_num]= sorted(glob.glob(files_dir+"x*tile_"+tile_num+".npy"))

    def __len__(self):
        #return 8
        return len(self.x_list)
    
    def normalize_input_minmax(self,input_ ):
        mi=np.min(input_)
        ma=np.max(input_)
        #print(mi, ma)
        if((ma-mi)==0):
            ma=1
            mi=0
        input_=(input_-mi)/(ma-mi)
        return input_

    def get_random_patch(self, x1,y1, size=[128,128,128]):
        x_ran= np.random.randint( size[0]//2, x1.shape[1]- (size[0]-size[0]//2) +1 )
        y_ran= np.random.randint( size[1]//2, x1.shape[2]- (size[1]-size[1]//2) +1 )
        z_ran= np.random.randint( size[2]//2, x1.shape[3]- (size[2]-size[2]//2) +1 )
        xo=x_ran- size[0]//2
        yo=y_ran- size[1]//2
        zo=z_ran- size[2]//2
        return x1[:,xo: xo+size[0], yo: yo+size[1],zo: zo+size[2],:],y1[:,xo: xo+size[0], yo: yo+size[1],zo: zo+size[2],:]
    
    def get_pair(self, idx):
        x1_name=self.x_list[idx]
        y1_name=x1_name.replace('x_','y_')
        x1=np.load(x1_name).astype('float')
        y1=np.load(y1_name).astype('float')
        
        x1= self.normalize_input_minmax(x1 )

        if(not x1.shape==self.img_size):
            x1,y1=self.get_random_patch(x1,y1, size=self.img_size)
        return x1,y1

    def Mixup(self,x1,x2,y1,y2,alfa=0.3):
        a=np.random.beta(alfa,alfa)
        x=a*x1+(1-a)*x2
        y=a*y1+(1-a)*y2
        return x,y

    def iqda(self, x, y):
        x=IQDA(x)
        return x,y

    def iqda_v2(self, x, y):
        x[0,:,:,:,:]=IQDA_v2(x[0,:,:,:,:])
        return x,y

    def rot(self, x, y):
        X=np.concatenate((x,y), axis=-1)
        X=batch_rot90(X)
        x=X[:,:,:,:,0:x.shape[-1]]
        y=X[:,:,:,:,x.shape[-1]:x.shape[-1]+y.shape[-1]]
        return x,y
    
    def rot4(self, x, y, z, h):
        X=np.concatenate((x,y,z,h), axis=-1)
        X=batch_rot90(X)
        x=X[:,:,:,:,0:x.shape[-1]]
        y=X[:,:,:,:,x.shape[-1]:x.shape[-1]+y.shape[-1]]
        z=X[:,:,:,:,x.shape[-1]+y.shape[-1]:x.shape[-1]+y.shape[-1]+z.shape[-1]]
        h=X[:,:,:,:,x.shape[-1]+y.shape[-1]+z.shape[-1]:]
        return x,y,z,h

    def __getitem__(self, idx):
        #print(idx)
        if torch.is_tensor(idx):
            idx = idx.tolist()

        np.random.shuffle(self.random_idx)
        x1_name=self.x_list[idx]
        y1_name=x1_name.replace('x_','y_')
        x1=np.load(x1_name)
        y1=np.load(y1_name)


        if(self.da=="rot"):
            inputs,label= self.rot( x1, y1)

        if(self.da=="iqda"):
            inputs,label= self.iqda( x1, y1)

        if(self.da=="iqda_v2"):
            inputs,label= self.iqda_v2( x1, y1)

        elif(self.da=="mixup"):
            ind_ran=self.random_idx[idx]
            x2_name=self.x_list[ind_ran]
            y2_name=x2_name.replace('x_','y_')
            x2=np.load(x2_name)
            y2=np.load(y2_name)
            inputs,label=self.Mixup(x1,x2,y1,y2)

        else:
            inputs=x1
            label=y1
        
        sample = {'inputs': inputs, 'label': label}
        if self.transform:
            sample=self.transform(sample)
        return sample

class TileDataset_with_reg(TileDataset):
    """Face Landmarks dataset."""
    def __init__(self,files_dir,tiles_to_consider=None, transform=None, da=False, nbNN=[5,5,5]):
        super().__init__(files_dir,tiles_to_consider, transform, da, nbNN)
        if(tiles_to_consider is None):
            self.x_list= sorted(glob.glob(files_dir+"x*.npy"))
        else:
            self.x_list=[]
            for tile in tiles_to_consider:
                [self.x_list.append(name) for name in sorted(glob.glob(files_dir+"x*tile"+str(tile)+".npy"))]
            
        self.transform = transform
        self.random_idx= np.arange(len(self.x_list))
        np.random.shuffle(self.random_idx)
        self.da=da

        n=0
        a=np.zeros((nbNN[0],nbNN[1],nbNN[2]))
        for x in range(nbNN[0]):
            for y in range(nbNN[1]):
                for z in range(nbNN[2]):
                    a[x,y,z]= n
                    n=n+1

        self.train_files_bytiles={}
        for stepx in range(1,nbNN[0]-1):
            for stepy in range(1,nbNN[1]-1):
                for stepz in range(1,nbNN[2]-1):
                    tile_num= str(int(a[stepx,stepy,stepz]))
                    self.train_files_bytiles[tile_num]= sorted(glob.glob(files_dir+"x*tile_"+tile_num+".npy"))
        #print(self.train_files_bytiles.keys())

    def Mixup_it(self,x1,x2,y1,y2, x1_,x2_,y1_,y2_, sim,sim_, alfa=0.3 ):
        a=np.random.beta(alfa,alfa)
        x1=a*x1+(1-a)*x1_
        y1=a*y1+(1-a)*y1_
        x2=a*x2+(1-a)*x2_
        y2=a*y2+(1-a)*y2_
        sim= a*sim+(1-a)*sim_
        return x1,x2,y1,y2,sim
        
    def gen_iqda_2it(self,idx,same_ratio=0.8,sim='output_diff'):
        op=np.random.choice(2,1,p=[same_ratio,1-same_ratio]) # 0 same 1 different
        x1, y1= self.get_pair(idx)
        if(op==0):        
            inter_sim=0.0
            y2, x2 = np.copy(y1), np.copy(x1) 
        else:
            x1, y1= self.get_pair(idx)
            ind_ran=self.random_idx[idx]
            x2,y2= self.get_pair(ind_ran)
            inter_sim= np.exp(-np.mean(np.square(y1-y2)))    
        return x1, x2, y1, y2 ,inter_sim

    def __getitem__(self, idx):
        #print(idx)
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x1, x2, y1, y2 ,inter_sim= self.gen_iqda_2it(idx)


        if(self.da=="rot"):
            x1, y1, x2, y2= self.rot4( x1, y1, x2, y2)

        elif(self.da=="iqda"):
            x1, y1= self.iqda( x1, y1)
            x2, y2= self.iqda( x2, y2)
            x1, y1, x2, y2= self.rot4( x1, y1, x2, y2)


        elif(self.da=="iqda_v2"):
            x1, y1= self.iqda_v2( x1, y1)
            x2, y2= self.iqda_v2( x2, y2)
            x1, y1, x2, y2= self.rot4( x1, y1, x2, y2)

        elif(self.da=="mixup"):
            ind_ran=self.random_idx[idx]
            x1_, x2_, y1_, y2_ ,inter_sim_= self.gen_iqda_2it(ind_ran)
            x1,x2,y1,y2,inter_sim =self.Mixup_it(x1,x2,y1,y2, x1_,x2_,y1_,y2_, inter_sim,inter_sim_)
            x1, y1, x2, y2= self.rot4( x1, y1, x2, y2)

        #"""
        nii.Nifti1Image(np.argmax(y1[0,:,:,:,:], axis=-1).astype('uint8'), np.eye(4,4) ).to_filename('y1.nii.gz')
        nii.Nifti1Image(x1[0,:,:,:,0], np.eye(4,4) ).to_filename('x1.nii.gz')
        #nii.Nifti1Image(inputs[0,:,:,:,0], np.eye(4,4) ).to_filename('in1.nii.gz')
        end
        #"""
        
        sample1 = {'inputs': x1, 'label': y1}
        sample2 = {'inputs': x2, 'label': y2}
        if self.transform:
            sample1=self.transform(sample1)
            sample2=self.transform(sample2)
        inter_sim= torch.tensor(inter_sim).float().to(device)
        #return sample1, sample2, inter_sim
        return {'inputs1': sample1['inputs'], 'label1': sample1['label'],
                'inputs2': sample2['inputs'], 'label2': sample2['label'],
                'inter_sim': inter_sim}

class da_v3_abstract(object):
    def __init__(self):

        gaussian_blur= tio.RandomBlur(std= (0.25, 1))
        edge_enhancement= tio.Lambda(lambda x: x+ np.random.uniform(1,5)*(x- gaussian_blur(x) ))
        rot_flip= tio.Lambda(lambda x: torch.from_numpy( rot_90(x.numpy().transpose(1,2,3,0) ).transpose(3,0,1,2).copy()   )  )
        distortion= tio.Lambda(lambda x: torch.from_numpy( ndimage.uniform_filter(x.numpy(), (1,1,int(np.random.uniform(1,6)),1)).copy()))
        
        self.spatial_transform = tio.Compose([
                    rot_flip,
                    tio.OneOf({                                # either
                        tio.RandomAffine(scales=(1,1.3),degrees=5, default_pad_value='otsu' ): 0.4,               # random affine
                        tio.RandomElasticDeformation(max_displacement=4): 0.6,   # or random elastic deformation
                    }, p=0),                                 # applied to 80% of images

                    ])
        #self.spatial_transform =tio.RandomElasticDeformation(max_displacement=4, p=0.8)

        
        #"""
        self.other_transforms = tio.Compose([

                    tio.RandomBiasField(p=0.5),                # magnetic field inhomogeneity 30% of times
                    
                    tio.OneOf({ 
                        tio.RandomAnisotropy(downsampling = (1.5, 4)):1,              # make images look anisotropic 25% of times
                        distortion:1,
                        tio.RandomBlur(std= (0.25, 1, 0.25, 1, 0.25, 1)):1 ,                    # blur 25% of times
                        edge_enhancement:1,
                    }, p=0.7 ),
                    
                    tio.OneOf({                                # either
                        tio.RandomMotion(degrees=5 , translation= 4): 1,                 # random motion artifact
                        tio.RandomSpike(intensity=(0.1,0.15) ): 1,                  # or spikes
                        tio.RandomGhosting(num_ghosts=(4,10), intensity=(0.25,0.75)): 1,               # or ghosts
                    }, p=0.5),                                 # applied to 50% of images

                    
                    tio.RandomNoise( std=(0, 0.1), p=0.5),                   # Gaussian noise 25% of times
                    
                    ])
        """
        self.other_transforms = tio.Compose([
                    tio.OneOf({
                        tio.RandomBiasField():1,                # magnetic field inhomogeneity 30% of times 
                        tio.RandomAnisotropy(downsampling = (1.5, 4)):1,              # make images look anisotropic 25% of times
                        distortion:1,
                        tio.RandomBlur(std= (0.25, 1, 0.25, 1, 0.25, 1)):1 ,                    # blur 25% of times
                        edge_enhancement:1,
                        tio.RandomMotion(degrees=5 , translation= 4): 1,                 # random motion artifact
                        tio.RandomSpike(intensity=(0.1,0.15) ): 1,                  # or spikes
                        tio.RandomGhosting(num_ghosts=(4,10), intensity=(0.25,0.75)): 1,               # or ghosts
                        tio.RandomNoise( std=(0, 0.1)):1,                   # Gaussian noise 25% of times
                    }),                                 # applied to 50% of images
                    ])
        """

class da_v3(da_v3_abstract):
    def __init__(self):
        super().__init__()

    def __call__(self, sample):
        inputs, label = sample['inputs'].transpose((0,4,1,2,3)),sample['label'].transpose((0,4,1,2,3))

        X=np.concatenate((inputs,label), axis=1)
        #X=np.pad(X, pad_width= ( (0,0), (0,0),(8,8) ,(8,8) ,(8,8) ), mode='symmetric')
        X= self.spatial_transform(X[0])
        #X=X[:,8:-8, 8:-8, 8:-8]
        X=np.expand_dims(X, axis=0)
        inputs, label= X[:,0:2,:,:,:], X[:,2:4,:,:,:]

        inputs[0,0:1,:,:,:]= self.other_transforms(inputs[0,0:1,:,:,:])
        inputs[0,1:2,:,:,:]= self.other_transforms(inputs[0,1:2,:,:,:])

        """
        nii.Nifti1Image(label[0,1,:,:,:].astype('uint8'), np.eye(4,4) ).to_filename('label.nii.gz')
        nii.Nifti1Image(label[0,1,:,:,:], np.eye(4,4) ).to_filename('bg.nii.gz')
        nii.Nifti1Image(inputs[0,1,:,:,:], np.eye(4,4) ).to_filename('in2.nii.gz')
        nii.Nifti1Image(inputs[0,0,:,:,:], np.eye(4,4) ).to_filename('in1.nii.gz')
        end
        #"""
        
        return {'inputs': inputs.transpose((0,2,3,4,1)), 'label': label.transpose((0,2,3,4,1))}

class spatial_transform(object):
    def __init__(self, patch_size, patch_center_dist_from_border=30,
                 do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                 do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                 do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                 border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=False, data_key="x1",
                 label_key="seg", p_el_per_sample=1, p_scale_per_sample=1, p_rot_per_sample=1,
                 independent_scale_for_each_axis=False, p_rot_per_axis:float=1, p_independent_scale_per_axis: int=1):
        self.independent_scale_for_each_axis = independent_scale_for_each_axis
        self.p_rot_per_sample = p_rot_per_sample
        self.p_scale_per_sample = p_scale_per_sample
        self.p_el_per_sample = p_el_per_sample
        self.data_key = data_key
        self.label_key = label_key
        self.patch_size = patch_size
        self.patch_center_dist_from_border = patch_center_dist_from_border
        self.do_elastic_deform = do_elastic_deform
        self.alpha = alpha
        self.sigma = sigma
        self.do_rotation = do_rotation
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.do_scale = do_scale
        self.scale = scale
        self.border_mode_data = border_mode_data
        self.border_cval_data = border_cval_data
        self.order_data = order_data
        self.border_mode_seg = border_mode_seg
        self.border_cval_seg = border_cval_seg
        self.order_seg = order_seg
        self.random_crop = random_crop
        self.p_rot_per_axis = p_rot_per_axis
        self.p_independent_scale_per_axis = p_independent_scale_per_axis

    def __call__(self, sample):
        x1, x2, label = sample['x1'],sample['x2'], sample['label']
        x1=x1.transpose((0,4,1,2,3))
        ret_val = augment_spatial(x1, seg=None, patch_size=self.patch_size,
                                  patch_center_dist_from_border=self.patch_center_dist_from_border,
                                  do_elastic_deform=self.do_elastic_deform,
                                  alpha=self.alpha, sigma=self.sigma,
                                  do_rotation=self.do_rotation, angle_x=self.angle_x, angle_y=self.angle_y,
                                  angle_z=self.angle_z, do_scale=self.do_scale, scale=self.scale,
                                  border_mode_data=self.border_mode_data,
                                  border_cval_data=self.border_cval_data, order_data=self.order_data,
                                  border_mode_seg=self.border_mode_seg, border_cval_seg=self.border_cval_seg,
                                  order_seg=self.order_seg, random_crop=self.random_crop,
                                  p_el_per_sample=self.p_el_per_sample, p_scale_per_sample=self.p_scale_per_sample,
                                  p_rot_per_sample=self.p_rot_per_sample,
                                  independent_scale_for_each_axis=self.independent_scale_for_each_axis,
                                  p_rot_per_axis=self.p_rot_per_axis,
                                  p_independent_scale_per_axis=self.p_independent_scale_per_axis)
        x2=x2.transpose((0,4,1,2,3))
        label=label.transpose((0,4,1,2,3))
        ret_val_ = augment_spatial(x2, seg=label, patch_size=self.patch_size,
                                  patch_center_dist_from_border=self.patch_center_dist_from_border,
                                  do_elastic_deform=self.do_elastic_deform,
                                  alpha=self.alpha, sigma=self.sigma,
                                  do_rotation=self.do_rotation, angle_x=self.angle_x, angle_y=self.angle_y,
                                  angle_z=self.angle_z, do_scale=self.do_scale, scale=self.scale,
                                  border_mode_data=self.border_mode_data,
                                  border_cval_data=self.border_cval_data, order_data=self.order_data,
                                  border_mode_seg=self.border_mode_seg, border_cval_seg=self.border_cval_seg,
                                  order_seg=self.order_seg, random_crop=self.random_crop,
                                  p_el_per_sample=self.p_el_per_sample, p_scale_per_sample=self.p_scale_per_sample,
                                  p_rot_per_sample=self.p_rot_per_sample,
                                  independent_scale_for_each_axis=self.independent_scale_for_each_axis,
                                  p_rot_per_axis=self.p_rot_per_axis,
                                  p_independent_scale_per_axis=self.p_independent_scale_per_axis)
        
        return {'x1': ret_val[0].transpose((0,2,3,4,1)),'x2': ret_val_[0].transpose((0,2,3,4,1)), 'label': ret_val_[1].transpose((0,2,3,4,1))}

class combine(object):
    def __call__(self, sample):
        x1, x2, label = sample['x1'],sample['x2'], sample['label']
        return {
        'inputs': np.concatenate((x1,x2), axis=-1),
        'label': label
            }

class augment_gt(object):
    def __init__(self, p_keep_same=0.1):
        print('augment_gt')
        self.p_keep_same= p_keep_same

    def __call__(self, sample):
        label =  sample['label']
        """
        nii.Nifti1Image(label[0,:,:,:,1].astype('uint8'), np.eye(4,4) ).to_filename('label.nii.gz')
        nii.Nifti1Image( sample['inputs'][0,:,:,:,1], np.eye(4,4) ).to_filename('in2.nii.gz')
        nii.Nifti1Image( sample['inputs'][0,:,:,:,0], np.eye(4,4) ).to_filename('in1.nii.gz')
        #end
        #"""
        new_label= augment_segmentation(label, self.p_keep_same)
        #nii.Nifti1Image(new_label[0,:,:,:,1].astype('uint8'), np.eye(4,4) ).to_filename('lab_after.nii.gz')
        #end
        sample['label']= new_label
        return sample

class ToTensor(object):
    def __call__(self, sample):
        out={}
        for key in sample.keys():
            out[key] = torch.from_numpy(sample[key][0].transpose((3,0,1,2))).float().to(device)
            #out[key] = torch.from_numpy(sample[key][0].transpose((3,0,1,2))).double().to(device)
        return out

class Std_normalize(object):
    def __call__(self, sample):
        channels=sample['inputs'].shape[-1]
        inp_mean=sample['inputs'].mean(axis=(0,1,2,3)).reshape(1,1,1,1,channels)
        inp_std=sample['inputs'].std(axis=(0,1,2,3)).reshape(1,1,1,1,channels)
        return {
        'inputs': (sample['inputs']-inp_mean)/inp_std,
        'label': sample['label']
            }

def init_weights(m):
    if type(m) == nn.Conv3d:
        print('init')
        nn.init.xavier_uniform(m.weight)
        #nn.init.xavier_uniform(m.bias)
        m.bias.data.fill_(0.001)

def train_model(model,optimizer,criterion,val_criterion, Epoch,dataset_loader,
                dataset_loader_val,eval_strategy,out_PATH,best_val_loss=100,early_stop=False,regularized=True,  loss_weights=[1,0.01],
                early_stop_treshold=100):
    count=0
    for epoch in range(Epoch):
        running_loss = 0.0
        model.train()
        k=0
        with tqdm(dataset_loader) as tepoch:
            for sample in tepoch:
                k=k+1
                optimizer.zero_grad()
                if(regularized):
                    inputs1, labels1 =sample['inputs1'].to(device) , sample['label1'].to(device)
                    inputs2, labels2 =sample['inputs2'].to(device) , sample['label2'].to(device)

                    bottleneck1,x3,x2,x1 = model.encoder(inputs1)
                    pred1=model.decoder(bottleneck1,x3,x2,x1)
                    bottleneck2,x3,x2,x1 = model.encoder(inputs2)
                    pred2=model.decoder(bottleneck2,x3,x2,x1)

                    latent_distance= 2*torch.sum(torch.square(bottleneck1-bottleneck2), dim=(1,2,3,4))/(torch.mean(torch.square(bottleneck1), dim=(1,2,3,4))+torch.mean(torch.square(bottleneck2), dim=(1,2,3,4)))
                    consistency=  torch.mean(latent_distance* torch.exp(-sample['inter_sim']))
                    loss_supervised= criterion(pred1, labels1)+criterion(pred2, labels2)
                    loss = loss_supervised *loss_weights[0]+ consistency*loss_weights[1]
                    tepoch.set_postfix(supervised_loss=loss_supervised.item(), regularization=consistency.item())

                else:
                    inputs, labels =sample['inputs'].to(device) , sample['label'].to(device)
                    outputs= model(inputs)
                    loss = criterion(outputs, labels)
                    tepoch.set_postfix(loss=loss.item(), Dice=100. * (1-loss.item()))

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                #break

            running_loss_val =0.0
            model.eval()
            j=0

        for sample in tqdm(dataset_loader_val):
            j=j+1
            inputs, labels =sample['inputs'].to(device) , sample['label'].to(device)
            with torch.no_grad():
                outputs = model(inputs)
            loss = val_criterion(outputs, labels)
            running_loss_val += loss.item()
        #res
        val_loss = running_loss_val/(j+1)
        
        train_loss= running_loss/(k+1)
        print('Epoch: '+str(epoch)+'   Loss: '+str(train_loss)+'   Val_loss: '+str(val_loss))

        if(val_loss<best_val_loss):
            #savingmodel
            count=0
            torch.save(model, out_PATH)
            best_val_loss=val_loss
            print('saving weights. new best loss: '+str(best_val_loss))
        else:
            count=count+1
            if(early_stop):
                if(count>early_stop_treshold):
                    print('val loss did not improve for:'+str(early_stop_treshold)+' epochs. Stopping training.')
                    break
    return best_val_loss

def moving_average_weights(running_model,stable_model,changing_ratio = 0.1):
    # update the weigts of a stable model with the running model weiths and ratio of changing.
    # then load updated weights to running model
    for param_tensor in running_model.state_dict():
        new_weights= changing_ratio* running_model.state_dict()[param_tensor] + (1-changing_ratio)* stable_model.state_dict()[param_tensor]
        running_model.state_dict()[param_tensor].copy_( new_weights )

def TransferWeights(new_model,old_model,stop_layer=None):
    for param_tensor in old_model.state_dict():
        print(param_tensor)
        if(stop_layer in param_tensor):
            print('stop layer: '+param_tensor)
            break
        new_model.state_dict()[param_tensor].copy_(old_model.state_dict()[param_tensor])

def freezeLayers(model,stop_layer=None):
    for layer_name, param in model.named_parameters():
        if(stop_layer in layer_name):
            print('stop layer: '+layer_name)
            break
        param.requires_grad=False
        print('layer freezed: '+layer_name)

def unfreezeLayers(model,stop_layer='Nonoe'):
    for layer_name, param in model.named_parameters():
        if(stop_layer in layer_name):
            print('stop layer: '+layer_name)
            break
        param.requires_grad=True

def verifyTrainable(model):
    for layer_name, param in model.named_parameters():
        print(layer_name)
        print(param.requires_grad)

def verifyTransferWeights(new_model,old_model,stop_layer=None):
    for param_tensor in old_model.state_dict():
        print(param_tensor)
        if(param_tensor==stop_layer):
            break
        if(new_model.state_dict()[param_tensor]==old_model.state_dict()[param_tensor]).all():
            print('good')
        else:
            print('bad')

def keyword_toList(path,keyword):
    search=os.path.join(path,'*'+keyword+'*')
    lista=sorted(glob.glob(search))
    print("list contains: "+str( len(lista))+" elements")
    return lista

def split_dataset(x_train,split_ratio=0.8,seed=43):
    np.random.seed(seed)
    random_idx= np.arange(x_train.shape[0])
    np.random.shuffle(random_idx)
    split=int(split_ratio*x_train.shape[0])
    x_test=x_train[split:]
    x_train=x_train[:split]
    return x_train,x_test

def kde_normalization(vol, contrast):
    # copied from FLEXCONN
    # slightly changed to fit our implementation
    temp = vol[np.nonzero(vol)].astype(float)
    q = np.percentile(temp, 99)
    temp = temp[temp <= q]
    temp = temp.reshape(-1, 1)
    bw = q / 80
    # print("99th quantile is %.4f, gridsize = %.4f" % (q, bw))

    kde = sm.nonparametric.KDEUnivariate(temp)

    kde.fit(kernel='gau', bw=bw, gridsize=80, fft=True)
    x_mat = 100.0 * kde.density
    y_mat = kde.support

    indx = argrelextrema(x_mat, np.greater)
    indx = np.asarray(indx, dtype=int)
    heights = x_mat[indx][0]
    peaks = y_mat[indx][0]
    peak = 0.00
    # print("%d peaks found." % (len(peaks)))

    # norm_vol = vol
    if contrast.lower() in ["t1", "mprage"]:
        peak = peaks[-1]
        # print("Peak found at %.4f for %s" % (peak, contrast))
        # norm_vol = vol/peak
        # norm_vol[norm_vol > 1.25] = 1.25
        # norm_vol = norm_vol/1.25
    elif contrast.lower() in ['t2', 'pd', 'flair', 'fl']:
        peak_height = np.amax(heights)
        idx = np.where(heights == peak_height)
        peak = peaks[idx]
        # print("Peak found at %.4f for %s" % (peak, contrast))
        # norm_vol = vol / peak
        # norm_vol[norm_vol > 3.5] = 3.5
        # norm_vol = norm_vol / 3.5
    else:
        print("Contrast must be either t1,t2,pd, or flair. You entered %s. Returning 0." % contrast)

    # return peak, norm_vol
    return peak

def keep_labels(scale,numfilestemp,lib_path, ii, all_labels,ratioPresence=0.33):
    histogramLabels=np.zeros((all_labels), dtype=int)
    p = int(ratioPresence*numfilestemp)
    for i in range(0,numfilestemp):
        fileLAB=lib_path+"/stack_lab_"+scale+"_"+str(i)+".npy"
        pLABb = np.load(fileLAB,mmap_mode='r')
        pLABb = pLABb[ii]
        lista=np.unique(pLABb)
        for j in range(len(lista)):
            histogramLabels[lista[j]] += 1
    lista = list()
    for i in range(len(histogramLabels)):
        if (histogramLabels[i] > p):
            lista.append(i)
    print("Labels: ", lista)
    print("Nb of labels: ", len(lista))
    return lista

def volume_to_3D_patches(V,nbNN,ps1,ps2,ps3,crop_bg = 4):
    overlap1 = np.floor((nbNN[0]*ps1 - (V.shape[0]-2*crop_bg)) / (nbNN[0]-1))
    offset1 = ps1 - overlap1.astype('int')
    overlap2 = np.floor((nbNN[1]*ps2 - (V.shape[1]-2*crop_bg)) / (nbNN[1]-1))
    offset2 = ps2 - overlap2.astype('int')
    overlap3 = np.floor((nbNN[2]*ps3 - (V.shape[2]-2*crop_bg)) / (nbNN[2]-1))
    offset3 = ps3 - overlap3.astype('int')
    pV=patch_extract_3D(V,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
    return pV

def patches_to_whole_volume(out_shape,pV,nbNN,ps):
    offset1,offset2,offset3,crop_bg= offset_from_patchshape_andnbNN(nbNN,ps,out_shape,crop_bg = 4)
    V=patch_reconstruct_3D(out_shape,pV,nbNN,offset1,offset2,offset3,crop_bg)
    return V

def patch_extract_3D(input,patch_shape,nbNN,offx=1,offy=1,offz=1,crop_bg=0):
    n=0
    numPatches=nbNN[0]*nbNN[1]*nbNN[2]
    local_patch = np.zeros((patch_shape[0],patch_shape[1],patch_shape[2]),input.dtype)
    patches_3D=np.zeros((numPatches,patch_shape[0],patch_shape[1],patch_shape[2]),input.dtype)
    for x in range(crop_bg,(nbNN[0]-1)*offx+crop_bg+1,offx):
        for y in range(crop_bg,(nbNN[1]-1)*offy+crop_bg+1,offy):
            for z in range(0,(nbNN[2]-1)*offz+1,offz): #spine is touching one side in Z dicrection, so crop has to be asymetric
                xx = x+patch_shape[0]
                if xx> input.shape[0]:
                    xx = input.shape[0]
                yy = y+patch_shape[1]
                if yy> input.shape[1]:
                    yy = input.shape[1]
                zz = z+patch_shape[2]
                if zz> input.shape[2]:
                    zz = input.shape[2]
                # To deal with defferent patch size due to border issue
                local_patch = local_patch*0
                local_patch[0:xx-x,0:yy-y,0:zz-z] = input[x:xx,y:yy,z:zz]
                a=np.reshape(local_patch,(1,patches_3D.shape[1],patches_3D.shape[2],patches_3D.shape[3]))
                patches_3D[n,:,:,:]=a
                n=n+1
    patches_3D=patches_3D[0:n,:,:,:]
    return patches_3D

def offset_from_patchshape_andnbNN(nbNN,ps,out_shape,crop_bg = 4):
    [ps1,ps2,ps3]= ps
    overlap1 = np.floor((nbNN[0]*ps1 - (out_shape[0]-2*crop_bg)) / (nbNN[0]-1))
    offset1 = ps1 - overlap1.astype('int')
    overlap2 = np.floor((nbNN[1]*ps2 - (out_shape[1]-2*crop_bg)) / (nbNN[1]-1))
    offset2 = ps2 - overlap2.astype('int')
    overlap3 = np.floor((nbNN[2]*ps3 - (out_shape[2]-2*crop_bg)) / (nbNN[2]-1))
    offset3 = ps3 - overlap3.astype('int')
    return offset1,offset2,offset3,crop_bg

def update_final_seg_with_tile_position(tile_num, final_seg,accumulation_update, out_to_put_in,nbNN,offset1,offset2,offset3,crop_bg ):
    pos,a= get_tile_pos(tile_num,nbNN)
    x= crop_bg+ offset1* pos[0][0]
    y= crop_bg+ offset2* pos[1][0]
    z= offset3* pos[2][0]
    out_shape= final_seg.shape
    xx = x+out_to_put_in.shape[0]
    if xx> out_shape[0]:
        xx = out_shape[0]
    yy = y+out_to_put_in.shape[1]
    if yy> out_shape[1]:
        yy = out_shape[1]
    zz = z+out_to_put_in.shape[2]
    if zz> out_shape[2]:
        zz = out_shape[2]
    final_seg[x:xx,y:yy,z:zz]=final_seg[x:xx,y:yy,z:zz]+ out_to_put_in[0:xx-x,0:yy-y,0:zz-z]
    accumulation_update[x:xx,y:yy,z:zz]=accumulation_update[x:xx,y:yy,z:zz]+1

def patch_reconstruct_3D(out_shape,patches,nbNN,offset1,offset2,offset3,crop_bg):
    n=0
    output=np.zeros(out_shape,patches.dtype)
    acu=np.zeros(out_shape,patches.dtype)
    pesos=np.ones((patches.shape[1],patches.shape[2],patches.shape[3]))
    for x in range(crop_bg,(nbNN[0]-1)*offset1+crop_bg+1,offset1):
        for y in range(crop_bg,(nbNN[1]-1)*offset2+crop_bg+1,offset2):
            for z in range(0,(nbNN[2]-1)*offset3+1,offset3):

                xx = x+patches.shape[1]
                if xx> out_shape[0]:
                    xx = out_shape[0]

                yy = y+patches.shape[2]
                if yy> out_shape[1]:
                    yy = out_shape[1]

                zz = z+patches.shape[3]
                if zz> out_shape[2]:
                    zz = out_shape[2]


                output[x:xx,y:yy,z:zz]=output[x:xx,y:yy,z:zz]+ patches[n]#local_patch[0:xx-x,0:yy-y,0:zz-z]
                acu[x:xx,y:yy,z:zz]=acu[x:xx,y:yy,z:zz]+1
                n=n+1

    ind=np.where(acu==0)
    acu[ind]=1
    output=output/acu
    return output

def normalize_kde_t1(T1,MASK):
    peak = kde_normalization(T1, 't1')
    T1=T1/peak
    return T1

def normalize_mean_std(T1,MASK):
    ind=np.where(MASK>0)
    indbg=np.where(MASK==0)
    T1[indbg] = 0
    m1=np.mean(T1[ind])
    s1=np.std(T1[ind])
    T1=(T1-m1)/s1
    return T1

def remove_folder_if_exist(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print('%s removed' % folder_path)
    else:
        print('%s does not exist' % folder_path)
