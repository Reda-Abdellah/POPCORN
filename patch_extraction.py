#################################################################
#
# DeepLesionBrain: Deep learning for lesion segmentation
#
# Authors: Jose Vicente Manjon Herrera
#          Pierrick Coupe
#
#    Date: 23/05/2019
#
#################################################################

import numpy as np
import random

def patch_extract_3D(input,patch_shape,xstep=1,ystep=1,zstep=1):
	n=0
	offx=patch_shape[0]/xstep
	offy=patch_shape[1]/ystep
	offz=patch_shape[2]/zstep
	numPatches=int((offx*input.shape[0]/(patch_shape[0]))*(offy*input.shape[1]/(patch_shape[1]))*(offz*input.shape[2]/(patch_shape[2])))
	patches_3D=np.zeros((numPatches,patch_shape[0],patch_shape[1],patch_shape[2]),input.dtype)
	for x in range(0,(input.shape[0]-patch_shape[0]+1),xstep):
		for y in range(0,(input.shape[1]-patch_shape[1]+1),ystep):
			for z in range(0,(input.shape[2]-patch_shape[2]+1),zstep):
				a=np.reshape(input[x:x+patch_shape[0],y:y+patch_shape[1],z:z+patch_shape[2]],(1,patches_3D.shape[1],patches_3D.shape[2],patches_3D.shape[3]))
				patches_3D[n,:,:,:]=a
				n=n+1
	patches_3D=patches_3D[0:n,:,:,:]
	return patches_3D


def patch_reconstruct_3D(out_shape,patches,xstep=1,ystep=1,zstep=1):
	n=0
	output=np.zeros(out_shape,patches.dtype)
	acu=np.zeros(out_shape,patches.dtype)
	pesos=np.ones((patches.shape[1],patches.shape[2],patches.shape[3]))
	for x in range(0,(out_shape[0]-patches.shape[1]+1),xstep):
		for y in range(0,(out_shape[1]-patches.shape[2]+1),ystep):
			for z in range(0,(out_shape[2]-patches.shape[3]+1),zstep):
				output[x:x+patches.shape[1],y:y+patches.shape[2],z:z+patches.shape[3]]=output[x:x+patches.shape[1],y:y+patches.shape[2],z:z+patches.shape[3]]+np.reshape(patches[n,:,:,:],(patches.shape[1],patches.shape[2],patches.shape[3]))
				acu[x:x+patches.shape[1],y:y+patches.shape[2],z:z+patches.shape[3]]=acu[x:x+patches.shape[1],y:y+patches.shape[2],z:z+patches.shape[3]]+pesos
				n=n+1
	ind=np.where(acu==0)
	acu[ind]=1
	output=output/acu
	return output

def patch_extract_3D_v2(input,patch_shape,nbNN,offx=1,offy=1,offz=1,crop_bg=0):
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

def patch_reconstruct_3D_v2(out_shape,patches,nbNN, offx=1,offy=1,offz=1,crop_bg=0):
	n=0
	output=np.zeros(out_shape,patches.dtype)
	acu=np.zeros(out_shape,patches.dtype)
	pesos=np.ones((patches.shape[1],patches.shape[2],patches.shape[3]))
	for x in range(crop_background_border,(nbNN-1)*offx+crop_background_border+1,offx):
		for y in range(crop_background_border,(nbNN-1)*offy+crop_background_border+1,offy):
			for z in range(crop_background_border,(nbNN-1)*offz+crop_background_border+1,offz):

				xx = x+patches.shape[1]
				if xx> input.shape[0]:
					xx = input.shape[0]
				yy = y+patches.shape[2]
				if yy> input.shape[1]:
					yy = input.shape[1]
				zz = z+patches.shape[3]
				if zz> input.shape[2]:
					zz = input.shape[2]

				output[x:xx,y:yy,z:zz]=output[x:xx,y:yy,z:zz]+np.reshape(patches[n,:,:,:],(patches.shape[1],patches.shape[2],patches.shape[3]))
				acu[x:xx,y:yy,z:zz]=acu[x:xx,y:yy,z:zz]+pesos
				n=n+1
	ind=np.where(acu==0)
	acu[ind]=1
	output=output/acu
	return output

def patch_extract_3D_mask(input,mask,patch_radio,norm=0):
	n=0
	length=np.sum(mask)
	patches_3D=np.zeros((length,2*patch_radio[0]+1,2*patch_radio[1]+1,2*patch_radio[2]+1),input.dtype)
	for x in range(0,input.shape[0]):
		for y in range(0,input.shape[1]):
			for z in range(0,input.shape[2]):
				if(mask[x,y,z]>0):
					xi=x-patch_radio[0]
					xf=x+patch_radio[0]+1
					yi=y-patch_radio[1]
					yf=y+patch_radio[1]+1
					zi=z-patch_radio[2]
					zf=z+patch_radio[2]+1
					if(xi>=0 and xf<=input.shape[0] and yi>=0 and yf<=input.shape[1] and zi>=0 and zf<=input.shape[2]):
						a=input[xi:xf,yi:yf,zi:zf]
						if(norm):
							m=np.mean(a)
							s=np.std(a)
							a=(a-m)/s
						#print(xi,xf,yi,yf,zi,zf)
						patches_3D[n,:,:,:]=a
						n=n+1
					else:
						a=np.zeros((2*patch_radio[0]+1,2*patch_radio[1]+1,2*patch_radio[2]+1))
						xi1=max(xi,0)
						yi1=max(yi,0)
						zi1=max(zi,0)
						xf1=min(xf,input.shape[0]-1)
						yf1=min(yf,input.shape[1]-1)
						zf1=min(zf,input.shape[2]-1)
						#print(xi1,xf1,yi1,yf1,zi1,zf1)
						a[(xi1-xi):(2*patch_radio[0]+1-(xf-xf1)),(yi1-yi):(2*patch_radio[1]+1-(yf-yf1)),(zi1-zi):(2*patch_radio[2]+1-(zf-zf1))]=input[xi1:xf1,yi1:yf1,zi1:zf1]
						if(norm):
							m=np.mean(a)
							s=np.std(a)
							a=(a-m)/s
						patches_3D[n,:,:,:]=a
						n=n+1

	return patches_3D

def patch_reconstruct_3D_mask(input,mask,labels):
	n=0
	for x in range(0,input.shape[0]):
		for y in range(0,input.shape[1]):
			for z in range(0,input.shape[2]):
				if(mask[x,y,z]>0):
					input[x,y,z]=labels[n]
					n=n+1
	return input

def Subimage_extraction(input,factor=2,norm=0):
	n=0
	for x in range(0,factor):
		for y in range(0,factor):
			for z in range(0,factor):
				temp=input[x::factor,y::factor,z::factor]
				temp=np.reshape(temp,(1,temp.shape[0],temp.shape[1],temp.shape[2]))
				if(norm>0):
					m=temp.mean()
					s=temp.std()
					temp=(temp-m)/s
				if(n==0):
					output=temp
				else:
					output=np.concatenate((output,temp), axis=0)
				n=n+1
	return output

def Subimage_reconstruction(input,size,factor=2):
	output=np.zeros(size)
	n=0
	for x in range(0,factor):
		for y in range(0,factor):
			for z in range(0,factor):
				output[x::factor,y::factor,z::factor,:]=input[n]
				n=n+1
	return output
