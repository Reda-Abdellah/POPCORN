#################################################################
#
# AssemblyNET: Deep learning for Brain segmentation
#
# Authors: Jose Vicente Manjon Herrera
#		  Pierrick Coupe
#		  Boris Mansencal (histogram selection of label)
#
#	Date: 12/02/2019
#
#################################################################

import os
import glob
import numpy as np
import nibabel as nii
import math
import operator
import patch_extraction
from scipy.ndimage.interpolation import zoom
from keras.models import load_model
from scipy import ndimage
import scipy.io as sio
from scipy import signal, stats
import statsmodels.api as sm
from scipy.signal import argrelextrema


def normalize_image(vol, contrast):
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

def read_Tile_1mm_symetrie(ii, lib_path, nbNN,yname='tiles_lab_1mm_',is_=''):
	n=0
	a=np.zeros((nbNN[0],nbNN[1],nbNN[2]))
	for x in range(nbNN[0]):
		for y in range(nbNN[1]):
			for z in range(nbNN[2]):
				a[x,y,z]= n
				n=n+1

	ind=np.where(a == ii)
	#print(ind)
	vv=int(a[-1-ind[0],ind[1],ind[2]])

	path=os.getcwd()
	os.chdir(lib_path)
	listaT1 = sorted(glob.glob("DLB*t1*.nii"))
	#filex="tiles_img_1mm_"+str(ii)+".npy"
	#filey="tiles_lab_1mm_"+str(ii)+".npy"
	#filelist="list_lab_1mm_"+str(ii)+".npy"
	filex="tiles_img_1mm_"+str(ii)+is_+".npy"
	filey=yname+str(ii)+is_+".npy"
	#print(filey)
	filelist="list_lab_1mm_0.npy"
	x=np.load(filex, mmap_mode='r+')
	y=np.load(filey, mmap_mode='r+')
	lista=np.load(filelist)

	#extend y
	numfiles=x.shape[0]
	ps1=x.shape[1]
	ps2=x.shape[2]
	ps3=x.shape[3]
	ye=np.empty([numfiles, ps1, ps2, ps3, len(lista)], dtype='uint8')
	for jjj, iii in enumerate(lista):
		ye[:, :, :, :, jjj] = (y==iii).astype('uint8')[:, :, :, :, 0]
	y = ye
	#"""
	if(ii< (nbNN[1]*nbNN[2]*(nbNN[0]-1)/2)  ):

		filex="tiles_img_1mm_"+str(vv)+is_+".npy"
		filey=yname+str(vv)+is_+".npy"
		filelist="list_lab_1mm_0.npy"
		x_=np.load(filex, mmap_mode='r+')
		y_=np.load(filey, mmap_mode='r+')
		lista=np.load(filelist)

		#extend y
		numfiles=x_.shape[0]
		ps1=x_.shape[1]
		ps2=x_.shape[2]
		ps3=x_.shape[3]
		ye_=np.empty([numfiles, ps1, ps2, ps3, len(lista)], dtype='uint8')
		for jjj, iii in enumerate(lista):
			ye_[:, :, :, :, jjj] = (y_==iii).astype('uint8')[:, :, :, :, 0]
		y_ = ye_
		x_=x_[:,-1::-1,:,:,:]
		y_=y_[:,-1::-1,:,:,:]
		x= np.concatenate((x,x_))
		y= np.concatenate((y,y_))

	os.chdir(path)

	return x,y,path,listaT1,lista




def read_Tile_1mm_symetrie_sum(ii, lib_path, nbNN):
	n=0
	a=np.zeros((nbNN[0],nbNN[1],nbNN[2]))
	for x in range(nbNN[0]):
		for y in range(nbNN[1]):
			for z in range(nbNN[2]):
				a[x,y,z]= n
				n=n+1

	ind=np.where(a == ii)
	vv=int(a[-1-ind[0],ind[1],ind[2]])

	path=os.getcwd()
	os.chdir(lib_path)
	listaT1 = sorted(glob.glob("DLB*t1*.nii"))
	#filex="tiles_img_1mm_"+str(ii)+".npy"
	#filey="tiles_lab_1mm_"+str(ii)+".npy"
	#filelist="list_lab_1mm_"+str(ii)+".npy"
	filex="tiles_img_1mm_"+str(ii)+".npy"
	filey_='tiles_lab2_1mm_'+str(ii)+".npy"
	filey='tiles_lab1_1mm_'+str(ii)+".npy"
	#print(filey)
	filelist="list_lab_1mm_0.npy"
	x=np.load(filex, mmap_mode='r')
	y=np.load(filey, mmap_mode='r')
	y_2=np.load(filey_, mmap_mode='r')
	y=y+y_2
	y[y==2]=1

	lista=np.load(filelist)

	#extend y
	numfiles=x.shape[0]
	ps1=x.shape[1]
	ps2=x.shape[2]
	ps3=x.shape[3]
	ye=np.empty([numfiles, ps1, ps2, ps3, len(lista)], dtype='uint8')
	for jjj, iii in enumerate(lista):
		ye[:, :, :, :, jjj] = (y==iii).astype('uint8')[:, :, :, :, 0]
	y = ye
	#"""
	if(ii< (nbNN[1]*nbNN[2]*(nbNN[0]-1)/2)  ):

		filex="tiles_img_1mm_"+str(vv)+".npy"
		filey_='tiles_lab2_1mm_'+str(vv)+".npy"
		filey='tiles_lab1_1mm_'+str(vv)+".npy"
		filelist="list_lab_1mm_0.npy"
		x_=np.load(filex, mmap_mode='r')
		y_=np.load(filey, mmap_mode='r')

		y_22=np.load(filey_, mmap_mode='r')
		y_=y_+y_22
		y_[y_==2]=1
		lista=np.load(filelist)

		#extend y
		numfiles=x_.shape[0]
		ps1=x_.shape[1]
		ps2=x_.shape[2]
		ps3=x_.shape[3]
		ye_=np.empty([numfiles, ps1, ps2, ps3, len(lista)], dtype='uint8')
		for jjj, iii in enumerate(lista):
			ye_[:, :, :, :, jjj] = (y_==iii).astype('uint8')[:, :, :, :, 0]
		y_ = ye_
		x_=x_[:,-1::-1,:,:,:]
		y_=y_[:,-1::-1,:,:,:]
		x= np.concatenate((x,x_))
		y= np.concatenate((y,y_))

	os.chdir(path)

	return x,y,path,listaT1,lista



def read_Tile_1mm(ii, lib_path):

	path=os.getcwd()
	os.chdir(lib_path)
	listaT1 = sorted(glob.glob("DLB*t1*.nii"))
	#filex="tiles_img_1mm_"+str(ii)+".npy"
	#filey="tiles_lab_1mm_"+str(ii)+".npy"
	#filelist="list_lab_1mm_"+str(ii)+".npy"
	filex="tiles_img_1mm_"+str(ii)+".npy"
	filey="tiles_lab_1mm_"+str(ii)+".npy"
	#filelist="list_lab_1mm_"+str(ii)+".npy"
	x=np.load(filex, mmap_mode='r')
	y=np.load(filey, mmap_mode='r')
	#lista=np.load(filelist)
	lista=np.array([0,1])


	#"""
	#extend y
	numfiles=x.shape[0]
	ps1=x.shape[1]
	ps2=x.shape[2]
	ps3=x.shape[3]
	#y_mean=np.mean(y,axis=0)
	#label_proba=np.ones((numfiles,ps1,ps2,ps3,1))*y_mean
	#x=np.concatenate((x,label_proba),axis=4)
	ye=np.empty([numfiles, ps1, ps2, ps3, len(lista)], dtype='uint8')
	for jjj, iii in enumerate(lista):
		ye[:, :, :, :, jjj] = (y==iii).astype('uint8')[:, :, :, :, 0]
	y = ye
	#"""
	os.chdir(path)

	return x,y,path,listaT1,lista

def read_Tile_1mm_ict(ii, lib_path):

	path=os.getcwd()
	os.chdir(lib_path)
	listaT1 = sorted(glob.glob("DLB*t1*.nii"))
	#filex="tiles_img_1mm_"+str(ii)+".npy"
	#filey="tiles_lab_1mm_"+str(ii)+".npy"
	#filelist="list_lab_1mm_"+str(ii)+".npy"
	filex="tiles_img_1mm_"+str(ii)+".npy"
	filey="tiles_lab_1mm_"+str(ii)+".npy"
	#filelist="list_lab_1mm_"+str(ii)+".npy"
	x=np.load(filex, mmap_mode='r')
	y=np.load(filey, mmap_mode='r')
	#lista=np.load(filelist)
	lista=np.array([0,1])

	#"""
	#extend y
	numfiles=x.shape[0]
	ps1=x.shape[1]
	ps2=x.shape[2]
	ps3=x.shape[3]
	#y_mean=np.mean(y,axis=0)
	#label_proba=np.ones((numfiles,ps1,ps2,ps3,1))*y_mean
	#x=np.concatenate((x,label_proba),axis=4)
	ye=np.empty([numfiles, ps1, ps2, ps3, len(lista)], dtype='uint8')
	for jjj, iii in enumerate(lista):
		ye[:, :, :, :, jjj] = (y==iii).astype('uint8')[:, :, :, :, 0]
	y = ye
	#"""
	os.chdir(path)

	return x,y,path,listaT1,lista
def read_Tile_1mm_(ii, lib_path):

	path=os.getcwd()
	os.chdir(lib_path)
	listaT1 = sorted(glob.glob("DLB*t1*.nii"))
	#filex="tiles_img_1mm_"+str(ii)+".npy"
	#filey="tiles_lab_1mm_"+str(ii)+".npy"
	#filelist="list_lab_1mm_"+str(ii)+".npy"
	filex="tiles_img_1mm_"+str(ii)+"_.npy"
	filey="tiles_lab_1mm_"+str(ii)+"_.npy"
	filelist="list_lab_1mm_"+str(ii)+"_.npy"
	x=np.load(filex, mmap_mode='r')
	y=np.load(filey, mmap_mode='r')
	lista=np.load(filelist)


	#"""
	#extend y
	numfiles=x.shape[0]
	ps1=x.shape[1]
	ps2=x.shape[2]
	ps3=x.shape[3]
	ye=np.empty([numfiles, ps1, ps2, ps3, len(lista)], dtype='uint8')
	for jjj, iii in enumerate(lista):
		ye[:, :, :, :, jjj] = (y==iii).astype('uint8')[:, :, :, :, 0]
	y = ye
	#"""
	os.chdir(path)

	return x,y,path,listaT1,lista





def save_Tile_1mm_from_npy_4mods(nbNN,ps, lib_path, AtlasPrior, Multiscale, FineTuning = 0, path_libteacher = None):

	path=os.getcwd()
	#params
	ps1=ps[0]
	ps2=ps[1]
	ps3=ps[2]

	# read data
	os.chdir(lib_path)

	if("isbi" in lib_path):

		listaT1 = sorted(glob.glob("*t1.npy"))
		listaFLAIR = sorted(glob.glob("*flair.npy"))
		listaT2 = sorted(glob.glob("*t2.npy"))
		listaPD = sorted(glob.glob("*pd.npy"))
		listaLAB = sorted(glob.glob("*lab.npy"))
		listaMASK = sorted(glob.glob("*mask.npy"))
	else:
		listaT1 = sorted(glob.glob("DLB*t1*.nii*"))
		listaFLAIR = sorted(glob.glob("DLB*flair*.nii*"))
		listaLAB = sorted(glob.glob("DLB*lesion*.nii*"))
		listaMASK = sorted(glob.glob("DLB*mask*.nii*"))


	numfiles=len(listaT1)


	#"""
	for i in range(0,numfiles):
	#for i in range(408,numfiles):

		print(" ")
		print("Images")
		print(str(i+1))
		print(listaT1[i])
		print(listaT2[i])
		print(listaPD[i])
		print(listaFLAIR[i])
		print(listaMASK[i])
		print(listaLAB[i])

		T1=np.load(listaT1[i])
		FLAIR=np.load(listaFLAIR[i])
		T2=np.load(listaT2[i])
		PD=np.load(listaPD[i])
		LABb=np.load(listaLAB[i])
		MASK = np.load(listaMASK[i])
		LABb=LABb.astype('int')
		MASK=MASK.astype('int')
		ind=np.where(MASK>0)
		indbg=np.where(MASK==0)

		#normalization
		T1[indbg] = 0
		m1=np.mean(T1[ind])
		s1=np.std(T1[ind])
		#T1=(T1-m1)/s1
		T1[ind]=(T1[ind]-m1)/(s1)

		T2[indbg] = 0
		m1=np.mean(T2[ind])
		s1=np.std(T2[ind])
		#T1=(T1-m1)/s1
		T2[ind]=(T2[ind]-m1)/(s1)



		PD[indbg] = 0
		m1=np.mean(PD[ind])
		s1=np.std(PD[ind])
		#T1=(T1-m1)/s1
		PD[ind]=(PD[ind]-m1)/(s1)

		FLAIR[indbg] = 0
		m1=np.mean(FLAIR[ind])
		s1=np.std(FLAIR[ind])
		#FLAIR=(FLAIR-m1)/s1
		FLAIR[ind]=(FLAIR[ind]-m1)/(s1)

		crop_bg = 4 # To crop null border.
		overlap1 = np.floor((nbNN[0]*ps1 - (T1.shape[0]-2*crop_bg)) / (nbNN[0]-1))
		offset1 = ps1 - overlap1.astype('int')
		#print("image size1",T1.shape[0])
		print("overlap1=",overlap1.astype('int'))
		print("offset1=",offset1)
		overlap2 = np.floor((nbNN[1]*ps2 - (T1.shape[1]-2*crop_bg)) / (nbNN[1]-1))
		offset2 = ps2 - overlap2.astype('int')
		#print("image size2",T1.shape[1])
		print("overlap2=",overlap2.astype('int'))
		print("offset2=",offset2)
		overlap3 = np.floor((nbNN[2]*ps3 - (T1.shape[2]-2*crop_bg)) / (nbNN[2]-1))
		offset3 = ps3 - overlap3.astype('int')
		#print("image size3",T1.shape[2])
		print("overlap3=",overlap3.astype('int'))
		print("offset3=",offset3)

		#pT1=patch_extraction.patch_extract_3D(T1,(ps1,ps2,ps3),off1,off2,off3)
		pT1=patch_extraction.patch_extract_3D_v2(T1,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		pFLAIR=patch_extraction.patch_extract_3D_v2(FLAIR,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		pLABb=patch_extraction.patch_extract_3D_v2(LABb,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		pPD=patch_extraction.patch_extract_3D_v2(PD,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		pT2=patch_extraction.patch_extract_3D_v2(T2,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		# We select only the ii location (to be optimized)

		fileT1="stack_t1_1mm_"+str(i)+".npy"
		fileT2="stack_t2_1mm_"+str(i)+".npy"
		filePD="stack_pd_1mm_"+str(i)+".npy"
		fileFLAIR="stack_flair_1mm_"+str(i)+".npy"
		fileLAB="stack_lab_1mm_"+str(i)+".npy"

		pT1 = pT1.astype('float32')
		pT2 = pT2.astype('float32')
		pPD = pPD.astype('float32')
		pFLAIR = pFLAIR.astype('float32')
		pLABb = pLABb.astype('uint16')
		np.save(fileT1, pT1)
		np.save(fileT2, pT2)
		np.save(filePD, pPD)
		np.save(fileLAB, pLABb)
		np.save(fileFLAIR, pFLAIR)

		if Multiscale == 1:
			pSEG2mm=patch_extraction.patch_extract_3D_v2(SEG2mm,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
			fileSEG2mm="stack_seg2mm_1mm_"+str(i)+".npy"
			pSEG2mm = pSEG2mm.astype('float32')
			np.save(fileSEG2mm, pSEG2mm)

		if AtlasPrior > 0:
			pAtlas=patch_extraction.patch_extract_3D_v2(Atlas,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
			fileAtlas="stack_atlas_1mm_"+str(i)+".npy"
			pAtlas = pAtlas.astype('float32')
			#np.save(fileSEG2mm, pSEG2mm)
			np.save(fileAtlas, pAtlas)
	#"""
	numLabels=2
	lista = np.array([0, 1])

	for ii in range(0,nbNN[0]*nbNN[1]*nbNN[2]):

		cpt = 0
		print(" ")
		print("Networks")
		print(str(ii+1))

		pLAB=np.empty([1, ps1, ps2, ps3, 1], dtype='uint8')

		pT1LastDim=4 # FLAIR + T1w
		if Multiscale == 1:
			pT1LastDim+=1
		if AtlasPrior > 0:
			pT1LastDim+=1

		cpt = 0
		for i in range(0,numfiles):

			fileLAB= "stack_lab_1mm_"+str(i)+".npy"
			pLABb = np.load(fileLAB, mmap_mode='r')
			pLABb = np.reshape(pLABb[ii], (1,pLABb.shape[1],pLABb.shape[2],pLABb.shape[3], 1))
			if np.sum(pLABb) > -1 : #to better balance the traning, at least one lesion has to be present in the tile
				#print("Size of lesions in image ",i," for the tile ",ii," = ", np.sum(pLABb))
				cpt = cpt +1

		x=np.empty([cpt, ps1, ps2, ps3, pT1LastDim], dtype='float32')
		print(x.shape)
		y=np.empty([cpt, ps1, ps2, ps3, 1], dtype='uint8')
		print(x.shape)

		cpt = 0
		for i in range(0,numfiles):

			fileLAB="stack_lab_1mm_"+str(i)+".npy"
			pLABb =np.load(fileLAB, mmap_mode='r')
			pLABb = np.reshape(pLABb[ii], (1,pLABb.shape[1],pLABb.shape[2],pLABb.shape[3], 1))

			if np.sum(pLABb) > -1 : #to better balance the traning, at least one lesion has to be present in the tile
				#print("Size of lesions in image ",i," for the tile ",ii," = ", np.sum(pLABb))
				fileT1="stack_t1_1mm_"+str(i)+".npy"
				pT1 =np.load(fileT1, mmap_mode='r')
				pT1 = np.reshape(pT1[ii], (1,pT1.shape[1],pT1.shape[2],pT1.shape[3], 1))

				fileT2="stack_t2_1mm_"+str(i)+".npy"
				pT2 =np.load(fileT2, mmap_mode='r')
				pT2 = np.reshape(pT2[ii], (1,pT1.shape[1],pT1.shape[2],pT1.shape[3], 1))

				filePD="stack_pd_1mm_"+str(i)+".npy"
				pPD =np.load(filePD, mmap_mode='r')
				pPD = np.reshape(pPD[ii], (1,pT1.shape[1],pT1.shape[2],pT1.shape[3], 1))

				fileFLAIR="stack_flair_1mm_"+str(i)+".npy"
				pFLAIR =np.load(fileFLAIR, mmap_mode='r')
				pFLAIR = np.reshape(pFLAIR[ii], (1,pT1.shape[1],pT1.shape[2],pT1.shape[3], 1))
				pT1=np.concatenate((pT1,pFLAIR,pT2,pPD), axis=4)


				if Multiscale == 1:
					fileSEG2mm="stack_seg2mm_1mm_"+str(i)+".npy"
					pSEG2mm =np.load(fileSEG2mm, mmap_mode='r')
					pSEG2mm = np.reshape(pSEG2mm[ii], (1,pSEG2mm.shape[1],pSEG2mm.shape[2],pSEG2mm.shape[3], 1))
					pT1=np.concatenate((pT1,pSEG2mm), axis=4) # Concatenate T1 and Seg from 2mm interpolated at 1mm

				if AtlasPrior > 0:
					fileAtlas="stack_atlas_1mm_"+str(i)+".npy"
					pAtlas =np.load(fileAtlas, mmap_mode='r')
					pAtlas = np.reshape(pAtlas[ii], (1,pAtlas.shape[1],pAtlas.shape[2],pAtlas.shape[3], 1))
					pT1=np.concatenate((pT1,pAtlas), axis=4) # Concatenate T1 and Seg from 2mm interpolated at 1mm

				#output
				for iii in lista:
					ind = np.where(pLABb == iii)
					pLAB[ind] = pLABb[ind]

				x[cpt, :, :, :, :] = pT1
				y[cpt, :, :, :, :] = pLAB
				cpt = cpt + 1


		print("sizeLAB=",y.shape)
		print("Number of used images = ",cpt," ( ", cpt/numfiles*100, " %) ")
		#x = x.astype('float32')
		#y = y.astype('uint16')

		#filex="tiles_img_1mm_"+str(ii)+".npy"
		#filey="tiles_lab_1mm_"+str(ii)+".npy"
		#filelist="list_lab_1mm_"+str(ii)+".npy"

		filex="tiles_img_1mm_"+str(ii)+".npy"
		filey="tiles_lab_1mm_"+str(ii)+".npy"
		filelist="list_lab_1mm_"+str(ii)+".npy"

		np.save(filex, x)
		np.save(filey, y)
		np.save(filelist,lista)

		#x = None
		#y = None

	os.chdir(path)

	return path,listaT1,lista


def save_Tile_1mm_from_npy(nbNN,ps, lib_path, AtlasPrior, Multiscale, FineTuning = 0, path_libteacher = None):

	path=os.getcwd()
	#params
	ps1=ps[0]
	ps2=ps[1]
	ps3=ps[2]

	# read data
	os.chdir(lib_path)

	if("isbi" in lib_path):

		listaT1 = sorted(glob.glob("*t1.npy"))
		listaFLAIR = sorted(glob.glob("*flair.npy"))
		listaLAB = sorted(glob.glob("*lab.npy"))
		listaMASK = sorted(glob.glob("*mask.npy"))
	else:
		listaT1 = sorted(glob.glob("DLB*t1*.nii*"))
		listaFLAIR = sorted(glob.glob("DLB*flair*.nii*"))
		listaLAB = sorted(glob.glob("DLB*lesion*.nii*"))
		listaMASK = sorted(glob.glob("DLB*mask*.nii*"))


	numfiles=len(listaT1)



	for i in range(0,numfiles):
	#for i in range(408,numfiles):

		print(" ")
		print("Images")
		print(str(i+1))
		print(listaT1[i])
		print(listaFLAIR[i])
		print(listaMASK[i])
		print(listaLAB[i])

		T1=np.load(listaT1[i])
		FLAIR=np.load(listaFLAIR[i])
		T2=np.load(listaT2[i])
		PD=np.load(listaPD[i])
		LABb=np.load(listaLAB[i])
		MASK = np.load(listaMASK[i])
		LABb=LABb.astype('int')
		MASK=MASK.astype('int')
		ind=np.where(MASK>0)
		indbg=np.where(MASK==0)
		#"""
		#normalization
		T1[indbg] = 0
		m1=np.mean(T1[ind])
		s1=np.std(T1[ind])
		#T1=(T1-m1)/s1
		T1[ind]=(T1[ind]-m1)/(s1)

		T2[indbg] = 0
		m1=np.mean(T2[ind])
		s1=np.std(T2[ind])
		#T1=(T1-m1)/s1
		T2[ind]=(T2[ind]-m1)/(s1)

		PD[indbg] = 0
		m1=np.mean(PD[ind])
		s1=np.std(PD[ind])
		#T1=(T1-m1)/s1
		PD[ind]=(PD[ind]-m1)/(s1)






		FLAIR[indbg] = 0
		m1=np.mean(FLAIR[ind])
		s1=np.std(FLAIR[ind])
		#FLAIR=(FLAIR-m1)/s1
		FLAIR[ind]=(FLAIR[ind]-m1)/(s1)
		#"""
		crop_bg = 4 # To crop null border.
		overlap1 = np.floor((nbNN[0]*ps1 - (T1.shape[0]-2*crop_bg)) / (nbNN[0]-1))
		offset1 = ps1 - overlap1.astype('int')
		#print("image size1",T1.shape[0])
		print("overlap1=",overlap1.astype('int'))
		print("offset1=",offset1)
		overlap2 = np.floor((nbNN[1]*ps2 - (T1.shape[1]-2*crop_bg)) / (nbNN[1]-1))
		offset2 = ps2 - overlap2.astype('int')
		#print("image size2",T1.shape[1])
		print("overlap2=",overlap2.astype('int'))
		print("offset2=",offset2)
		overlap3 = np.floor((nbNN[2]*ps3 - (T1.shape[2]-2*crop_bg)) / (nbNN[2]-1))
		offset3 = ps3 - overlap3.astype('int')
		#print("image size3",T1.shape[2])
		print("overlap3=",overlap3.astype('int'))
		print("offset3=",offset3)

		#pT1=patch_extraction.patch_extract_3D(T1,(ps1,ps2,ps3),off1,off2,off3)
		pT1=patch_extraction.patch_extract_3D_v2(T1,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		pFLAIR=patch_extraction.patch_extract_3D_v2(FLAIR,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		pT2=patch_extraction.patch_extract_3D_v2(T2,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		pPD=patch_extraction.patch_extract_3D_v2(PD,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		pLABb=patch_extraction.patch_extract_3D_v2(LABb,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		# We select only the ii location (to be optimized)

		fileT1="stack_t1_1mm_"+str(i)+".npy"
		fileFLAIR="stack_flair_1mm_"+str(i)+".npy"
		fileT2="stack_t2_1mm_"+str(i)+".npy"
		filePD="stack_pd_1mm_"+str(i)+".npy"
		fileLAB="stack_lab_1mm_"+str(i)+".npy"

		pT1 = pT1.astype('float32')
		pFLAIR = pFLAIR.astype('float32')
		pT2 = pT2.astype('float32')
		pPD = pPD.astype('float32')
		pLABb = pLABb.astype('uint16')
		np.save(fileT1, pT1)
		np.save(fileLAB, pLABb)
		np.save(fileT2, pT2)
		np.save(filePD, pPD)
		np.save(fileFLAIR, pFLAIR)

		if Multiscale == 1:
			pSEG2mm=patch_extraction.patch_extract_3D_v2(SEG2mm,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
			fileSEG2mm="stack_seg2mm_1mm_"+str(i)+".npy"
			pSEG2mm = pSEG2mm.astype('float32')
			np.save(fileSEG2mm, pSEG2mm)

		if AtlasPrior > 0:
			pAtlas=patch_extraction.patch_extract_3D_v2(Atlas,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
			fileAtlas="stack_atlas_1mm_"+str(i)+".npy"
			pAtlas = pAtlas.astype('float32')
			#np.save(fileSEG2mm, pSEG2mm)
			np.save(fileAtlas, pAtlas)

	numLabels=2
	lista = np.array([0, 1])

	for ii in range(0,nbNN[0]*nbNN[1]*nbNN[2]):

		cpt = 0
		print(" ")
		print("Networks")
		print(str(ii+1))

		pLAB=np.empty([1, ps1, ps2, ps3, 1], dtype='uint8')

		pT1LastDim=2 # FLAIR + T1w
		if Multiscale == 1:
			pT1LastDim+=1
		if AtlasPrior > 0:
			pT1LastDim+=1

		cpt = 0
		for i in range(0,numfiles):

			fileLAB= "stack_lab_1mm_"+str(i)+".npy"
			pLABb = np.load(fileLAB, mmap_mode='r')
			pLABb = np.reshape(pLABb[ii], (1,pLABb.shape[1],pLABb.shape[2],pLABb.shape[3], 1))
			if np.sum(pLABb) > -1 : #to better balance the traning, at least one lesion has to be present in the tile
				#print("Size of lesions in image ",i," for the tile ",ii," = ", np.sum(pLABb))
				cpt = cpt +1

		x=np.empty([cpt, ps1, ps2, ps3, 4], dtype='float32')
		print(x.shape)
		y=np.empty([cpt, ps1, ps2, ps3, 1], dtype='uint8')
		print(x.shape)

		cpt = 0
		for i in range(0,numfiles):

			fileLAB="stack_lab_1mm_"+str(i)+".npy"
			pLABb =np.load(fileLAB, mmap_mode='r')
			pLABb = np.reshape(pLABb[ii], (1,pLABb.shape[1],pLABb.shape[2],pLABb.shape[3], 1))

			if np.sum(pLABb) > -1 : #to better balance the traning, at least one lesion has to be present in the tile
				#print("Size of lesions in image ",i," for the tile ",ii," = ", np.sum(pLABb))
				fileT1="stack_t1_1mm_"+str(i)+".npy"
				pT1 =np.load(fileT1, mmap_mode='r')
				pT1 = np.reshape(pT1[ii], (1,pT1.shape[1],pT1.shape[2],pT1.shape[3], 1))

				fileT1="stack_t2_1mm_"+str(i)+".npy"
				pT2 =np.load(fileT2, mmap_mode='r')
				pT2 = np.reshape(pT2[ii], (1,pT1.shape[1],pT1.shape[2],pT1.shape[3], 1))

				filePD="stack_pd_1mm_"+str(i)+".npy"
				pPD =np.load(filePD, mmap_mode='r')
				pPD = np.reshape(pPD[ii], (1,pT1.shape[1],pT1.shape[2],pT1.shape[3], 1))

				fileFLAIR="stack_flair_1mm_"+str(i)+".npy"
				pFLAIR =np.load(fileFLAIR, mmap_mode='r')
				pFLAIR = np.reshape(pFLAIR[ii], (1,pT1.shape[1],pT1.shape[2],pT1.shape[3], 1))
				pT1=np.concatenate((pT1,pFLAIR,pT2,pPD), axis=4)


				if Multiscale == 1:
					fileSEG2mm="stack_seg2mm_1mm_"+str(i)+".npy"
					pSEG2mm =np.load(fileSEG2mm, mmap_mode='r')
					pSEG2mm = np.reshape(pSEG2mm[ii], (1,pSEG2mm.shape[1],pSEG2mm.shape[2],pSEG2mm.shape[3], 1))
					pT1=np.concatenate((pT1,pSEG2mm), axis=4) # Concatenate T1 and Seg from 2mm interpolated at 1mm

				if AtlasPrior > 0:
					fileAtlas="stack_atlas_1mm_"+str(i)+".npy"
					pAtlas =np.load(fileAtlas, mmap_mode='r')
					pAtlas = np.reshape(pAtlas[ii], (1,pAtlas.shape[1],pAtlas.shape[2],pAtlas.shape[3], 1))
					pT1=np.concatenate((pT1,pAtlas), axis=4) # Concatenate T1 and Seg from 2mm interpolated at 1mm

				#output
				for iii in lista:
					ind = np.where(pLABb == iii)
					pLAB[ind] = pLABb[ind]

				x[cpt, :, :, :, :] = pT1
				y[cpt, :, :, :, :] = pLAB
				cpt = cpt + 1


		print("sizeLAB=",y.shape)
		print("Number of used images = ",cpt," ( ", cpt/numfiles*100, " %) ")
		#x = x.astype('float32')
		#y = y.astype('uint16')

		#filex="tiles_img_1mm_"+str(ii)+".npy"
		#filey="tiles_lab_1mm_"+str(ii)+".npy"
		#filelist="list_lab_1mm_"+str(ii)+".npy"

		filex="tiles_img_1mm_"+str(ii)+".npy"
		filey="tiles_lab_1mm_"+str(ii)+".npy"
		filelist="list_lab_1mm_"+str(ii)+".npy"

		np.save(filex, x)
		np.save(filey, y)
		np.save(filelist,lista)

		#x = None
		#y = None

	os.chdir(path)

	return path,listaT1,lista


def read_Tile_2mm(ii, lib_path):

	path=os.getcwd()
	os.chdir(lib_path)
	listaT1 = sorted(glob.glob("DLB*t1*.nii"))

	filex="tiles_img_2mm_"+str(ii)+".npy"
	filey="tiles_lab_2mm_"+str(ii)+".npy"
	filelist="list_lab_2mm_"+str(ii)+".npy"
	x=np.load(filex, mmap_mode='r')
	y=np.load(filey, mmap_mode='r')
	lista=np.load(filelist)

	#extend y
	numfiles=x.shape[0]
	ps1=x.shape[1]
	ps2=x.shape[2]
	ps3=x.shape[3]
	ye=np.empty([numfiles, ps1, ps2, ps3, len(lista)], dtype='uint8')
	for jjj, iii in enumerate(lista):
		ye[:, :, :, :, jjj] = (y==iii).astype('uint8')[:, :, :, :, 0]
	y = ye

	os.chdir(path)

	return x,y,path,listaT1,lista


def save_lesion_heatmap(nbNN,ps):

	ps1=ps[0]
	ps2=ps[1]
	ps3=ps[2]

	T1_img = nii.load("lib/LAB_heatmap_50_100.nii.gz")
	T1= T1_img.get_data()

	crop_bg = 4 # To crop null border.
	overlap1 = np.floor((nbNN[0]*ps1 - (T1.shape[0]-2*crop_bg)) / (nbNN[0]-1))
	offset1 = ps1 - overlap1.astype('int')
	overlap2 = np.floor((nbNN[1]*ps2 - (T1.shape[1]-2*crop_bg)) / (nbNN[1]-1))
	offset2 = ps2 - overlap2.astype('int')
	overlap3 = np.floor((nbNN[2]*ps3 - (T1.shape[2]-2*crop_bg)) / (nbNN[2]-1))
	offset3 = ps3 - overlap3.astype('int')
	pT1=patch_extraction.patch_extract_3D_v2(T1,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)

	pT1[:,0:8,:,:]=0
	pT1[:,:,:,0:8]=0
	pT1[:,:,0:8,:]=0
	pT1[:,:,:,80-7:80]=0
	pT1[:,:,96-7:96,:]=0
	pT1[:,80-7:80,:,:]=0

	filex="small_label.npy"

	np.save(filex, pT1)

	return True



def save_Tile_1mm(nbNN,ps, lib_path, AtlasPrior, Multiscale, FineTuning = 0, path_libteacher = None):
	if(AtlasPrior == 'fine_tune'):
		AtlasPrior =0


	path=os.getcwd()
	#params
	ps1=ps[0]
	ps2=ps[1]
	ps3=ps[2]

	# read data
	os.chdir(lib_path)
	if("volBrain" in lib_path):

		listaT1 = sorted(glob.glob("n_*t1*.nii"))
		listaFLAIR = sorted(glob.glob("n_*flair*.nii"))
		listaLAB = sorted(glob.glob("wmhlesion*.nii*"))
		listaMASK = sorted(glob.glob("mask*.nii"))

	elif("volbrain_by_cluster" in lib_path):

		listaT1 = sorted(glob.glob("*/n_*t1*.nii"))
		listaFLAIR = sorted(glob.glob("*/n_*flair*.nii"))
		#listaLAB = sorted(glob.glob("*/wmhlesion*.nii*"))
		listaLAB = sorted(glob.glob("*/Assembly_seg_*.nii*"))

		listaMASK = sorted(glob.glob("*/mask*.nii"))

	elif("volbrain_qc" in lib_path):

		listaT1 = sorted(glob.glob("n_*t1*.nii"))
		listaFLAIR = sorted(glob.glob("n_*flair*.nii"))
		#listaLAB = sorted(glob.glob("wmhlesion*.nii*"))
		listaLAB = sorted(glob.glob("seg/Assembly_seg_*.nii*"))

		listaMASK = sorted(glob.glob("mask*.nii"))




	elif("WMH" in lib_path):
		listaT1 = sorted(glob.glob("n_mfmni*T1*.nii*"))
		listaFLAIR = sorted(glob.glob("n_mfmni*FLAIR*.nii*"))
		listaLAB = sorted(glob.glob("*onMNI*.nii*"))
		listaMASK = sorted(glob.glob("mask*.nii*"))

	else:
		listaT1 = sorted(glob.glob("DLB*t1*.nii*"))
		listaFLAIR = sorted(glob.glob("DLB*flair*.nii*"))
		listaLAB = sorted(glob.glob("DLB*lesion*.nii*"))
		listaMASK = sorted(glob.glob("DLB*mask*.nii*"))


	numfiles=len(listaT1)


	# os.chdir(path)
	# os.chdir('results')
	# Seg from 2mm scale interpolated at 1mm
	listaSEG2mm = sorted(glob.glob("up_seg_2mm_*.nii*"))

	if (AtlasPrior == 1):
		listaatlasprior = sorted(glob.glob("wlab_*.nii*"))
	if (AtlasPrior == 2):
		listaatlasprior = sorted(glob.glob("Assembly_seg_1mm_n_mmni_*.nii*"))
		listaSEG2mm = sorted(glob.glob("up_seg_2mm_2pass_n_*.nii*"))

	"""
	for i in range(0,numfiles):
	#for i in range(1300,numfiles):
	#for i in range(numfiles,numfiles):

		print(" ")
		print("Images")
		print(str(i+1))
		print(listaT1[i])
		print(listaFLAIR[i])
		print(listaLAB[i])
		print(listaMASK[i])
		if AtlasPrior > 0:
			print(listaatlasprior[i])
		if Multiscale == 1:
			print(listaSEG2mm[i])

		T1_img = nii.load(listaT1[i])
		FLAIR_img = nii.load(listaFLAIR[i])
		LAB_img = nii.load(listaLAB[i])
		MASK_img = nii.load(listaMASK[i])
		T1=T1_img.get_data()
		FLAIR=FLAIR_img.get_data()
		LABb=LAB_img.get_data()
		MASK = MASK_img.get_data()
		LABb=LABb.astype('int')


		sim=np.abs(stats.pearsonr(T1.reshape((-1)),FLAIR.reshape((-1)))[0])

		if(sim ==1 or sim<0.5):
			print(listaT1[i]+' is not good, CORR:'+str(sim))
			continue
		print('quality good')
		if("WMH" in lib_path):
			LABb[LABb==2]=0
		MASK=MASK.astype('int')
		ind=np.where(MASK>0)
		indbg=np.where(MASK==0)

		if AtlasPrior > 0:
			Atlas_img = nii.load(listaatlasprior[i])
			Atlas=Atlas_img.get_data()
			Atlas=Atlas.astype('float32')
			Atlasc = np.zeros(Atlas.shape)
			for indexlab, lab in enumerate(labels_SLANT):
				indl=np.where(Atlas==lab)
				Atlasc[indl] = Atlas[indl]
			print('Removed inconsistent labels Atlas: ', list(set(np.unique(Atlas)) - set(np.unique(Atlasc))))
			Atlas = Atlasc
			Atlas[indbg] = 0
			#inda=np.where(Atlas>0)
			m1=np.mean(Atlas[ind])
			s1=np.std(Atlas[ind])
			Atlas=(Atlas-m1)/s1

		if Multiscale == 1:
			# os.chdir(path)
			# os.chdir('results')
			# Seg from 2mm scale interpolated at 1mm
			SEG2mm_img = nii.load(listaSEG2mm[i])
			SEG2mm=SEG2mm_img.get_data()
			SEG2mm=SEG2mm.astype('float32')
			SEG2mm[indbg] = 0
			#indm=np.where(SEG2mm>0)
			m1=np.mean(SEG2mm[ind])
			s1=np.std(SEG2mm[ind])
			SEG2mm=(SEG2mm-m1)/s1
			# os.chdir(path)
			# os.chdir(lib_path)


		#normalization
		T1[indbg] = 0
		m1=np.mean(T1[ind])
		s1=np.std(T1[ind])
		T1[ind]=(T1[ind]-m1)/s1
		#T1[ind]=(T1[ind]-m1)/s1

		FLAIR[indbg] = 0
		m1=np.mean(FLAIR[ind])
		s1=np.std(FLAIR[ind])
		FLAIR[ind]=(FLAIR[ind]-m1)/s1

		# LAB = np.zeros(LABb.shape)
		# for indexlab, lab in enumerate(labels_SLANT):
		# 	ind=np.where(LABb==lab)
		# 	LAB[ind] = LABb[ind]
		# print('Removed inconsistent labels : ', list(set(np.unique(LABb)) - set(np.unique(LAB))))
		# LABb = LAB


		#subvolume extraction
		# We read the N subvolumes
		# First we estimate the overlap
		crop_bg = 4 # To crop null border.
		overlap1 = np.floor((nbNN[0]*ps1 - (T1.shape[0]-2*crop_bg)) / (nbNN[0]-1))
		offset1 = ps1 - overlap1.astype('int')
		#print("image size1",T1.shape[0])
		print("overlap1=",overlap1.astype('int'))
		print("offset1=",offset1)
		overlap2 = np.floor((nbNN[1]*ps2 - (T1.shape[1]-2*crop_bg)) / (nbNN[1]-1))
		offset2 = ps2 - overlap2.astype('int')
		#print("image size2",T1.shape[1])
		print("overlap2=",overlap2.astype('int'))
		print("offset2=",offset2)
		overlap3 = np.floor((nbNN[2]*ps3 - (T1.shape[2]-2*crop_bg)) / (nbNN[2]-1))
		offset3 = ps3 - overlap3.astype('int')
		#print("image size3",T1.shape[2])
		print("overlap3=",overlap3.astype('int'))
		print("offset3=",offset3)

		#pT1=patch_extraction.patch_extract_3D(T1,(ps1,ps2,ps3),off1,off2,off3)
		pT1=patch_extraction.patch_extract_3D_v2(T1,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		pFLAIR=patch_extraction.patch_extract_3D_v2(FLAIR,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		pLABb=patch_extraction.patch_extract_3D_v2(LABb,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		# We select only the ii location (to be optimized)

		fileT1="stack_t1_1mm_"+str(i)+".npy"
		fileFLAIR="stack_flair_1mm_"+str(i)+".npy"
		fileLAB="stack_lab_1mm_"+str(i)+".npy"

		pT1 = pT1.astype('float32')
		pFLAIR = pFLAIR.astype('float32')
		pLABb = pLABb.astype('uint16')
		np.save(fileT1, pT1)
		np.save(fileLAB, pLABb)
		np.save(fileFLAIR, pFLAIR)

		if Multiscale == 1:
			pSEG2mm=patch_extraction.patch_extract_3D_v2(SEG2mm,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
			fileSEG2mm="stack_seg2mm_1mm_"+str(i)+".npy"
			pSEG2mm = pSEG2mm.astype('float32')
			np.save(fileSEG2mm, pSEG2mm)

		if AtlasPrior > 0:
			pAtlas=patch_extraction.patch_extract_3D_v2(Atlas,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
			fileAtlas="stack_atlas_1mm_"+str(i)+".npy"
			pAtlas = pAtlas.astype('float32')
			#np.save(fileSEG2mm, pSEG2mm)
			np.save(fileAtlas, pAtlas)

	"""
	numLabels=2
	lista = np.array([0, 1])
	for ii in range(0,nbNN[0]*nbNN[1]*nbNN[2]):

		cpt = 0
		print(" ")
		print("Networks")
		print(str(ii+1))

		pLAB=np.empty([1, ps1, ps2, ps3, 1], dtype='uint8')

		pT1LastDim=2 # FLAIR + T1w
		if Multiscale == 1:
			pT1LastDim+=1
		if AtlasPrior > 0:
			pT1LastDim+=1

		cpt = 0
		for i in range(0,numfiles):

			fileLAB= "stack_lab_1mm_"+str(i)+".npy"
			try:
				pLABb = np.load(fileLAB, mmap_mode='r')
			except:
				continue
			pLABb = np.reshape(pLABb[ii], (1,pLABb.shape[1],pLABb.shape[2],pLABb.shape[3], 1))
			if np.sum(pLABb) > 0 : #to better balance the traning, at least one lesion has to be present in the tile
				#print("Size of lesions in image ",i," for the tile ",ii," = ", np.sum(pLABb))
				cpt = cpt +1

		x=np.empty([cpt, ps1, ps2, ps3, pT1LastDim], dtype='float32')
		print(x.shape)
		y=np.empty([cpt, ps1, ps2, ps3, 1], dtype='uint8')
		print(x.shape)

		cpt = 0
		for i in range(0,numfiles):

			fileLAB="stack_lab_1mm_"+str(i)+".npy"
			try:
				pLABb = np.load(fileLAB, mmap_mode='r')
			except:
				continue
			pLABb = np.reshape(pLABb[ii], (1,pLABb.shape[1],pLABb.shape[2],pLABb.shape[3], 1))

			if np.sum(pLABb) > 0 : #to better balance the traning, at least one lesion has to be present in the tile
				#print("Size of lesions in image ",i," for the tile ",ii," = ", np.sum(pLABb))
				fileT1="stack_t1_1mm_"+str(i)+".npy"
				pT1 =np.load(fileT1, mmap_mode='r')
				pT1 = np.reshape(pT1[ii], (1,pT1.shape[1],pT1.shape[2],pT1.shape[3], 1))
				fileFLAIR="stack_flair_1mm_"+str(i)+".npy"
				pFLAIR =np.load(fileFLAIR, mmap_mode='r')
				pFLAIR = np.reshape(pFLAIR[ii], (1,pT1.shape[1],pT1.shape[2],pT1.shape[3], 1))
				pT1=np.concatenate((pT1,pFLAIR), axis=4)


				if Multiscale == 1:
					fileSEG2mm="stack_seg2mm_1mm_"+str(i)+".npy"
					pSEG2mm =np.load(fileSEG2mm, mmap_mode='r')
					pSEG2mm = np.reshape(pSEG2mm[ii], (1,pSEG2mm.shape[1],pSEG2mm.shape[2],pSEG2mm.shape[3], 1))
					pT1=np.concatenate((pT1,pSEG2mm), axis=4) # Concatenate T1 and Seg from 2mm interpolated at 1mm

				if AtlasPrior > 0:
					fileAtlas="stack_atlas_1mm_"+str(i)+".npy"
					pAtlas =np.load(fileAtlas, mmap_mode='r')
					pAtlas = np.reshape(pAtlas[ii], (1,pAtlas.shape[1],pAtlas.shape[2],pAtlas.shape[3], 1))
					pT1=np.concatenate((pT1,pAtlas), axis=4) # Concatenate T1 and Seg from 2mm interpolated at 1mm

				#output
				for iii in lista:
					ind = np.where(pLABb == iii)
					pLAB[ind] = pLABb[ind]

				x[cpt, :, :, :, :] = pT1
				y[cpt, :, :, :, :] = pLAB
				cpt = cpt + 1


		print("sizeLAB=",y.shape)
		print("Number of used images = ",cpt," ( ", cpt/numfiles*100, " %) ")
		#x = x.astype('float32')
		#y = y.astype('uint16')

		#filex="tiles_img_1mm_"+str(ii)+".npy"
		#filey="tiles_lab_1mm_"+str(ii)+".npy"
		#filelist="list_lab_1mm_"+str(ii)+".npy"

		filex="tiles_img_1mm_"+str(ii)+".npy"
		filey="tiles_lab_1mm_"+str(ii)+".npy"
		filelist="list_lab_1mm_"+str(ii)+".npy"

		np.save(filex, x)
		np.save(filey, y)
		np.save(filelist,lista)

		#x = None
		#y = None

	os.chdir(path)

	return path,listaT1,lista



def save_Tile_1mm_4(nbNN,ps, lib_path, AtlasPrior, Multiscale, FineTuning = 0, path_libteacher = None):
	if(AtlasPrior == 'fine_tune'):
		AtlasPrior =0


	path=os.getcwd()
	#params
	ps1=ps[0]
	ps2=ps[1]
	ps3=ps[2]

	# read data
	os.chdir(lib_path)

	listaT1 = sorted(glob.glob("n_mfmni*mprage*.nii*"))
	listaT2 = sorted(glob.glob("registred*t2*.nii*"))
	listaPD = sorted(glob.glob("registred*pd*.nii*"))
	listaFLAIR = sorted(glob.glob("n_mfmni*flair*.nii*"))
	listaMASK = sorted(glob.glob("mask*.nii*"))
	listaLAB= sorted(glob.glob("registred*sum_mask*.nii*"))


	numfiles=len(listaT1)


	# os.chdir(path)
	# os.chdir('results')
	# Seg from 2mm scale interpolated at 1mm
	listaSEG2mm = sorted(glob.glob("up_seg_2mm_*.nii*"))

	if (AtlasPrior == 1):
		listaatlasprior = sorted(glob.glob("wlab_*.nii*"))
	if (AtlasPrior == 2):
		listaatlasprior = sorted(glob.glob("Assembly_seg_1mm_n_mmni_*.nii*"))
		listaSEG2mm = sorted(glob.glob("up_seg_2mm_2pass_n_*.nii*"))


	for i in range(0,numfiles):
		#break
		print(" ")
		print("Images")
		print(str(i+1))
		print(listaT1[i])
		print(listaT2[i])
		print(listaPD[i])
		print(listaFLAIR[i])
		print(listaLAB[i])
		print(listaMASK[i])
		if AtlasPrior > 0:
			print(listaatlasprior[i])
		if Multiscale == 1:
			print(listaSEG2mm[i])

		T1_img = nii.load(listaT1[i])
		T2_img = nii.load(listaT2[i])
		PD_img = nii.load(listaPD[i])
		FLAIR_img = nii.load(listaFLAIR[i])
		LAB_img = nii.load(listaLAB[i])
		MASK_img = nii.load(listaMASK[i])
		T1=T1_img.get_data()
		T2=T2_img.get_data()
		PD=PD_img.get_data()
		FLAIR=FLAIR_img.get_data()
		LABb=LAB_img.get_data()
		MASK = MASK_img.get_data()
		LABb=LABb.astype('int')

		if("WMH" in lib_path):
			LABb[LABb==2]=0
		MASK=MASK.astype('int')
		ind=np.where(MASK>0)
		indbg=np.where(MASK==0)

		if AtlasPrior > 0:
			Atlas_img = nii.load(listaatlasprior[i])
			Atlas=Atlas_img.get_data()
			Atlas=Atlas.astype('float32')
			Atlasc = np.zeros(Atlas.shape)
			for indexlab, lab in enumerate(labels_SLANT):
				indl=np.where(Atlas==lab)
				Atlasc[indl] = Atlas[indl]
			print('Removed inconsistent labels Atlas: ', list(set(np.unique(Atlas)) - set(np.unique(Atlasc))))
			Atlas = Atlasc
			Atlas[indbg] = 0
			#inda=np.where(Atlas>0)
			m1=np.mean(Atlas[ind])
			s1=np.std(Atlas[ind])
			Atlas=(Atlas-m1)/s1

		if Multiscale == 1:
			# os.chdir(path)
			# os.chdir('results')
			# Seg from 2mm scale interpolated at 1mm
			SEG2mm_img = nii.load(listaSEG2mm[i])
			SEG2mm=SEG2mm_img.get_data()
			SEG2mm=SEG2mm.astype('float32')
			SEG2mm[indbg] = 0
			#indm=np.where(SEG2mm>0)
			m1=np.mean(SEG2mm[ind])
			s1=np.std(SEG2mm[ind])
			SEG2mm=(SEG2mm-m1)/s1
			# os.chdir(path)
			# os.chdir(lib_path)



		T2[indbg] = 0
		m1=np.mean(T2[ind])
		s1=np.std(T2[ind])
		T2[ind]=(T2[ind]-m1)/s1

		PD[indbg] = 0
		m1=np.mean(PD[ind])
		s1=np.std(PD[ind])
		PD[ind]=(PD[ind]-m1)/s1


		T1[indbg] = 0
		m1=np.mean(T1[ind])
		s1=np.std(T1[ind])
		T1[ind]=(T1[ind]-m1)/s1
		#T1[ind]=(T1[ind]-m1)/s1

		FLAIR[indbg] = 0
		m1=np.mean(FLAIR[ind])
		s1=np.std(FLAIR[ind])
		FLAIR[ind]=(FLAIR[ind]-m1)/s1

		# LAB = np.zeros(LABb.shape)
		# for indexlab, lab in enumerate(labels_SLANT):
		# 	ind=np.where(LABb==lab)
		# 	LAB[ind] = LABb[ind]
		# print('Removed inconsistent labels : ', list(set(np.unique(LABb)) - set(np.unique(LAB))))
		# LABb = LAB


		#subvolume extraction
		# We read the N subvolumes
		# First we estimate the overlap
		crop_bg = 4 # To crop null border.
		overlap1 = np.floor((nbNN[0]*ps1 - (T1.shape[0]-2*crop_bg)) / (nbNN[0]-1))
		offset1 = ps1 - overlap1.astype('int')
		#print("image size1",T1.shape[0])
		print("overlap1=",overlap1.astype('int'))
		print("offset1=",offset1)
		overlap2 = np.floor((nbNN[1]*ps2 - (T1.shape[1]-2*crop_bg)) / (nbNN[1]-1))
		offset2 = ps2 - overlap2.astype('int')
		#print("image size2",T1.shape[1])
		print("overlap2=",overlap2.astype('int'))
		print("offset2=",offset2)
		overlap3 = np.floor((nbNN[2]*ps3 - (T1.shape[2]-2*crop_bg)) / (nbNN[2]-1))
		offset3 = ps3 - overlap3.astype('int')
		#print("image size3",T1.shape[2])
		print("overlap3=",overlap3.astype('int'))
		print("offset3=",offset3)

		#pT1=patch_extraction.patch_extract_3D(T1,(ps1,ps2,ps3),off1,off2,off3)
		pT1=patch_extraction.patch_extract_3D_v2(T1,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		pT2=patch_extraction.patch_extract_3D_v2(T2,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		pPD=patch_extraction.patch_extract_3D_v2(PD,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		pFLAIR=patch_extraction.patch_extract_3D_v2(FLAIR,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		pLABb=patch_extraction.patch_extract_3D_v2(LABb,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		# We select only the ii location (to be optimized)

		fileT1="stack_t1_1mm_"+str(i)+".npy"
		fileT2="stack_t2_1mm_"+str(i)+".npy"
		filePD="stack_pd_1mm_"+str(i)+".npy"
		fileFLAIR="stack_flair_1mm_"+str(i)+".npy"
		fileLAB="stack_lab_1mm_"+str(i)+".npy"

		pT1 = pT1.astype('float32')
		pT2 = pT2.astype('float32')
		pPD = pPD.astype('float32')

		pFLAIR = pFLAIR.astype('float32')
		pLABb = pLABb.astype('uint16')
		np.save(fileT1, pT1)
		np.save(fileT2, pT2)
		np.save(filePD, pPD)

		np.save(fileLAB, pLABb)
		np.save(fileFLAIR, pFLAIR)

		if Multiscale == 1:
			pSEG2mm=patch_extraction.patch_extract_3D_v2(SEG2mm,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
			fileSEG2mm="stack_seg2mm_1mm_"+str(i)+".npy"
			pSEG2mm = pSEG2mm.astype('float32')
			np.save(fileSEG2mm, pSEG2mm)

		if AtlasPrior > 0:
			pAtlas=patch_extraction.patch_extract_3D_v2(Atlas,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
			fileAtlas="stack_atlas_1mm_"+str(i)+".npy"
			pAtlas = pAtlas.astype('float32')
			#np.save(fileSEG2mm, pSEG2mm)
			np.save(fileAtlas, pAtlas)

	numLabels=2
	lista = np.array([0, 1])

	for ii in range(0,nbNN[0]*nbNN[1]*nbNN[2]):

		cpt = 0
		print(" ")
		print("Networks")
		print(str(ii+1))

		pLAB=np.empty([1, ps1, ps2, ps3, 1], dtype='uint8')

		pT1LastDim=4 # FLAIR + T1w
		if Multiscale == 1:
			pT1LastDim+=1
		if AtlasPrior > 0:
			pT1LastDim+=1

		cpt = 0
		for i in range(0,numfiles):

			fileLAB= "stack_lab_1mm_"+str(i)+".npy"
			pLABb = np.load(fileLAB, mmap_mode='r')
			pLABb = np.reshape(pLABb[ii], (1,pLABb.shape[1],pLABb.shape[2],pLABb.shape[3], 1))
			if np.sum(pLABb) > 0 : #to better balance the traning, at least one lesion has to be present in the tile
				#print("Size of lesions in image ",i," for the tile ",ii," = ", np.sum(pLABb))
				cpt = cpt +1

		x=np.empty([cpt, ps1, ps2, ps3, pT1LastDim], dtype='float32')
		print(x.shape)
		y=np.empty([cpt, ps1, ps2, ps3, 1], dtype='uint8')
		print(x.shape)

		cpt = 0
		for i in range(0,numfiles):

			fileLAB="stack_lab_1mm_"+str(i)+".npy"
			pLABb =np.load(fileLAB, mmap_mode='r')
			pLABb = np.reshape(pLABb[ii], (1,pLABb.shape[1],pLABb.shape[2],pLABb.shape[3], 1))

			if np.sum(pLABb) > 0 : #to better balance the traning, at least one lesion has to be present in the tile
				#print("Size of lesions in image ",i," for the tile ",ii," = ", np.sum(pLABb))
				fileT1="stack_t1_1mm_"+str(i)+".npy"
				pT1 =np.load(fileT1, mmap_mode='r')
				pT1 = np.reshape(pT1[ii], (1,pT1.shape[1],pT1.shape[2],pT1.shape[3], 1))

				fileT2="stack_t2_1mm_"+str(i)+".npy"
				pT2 =np.load(fileT2, mmap_mode='r')
				pT2 = np.reshape(pT2[ii], (1,pT1.shape[1],pT1.shape[2],pT1.shape[3], 1))

				filePD="stack_pd_1mm_"+str(i)+".npy"
				pPD =np.load(filePD, mmap_mode='r')
				pPD = np.reshape(pPD[ii], (1,pT1.shape[1],pT1.shape[2],pT1.shape[3], 1))


				fileFLAIR="stack_flair_1mm_"+str(i)+".npy"
				pFLAIR =np.load(fileFLAIR, mmap_mode='r')
				pFLAIR = np.reshape(pFLAIR[ii], (1,pT1.shape[1],pT1.shape[2],pT1.shape[3], 1))
				pT1=np.concatenate((pT1,pFLAIR,pT2,pPD), axis=4)


				if Multiscale == 1:
					fileSEG2mm="stack_seg2mm_1mm_"+str(i)+".npy"
					pSEG2mm =np.load(fileSEG2mm, mmap_mode='r')
					pSEG2mm = np.reshape(pSEG2mm[ii], (1,pSEG2mm.shape[1],pSEG2mm.shape[2],pSEG2mm.shape[3], 1))
					pT1=np.concatenate((pT1,pSEG2mm), axis=4) # Concatenate T1 and Seg from 2mm interpolated at 1mm

				if AtlasPrior > 0:
					fileAtlas="stack_atlas_1mm_"+str(i)+".npy"
					pAtlas =np.load(fileAtlas, mmap_mode='r')
					pAtlas = np.reshape(pAtlas[ii], (1,pAtlas.shape[1],pAtlas.shape[2],pAtlas.shape[3], 1))
					pT1=np.concatenate((pT1,pAtlas), axis=4) # Concatenate T1 and Seg from 2mm interpolated at 1mm

				#output
				for iii in lista:
					ind = np.where(pLABb == iii)
					pLAB[ind] = pLABb[ind]

				x[cpt, :, :, :, :] = pT1
				y[cpt, :, :, :, :] = pLAB
				cpt = cpt + 1


		print("sizeLAB=",y.shape)
		print("Number of used images = ",cpt," ( ", cpt/numfiles*100, " %) ")
		#x = x.astype('float32')
		#y = y.astype('uint16')

		#filex="tiles_img_1mm_"+str(ii)+".npy"
		#filey="tiles_lab_1mm_"+str(ii)+".npy"
		#filelist="list_lab_1mm_"+str(ii)+".npy"

		filex="tiles_img_1mm_"+str(ii)+".npy"
		filey="tiles_lab_1mm_"+str(ii)+".npy"
		filelist="list_lab_1mm_"+str(ii)+".npy"

		np.save(filex, x)
		np.save(filey, y)
		np.save(filelist,lista)

		#x = None
		#y = None

	os.chdir(path)

	return path,listaT1,lista







def save_Tile_1mm_4_preprocess(nbNN,ps, lib_path, AtlasPrior, Multiscale, FineTuning = 0, path_libteacher = None,is_kde=False):
	if(AtlasPrior == 'fine_tune'):
		AtlasPrior =0


	path=os.getcwd()
	#params
	ps1=ps[0]
	ps2=ps[1]
	ps3=ps[2]

	# read data
	os.chdir(lib_path)

	listaT1 = sorted(glob.glob("*mprage*.nii*"))
	listaT2 = sorted(glob.glob("*t2*.nii*"))
	listaPD = sorted(glob.glob("*pd*.nii*"))
	listaFLAIR = sorted(glob.glob("*flair*.nii*"))
	#listaMASK = sorted(glob.glob("mask*.nii*"))
	#listaLAB2= sorted(glob.glob("*mask_staple*.nii*"))
	listaLAB2= sorted(glob.glob("*mask2*.nii*"))
	listaLAB1= sorted(glob.glob("*mask1*.nii*"))


	numfiles=len(listaT1)




	for i in range(0,numfiles):
		#break
		print(" ")
		print("Images")
		print(str(i+1))
		print(listaT1[i])
		print(listaT2[i])
		print(listaPD[i])
		print(listaFLAIR[i])
		print(listaLAB1[i])
		print(listaLAB2[i])
		#print(listaMASK[i])
		if AtlasPrior > 0:
			print(listaatlasprior[i])
		if Multiscale == 1:
			print(listaSEG2mm[i])

		T1_img = nii.load(listaT1[i])
		T2_img = nii.load(listaT2[i])
		PD_img = nii.load(listaPD[i])
		FLAIR_img = nii.load(listaFLAIR[i])
		LAB1_img = nii.load(listaLAB1[i])
		LAB2_img = nii.load(listaLAB2[i])
		T1=T1_img.get_data()
		T2=T2_img.get_data()
		PD=PD_img.get_data()
		FLAIR=FLAIR_img.get_data()
		LABb1=LAB1_img.get_data()
		LABb2=LAB2_img.get_data()
		LABb1=LABb1.astype('int')
		LABb2=LABb2.astype('int')

		MASK = (1-(T1==0).astype('int'))
		ind=np.where(MASK>0)
		indbg=np.where(MASK==0)
		MASK_img=(nii.Nifti1Image(MASK, T1_img.affine ))

		if(is_kde):

			peak = normalize_image(T1, 't1')
			T1=T1/peak
			peak = normalize_image(FLAIR, 'flair')
			FLAIR=FLAIR/peak
			peak = normalize_image(T2, 't2')
			T2=T2/peak
			peak = normalize_image(PD, 'pd')
			PD=PD/peak
		else:

			T2[indbg] = 0
			m1=np.mean(T2[ind])
			s1=np.std(T2[ind])
			T2[ind]=(T2[ind]-m1)/s1
			PD[indbg] = 0
			m1=np.mean(PD[ind])
			s1=np.std(PD[ind])
			PD[ind]=(PD[ind]-m1)/s1


			T1[indbg] = 0
			m1=np.mean(T1[ind])
			s1=np.std(T1[ind])
			T1[ind]=(T1[ind]-m1)/s1
			#T1[ind]=(T1[ind]-m1)/s1

			FLAIR[indbg] = 0
			m1=np.mean(FLAIR[ind])
			s1=np.std(FLAIR[ind])
			FLAIR[ind]=(FLAIR[ind]-m1)/s1


		crop_bg = 4 # To crop null border.
		overlap1 = np.floor((nbNN[0]*ps1 - (T1.shape[0]-2*crop_bg)) / (nbNN[0]-1))
		offset1 = ps1 - overlap1.astype('int')
		#print("image size1",T1.shape[0])
		print("overlap1=",overlap1.astype('int'))
		print("offset1=",offset1)
		overlap2 = np.floor((nbNN[1]*ps2 - (T1.shape[1]-2*crop_bg)) / (nbNN[1]-1))
		offset2 = ps2 - overlap2.astype('int')
		#print("image size2",T1.shape[1])
		print("overlap2=",overlap2.astype('int'))
		print("offset2=",offset2)
		overlap3 = np.floor((nbNN[2]*ps3 - (T1.shape[2]-2*crop_bg)) / (nbNN[2]-1))
		offset3 = ps3 - overlap3.astype('int')
		#print("image size3",T1.shape[2])
		print("overlap3=",overlap3.astype('int'))
		print("offset3=",offset3)

		#pT1=patch_extraction.patch_extract_3D(T1,(ps1,ps2,ps3),off1,off2,off3)
		pT1=patch_extraction.patch_extract_3D_v2(T1,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		pT2=patch_extraction.patch_extract_3D_v2(T2,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		pPD=patch_extraction.patch_extract_3D_v2(PD,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		pFLAIR=patch_extraction.patch_extract_3D_v2(FLAIR,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		pLABb2=patch_extraction.patch_extract_3D_v2(LABb2,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		pLABb1=patch_extraction.patch_extract_3D_v2(LABb1,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		# We select only the ii location (to be optimized)

		fileT1="stack_t1_1mm_"+str(i)+".npy"
		fileT2="stack_t2_1mm_"+str(i)+".npy"
		filePD="stack_pd_1mm_"+str(i)+".npy"
		fileFLAIR="stack_flair_1mm_"+str(i)+".npy"
		fileLAB1="stack_lab1_1mm_"+str(i)+".npy"
		fileLAB2="stack_lab2_1mm_"+str(i)+".npy"

		pT1 = pT1.astype('float32')
		pT2 = pT2.astype('float32')
		pPD = pPD.astype('float32')

		pFLAIR = pFLAIR.astype('float32')
		pLABb1 = pLABb1.astype('uint16')
		pLABb2 = pLABb2.astype('uint16')
		np.save(fileT1, pT1)
		np.save(fileT2, pT2)
		np.save(filePD, pPD)

		np.save(fileLAB1, pLABb1)
		np.save(fileLAB2, pLABb2)
		np.save(fileFLAIR, pFLAIR)



	numLabels=2
	lista = np.array([0, 1])

	for ii in range(0,nbNN[0]*nbNN[1]*nbNN[2]):

		cpt = 0
		print(" ")
		print("Networks")
		print(str(ii+1))

		pLAB1=np.empty([1, ps1, ps2, ps3, 1], dtype='uint8')
		pLAB2=np.empty([1, ps1, ps2, ps3, 1], dtype='uint8')

		pT1LastDim=4 # FLAIR + T1w



		cpt = 0
		for i in range(0,numfiles):

			fileLAB2= "stack_lab2_1mm_"+str(i)+".npy"
			fileLAB1= "stack_lab1_1mm_"+str(i)+".npy"

			pLABb1 = np.load(fileLAB1, mmap_mode='r')
			pLABb1 = np.reshape(pLABb1[ii], (1,pLABb1.shape[1],pLABb1.shape[2],pLABb1.shape[3], 1))

			pLABb2 = np.load(fileLAB2, mmap_mode='r')
			pLABb2 = np.reshape(pLABb2[ii], (1,pLABb2.shape[1],pLABb2.shape[2],pLABb2.shape[3], 1))

			if np.sum(pLABb1) > 0 : #to better balance the traning, at least one lesion has to be present in the tile
				#print("Size of lesions in image ",i," for the tile ",ii," = ", np.sum(pLABb))
				cpt = cpt +1

		x=np.empty([cpt, ps1, ps2, ps3, pT1LastDim], dtype='float32')
		print(x.shape)
		y=np.empty([cpt, ps1, ps2, ps3, 1], dtype='uint8')
		y_=np.empty([cpt, ps1, ps2, ps3, 1], dtype='uint8')
		print(x.shape)

		cpt = 0
		for i in range(0,numfiles):

			fileLAB2= "stack_lab2_1mm_"+str(i)+".npy"
			fileLAB1= "stack_lab1_1mm_"+str(i)+".npy"

			pLABb1 = np.load(fileLAB1, mmap_mode='r')
			pLABb1 = np.reshape(pLABb1[ii], (1,pLABb1.shape[1],pLABb1.shape[2],pLABb1.shape[3], 1))

			pLABb2 = np.load(fileLAB2, mmap_mode='r')
			pLABb2 = np.reshape(pLABb2[ii], (1,pLABb2.shape[1],pLABb2.shape[2],pLABb2.shape[3], 1))

			if np.sum(pLABb1) > 0 : #to better balance the traning, at least one lesion has to be present in the tile
				#print("Size of lesions in image ",i," for the tile ",ii," = ", np.sum(pLABb))
				fileT1="stack_t1_1mm_"+str(i)+".npy"
				pT1 =np.load(fileT1, mmap_mode='r')
				pT1 = np.reshape(pT1[ii], (1,pT1.shape[1],pT1.shape[2],pT1.shape[3], 1))

				fileT2="stack_t2_1mm_"+str(i)+".npy"
				pT2 =np.load(fileT2, mmap_mode='r')
				pT2 = np.reshape(pT2[ii], (1,pT1.shape[1],pT1.shape[2],pT1.shape[3], 1))

				filePD="stack_pd_1mm_"+str(i)+".npy"
				pPD =np.load(filePD, mmap_mode='r')
				pPD = np.reshape(pPD[ii], (1,pT1.shape[1],pT1.shape[2],pT1.shape[3], 1))


				fileFLAIR="stack_flair_1mm_"+str(i)+".npy"
				pFLAIR =np.load(fileFLAIR, mmap_mode='r')
				pFLAIR = np.reshape(pFLAIR[ii], (1,pT1.shape[1],pT1.shape[2],pT1.shape[3], 1))
				pT1=np.concatenate((pT1,pFLAIR,pT2,pPD), axis=4)




				#output
				for iii in lista:
					ind = np.where(pLABb1 == iii)
					pLAB1[ind] = pLABb1[ind]

				for iii in lista:
					ind = np.where(pLABb2 == iii)
					pLAB2[ind] = pLABb2[ind]

				x[cpt, :, :, :, :] = pT1
				y[cpt, :, :, :, :] = pLAB1
				y_[cpt, :, :, :, :] = pLAB2
				cpt = cpt + 1


		print("sizeLAB=",y.shape)
		print("Number of used images = ",cpt," ( ", cpt/numfiles*100, " %) ")
		#x = x.astype('float32')
		#y = y.astype('uint16')

		#filex="tiles_img_1mm_"+str(ii)+".npy"
		#filey="tiles_lab_1mm_"+str(ii)+".npy"
		#filelist="list_lab_1mm_"+str(ii)+".npy"

		filex="tiles_img_1mm_"+str(ii)+".npy"
		filey="tiles_lab1_1mm_"+str(ii)+".npy"
		filey2="tiles_lab2_1mm_"+str(ii)+".npy"
		#filey2="tiles_lab_staple_1mm_"+str(ii)+".npy"
		filelist="list_lab_1mm_"+str(ii)+".npy"

		np.save(filex, x)
		np.save(filey, y)
		np.save(filey2, y_)
		np.save(filelist,lista)

		#x = None
		#y = None

	os.chdir(path)

	return path,listaT1,lista






def save_Tile_1mm_2_preprocess(nbNN,ps, lib_path, AtlasPrior, Multiscale, FineTuning = 0, path_libteacher = None,is_kde=False):
	if(AtlasPrior == 'fine_tune'):
		AtlasPrior =0


	path=os.getcwd()
	#params
	ps1=ps[0]
	ps2=ps[1]
	ps3=ps[2]

	# read data
	os.chdir(lib_path)

	listaT1 = sorted(glob.glob("*t1*.nii*"))
	listaFLAIR = sorted(glob.glob("*flair*.nii*"))
	llistaMASK = sorted(glob.glob("mask*.nii*"))
	listaLAB1= sorted(glob.glob("*lesion*.nii*"))


	numfiles=len(listaT1)




	for i in range(0,numfiles):
		#break
		print(" ")
		print("Images")
		print(str(i+1))
		print(listaT1[i])
		print(listaFLAIR[i])
		print(listaLAB1[i])
		#print(listaMASK[i])
		if AtlasPrior > 0:
			print(listaatlasprior[i])
		if Multiscale == 1:
			print(listaSEG2mm[i])

		T1_img = nii.load(listaT1[i])

		FLAIR_img = nii.load(listaFLAIR[i])
		LAB1_img = nii.load(listaLAB1[i])

		T1=T1_img.get_data()

		FLAIR=FLAIR_img.get_data()
		LABb1=LAB1_img.get_data()
		LABb1=LABb1.astype('int')

		MASK_img=nii.load(listaMASK[i])
		MASK = MASK_img.get_data().astype('int')
		#MASK = (1-(T1==0).astype('int'))
		#MASK_img=(nii.Nifti1Image(MASK, T1_img.affine ))
		ind=np.where(MASK>0)
		indbg=np.where(MASK==0)

		if(is_kde):
			T1=T1*MASK
			FLAIR=FLAIR*MASK
			peak = normalize_image(T1, 't1')
			T1=T1/peak
			peak = normalize_image(FLAIR, 'flair')
			FLAIR=FLAIR/peak

		else:


			T1[indbg] = 0
			m1=np.mean(T1[ind])
			s1=np.std(T1[ind])
			T1[ind]=(T1[ind]-m1)/s1
			#T1[ind]=(T1[ind]-m1)/s1

			FLAIR[indbg] = 0
			m1=np.mean(FLAIR[ind])
			s1=np.std(FLAIR[ind])
			FLAIR[ind]=(FLAIR[ind]-m1)/s1


		crop_bg = 4 # To crop null border.
		overlap1 = np.floor((nbNN[0]*ps1 - (T1.shape[0]-2*crop_bg)) / (nbNN[0]-1))
		offset1 = ps1 - overlap1.astype('int')
		#print("image size1",T1.shape[0])
		print("overlap1=",overlap1.astype('int'))
		print("offset1=",offset1)
		overlap2 = np.floor((nbNN[1]*ps2 - (T1.shape[1]-2*crop_bg)) / (nbNN[1]-1))
		offset2 = ps2 - overlap2.astype('int')
		#print("image size2",T1.shape[1])
		print("overlap2=",overlap2.astype('int'))
		print("offset2=",offset2)
		overlap3 = np.floor((nbNN[2]*ps3 - (T1.shape[2]-2*crop_bg)) / (nbNN[2]-1))
		offset3 = ps3 - overlap3.astype('int')
		#print("image size3",T1.shape[2])
		print("overlap3=",overlap3.astype('int'))
		print("offset3=",offset3)

		#pT1=patch_extraction.patch_extract_3D(T1,(ps1,ps2,ps3),off1,off2,off3)
		pT1=patch_extraction.patch_extract_3D_v2(T1,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)

		pFLAIR=patch_extraction.patch_extract_3D_v2(FLAIR,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)

		pLABb1=patch_extraction.patch_extract_3D_v2(LABb1,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		# We select only the ii location (to be optimized)

		fileT1="stack_t1_1mm_"+str(i)+".npy"

		fileFLAIR="stack_flair_1mm_"+str(i)+".npy"
		fileLAB1="stack_lab1_1mm_"+str(i)+".npy"


		pT1 = pT1.astype('float32')

		pFLAIR = pFLAIR.astype('float32')
		pLABb1 = pLABb1.astype('uint16')

		np.save(fileT1, pT1)

		np.save(fileLAB1, pLABb1)

		np.save(fileFLAIR, pFLAIR)



	numLabels=2
	lista = np.array([0, 1])

	for ii in range(0,nbNN[0]*nbNN[1]*nbNN[2]):

		cpt = 0
		print(" ")
		print("Networks")
		print(str(ii+1))

		pLAB1=np.empty([1, ps1, ps2, ps3, 1], dtype='uint8')

		pT1LastDim=2 # FLAIR + T1w



		cpt = 0
		for i in range(0,numfiles):

			fileLAB1= "stack_lab1_1mm_"+str(i)+".npy"

			pLABb1 = np.load(fileLAB1, mmap_mode='r')
			pLABb1 = np.reshape(pLABb1[ii], (1,pLABb1.shape[1],pLABb1.shape[2],pLABb1.shape[3], 1))


			if np.sum(pLABb1) > 0 : #to better balance the traning, at least one lesion has to be present in the tile
				#print("Size of lesions in image ",i," for the tile ",ii," = ", np.sum(pLABb))
				cpt = cpt +1

		x=np.empty([cpt, ps1, ps2, ps3, pT1LastDim], dtype='float32')
		print(x.shape)
		y=np.empty([cpt, ps1, ps2, ps3, 1], dtype='uint8')

		print(x.shape)

		cpt = 0
		for i in range(0,numfiles):


			fileLAB1= "stack_lab1_1mm_"+str(i)+".npy"

			pLABb1 = np.load(fileLAB1, mmap_mode='r')
			pLABb1 = np.reshape(pLABb1[ii], (1,pLABb1.shape[1],pLABb1.shape[2],pLABb1.shape[3], 1))

			if np.sum(pLABb1) > 0 : #to better balance the traning, at least one lesion has to be present in the tile
				#print("Size of lesions in image ",i," for the tile ",ii," = ", np.sum(pLABb))
				fileT1="stack_t1_1mm_"+str(i)+".npy"
				pT1 =np.load(fileT1, mmap_mode='r')
				pT1 = np.reshape(pT1[ii], (1,pT1.shape[1],pT1.shape[2],pT1.shape[3], 1))

				fileFLAIR="stack_flair_1mm_"+str(i)+".npy"
				pFLAIR =np.load(fileFLAIR, mmap_mode='r')
				pFLAIR = np.reshape(pFLAIR[ii], (1,pT1.shape[1],pT1.shape[2],pT1.shape[3], 1))
				pT1=np.concatenate((pT1,pFLAIR), axis=4)




				#output
				for iii in lista:
					ind = np.where(pLABb1 == iii)
					pLAB1[ind] = pLABb1[ind]

				x[cpt, :, :, :, :] = pT1
				y[cpt, :, :, :, :] = pLAB1

				cpt = cpt + 1


		print("sizeLAB=",y.shape)
		print("Number of used images = ",cpt," ( ", cpt/numfiles*100, " %) ")
		#x = x.astype('float32')
		#y = y.astype('uint16')

		#filex="tiles_img_1mm_"+str(ii)+".npy"
		#filey="tiles_lab_1mm_"+str(ii)+".npy"
		#filelist="list_lab_1mm_"+str(ii)+".npy"

		filex="tiles_img_1mm_"+str(ii)+".npy"
		filey="tiles_lab1_1mm_"+str(ii)+".npy"

		filelist="list_lab_1mm_"+str(ii)+".npy"

		np.save(filex, x)
		np.save(filey, y)

		np.save(filelist,lista)

		#x = None
		#y = None

	os.chdir(path)

	return path,listaT1,lista

def save_Tile_1mm_(nbNN,ps, lib_path, AtlasPrior, Multiscale, FineTuning = 0, path_libteacher = None):

	path=os.getcwd()
	#params
	ps1=ps[0]
	ps2=ps[1]
	ps3=ps[2]

	# read data
	os.chdir(lib_path)

	if("volBrain" in lib_path):

		listaT1 = sorted(glob.glob("n_*t1*.nii"))
		listaFLAIR = sorted(glob.glob("n_*flair*.nii"))
		listaLAB = sorted(glob.glob("wmhlesion*.nii*"))
		listaMASK = sorted(glob.glob("mask*.nii"))

	else:
		listaT1 = sorted(glob.glob("DLB*t1*.nii*"))
		listaFLAIR = sorted(glob.glob("DLB*flair*.nii*"))
		listaLAB = sorted(glob.glob("DLB*lesion*.nii*"))
		listaMASK = sorted(glob.glob("DLB*mask*.nii*"))

	numfiles=len(listaT1)


	# os.chdir(path)
	# os.chdir('results')
	# Seg from 2mm scale interpolated at 1mm
	listaSEG2mm = sorted(glob.glob("up_seg_2mm_*.nii*"))

	if (AtlasPrior == 1):
		listaatlasprior = sorted(glob.glob("wlab_*.nii*"))
	if (AtlasPrior == 2):
		listaatlasprior = sorted(glob.glob("Assembly_seg_1mm_n_mmni_*.nii*"))
		listaSEG2mm = sorted(glob.glob("up_seg_2mm_2pass_n_*.nii*"))


	for i in range(0,numfiles):

		print(" ")
		print("Images")
		print(str(i+1))
		print(listaT1[i])
		print(listaFLAIR[i])
		print(listaMASK[i])
		print(listaLAB[i])
		if AtlasPrior > 0:
			print(listaatlasprior[i])
		if Multiscale == 1:
			print(listaSEG2mm[i])

		T1_img = nii.load(listaT1[i])
		FLAIR_img = nii.load(listaFLAIR[i])
		LAB_img = nii.load(listaLAB[i])
		MASK_img = nii.load(listaMASK[i])
		T1=T1_img.get_data()
		FLAIR=FLAIR_img.get_data()
		LABb=LAB_img.get_data()
		MASK = MASK_img.get_data()
		LABb=LABb.astype('int')
		MASK=MASK.astype('int')
		ind=np.where(MASK>0)
		indbg=np.where(MASK==0)

		if AtlasPrior > 0:
			Atlas_img = nii.load(listaatlasprior[i])
			Atlas=Atlas_img.get_data()
			Atlas=Atlas.astype('float32')
			Atlasc = np.zeros(Atlas.shape)
			for indexlab, lab in enumerate(labels_SLANT):
				indl=np.where(Atlas==lab)
				Atlasc[indl] = Atlas[indl]
			print('Removed inconsistent labels Atlas: ', list(set(np.unique(Atlas)) - set(np.unique(Atlasc))))
			Atlas = Atlasc
			Atlas[indbg] = 0
			#inda=np.where(Atlas>0)
			m1=np.mean(Atlas[ind])
			s1=np.std(Atlas[ind])
			Atlas=(Atlas-m1)/s1

		if Multiscale == 1:
			# os.chdir(path)
			# os.chdir('results')
			# Seg from 2mm scale interpolated at 1mm
			SEG2mm_img = nii.load(listaSEG2mm[i])
			SEG2mm=SEG2mm_img.get_data()
			SEG2mm=SEG2mm.astype('float32')
			SEG2mm[indbg] = 0
			#indm=np.where(SEG2mm>0)
			m1=np.mean(SEG2mm[ind])
			s1=np.std(SEG2mm[ind])
			SEG2mm=(SEG2mm-m1)/s1
			# os.chdir(path)
			# os.chdir(lib_path)


		#normalization
		T1[indbg] = 0
		m1=np.mean(T1[ind])
		s1=np.std(T1[ind])
		T1[ind]=(T1[ind]-m1)/s1
		#T1[ind]=(T1[ind]-m1)/s1

		FLAIR[indbg] = 0
		m1=np.mean(FLAIR[ind])
		s1=np.std(FLAIR[ind])
		FLAIR[ind]=(FLAIR[ind]-m1)/s1

		# LAB = np.zeros(LABb.shape)
		# for indexlab, lab in enumerate(labels_SLANT):
		# 	ind=np.where(LABb==lab)
		# 	LAB[ind] = LABb[ind]
		# print('Removed inconsistent labels : ', list(set(np.unique(LABb)) - set(np.unique(LAB))))
		# LABb = LAB


		#subvolume extraction
		# We read the N subvolumes
		# First we estimate the overlap
		crop_bg = 4 # To crop null border.
		overlap1 = np.floor((nbNN[0]*ps1 - (T1.shape[0]-2*crop_bg)) / (nbNN[0]-1))
		offset1 = ps1 - overlap1.astype('int')
		#print("image size1",T1.shape[0])
		print("overlap1=",overlap1.astype('int'))
		print("offset1=",offset1)
		overlap2 = np.floor((nbNN[1]*ps2 - (T1.shape[1]-2*crop_bg)) / (nbNN[1]-1))
		offset2 = ps2 - overlap2.astype('int')
		#print("image size2",T1.shape[1])
		print("overlap2=",overlap2.astype('int'))
		print("offset2=",offset2)
		overlap3 = np.floor((nbNN[2]*ps3 - (T1.shape[2]-2*crop_bg)) / (nbNN[2]-1))
		offset3 = ps3 - overlap3.astype('int')
		#print("image size3",T1.shape[2])
		print("overlap3=",overlap3.astype('int'))
		print("offset3=",offset3)

		#pT1=patch_extraction.patch_extract_3D(T1,(ps1,ps2,ps3),off1,off2,off3)
		pT1=patch_extraction.patch_extract_3D_v2(T1,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		pFLAIR=patch_extraction.patch_extract_3D_v2(FLAIR,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		pLABb=patch_extraction.patch_extract_3D_v2(LABb,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		# We select only the ii location (to be optimized)

		fileT1="stack_t1_1mm_"+str(i)+".npy"
		fileFLAIR="stack_flair_1mm_"+str(i)+".npy"
		fileLAB="stack_lab_1mm_"+str(i)+".npy"

		pT1 = pT1.astype('float32')
		pFLAIR = pFLAIR.astype('float32')
		pLABb = pLABb.astype('uint16')
		np.save(fileT1, pT1)
		np.save(fileLAB, pLABb)
		np.save(fileFLAIR, pFLAIR)

		if Multiscale == 1:
			pSEG2mm=patch_extraction.patch_extract_3D_v2(SEG2mm,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
			fileSEG2mm="stack_seg2mm_1mm_"+str(i)+".npy"
			pSEG2mm = pSEG2mm.astype('float32')
			np.save(fileSEG2mm, pSEG2mm)

		if AtlasPrior > 0:
			pAtlas=patch_extraction.patch_extract_3D_v2(Atlas,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
			fileAtlas="stack_atlas_1mm_"+str(i)+".npy"
			pAtlas = pAtlas.astype('float32')
			#np.save(fileSEG2mm, pSEG2mm)
			np.save(fileAtlas, pAtlas)

	numLabels=2
	lista = np.array([0, 1])

	for ii in range(0,nbNN[0]*nbNN[1]*nbNN[2]):

		cpt = 0
		print(" ")
		print("Networks")
		print(str(ii+1))

		pLAB=np.empty([1, ps1, ps2, ps3, 1], dtype='uint8')

		pT1LastDim=2 # FLAIR + T1w
		if Multiscale == 1:
			pT1LastDim+=1
		if AtlasPrior > 0:
			pT1LastDim+=1

		cpt = 0
		for i in range(0,numfiles):

			fileLAB= "stack_lab_1mm_"+str(i)+".npy"
			pLABb = np.load(fileLAB, mmap_mode='r')
			pLABb = np.reshape(pLABb[ii], (1,pLABb.shape[1],pLABb.shape[2],pLABb.shape[3], 1))
			if np.sum(pLABb) > 0 : #to better balance the traning, at least one lesion has to be present in the tile
				#print("Size of lesions in image ",i," for the tile ",ii," = ", np.sum(pLABb))
				cpt = cpt +1

		x=np.empty([cpt, ps1, ps2, ps3, pT1LastDim], dtype='float32')
		print(x.shape)
		y=np.empty([cpt, ps1, ps2, ps3, 1], dtype='uint8')
		print(x.shape)

		cpt = 0
		for i in range(0,numfiles):

			fileLAB="stack_lab_1mm_"+str(i)+".npy"
			pLABb =np.load(fileLAB, mmap_mode='r')
			pLABb = np.reshape(pLABb[ii], (1,pLABb.shape[1],pLABb.shape[2],pLABb.shape[3], 1))

			if np.sum(pLABb) > 0 : #to better balance the traning, at least one lesion has to be present in the tile
				#print("Size of lesions in image ",i," for the tile ",ii," = ", np.sum(pLABb))
				fileT1="stack_t1_1mm_"+str(i)+".npy"
				pT1 =np.load(fileT1, mmap_mode='r')
				pT1 = np.reshape(pT1[ii], (1,pT1.shape[1],pT1.shape[2],pT1.shape[3], 1))
				fileFLAIR="stack_flair_1mm_"+str(i)+".npy"
				pFLAIR =np.load(fileFLAIR, mmap_mode='r')
				pFLAIR = np.reshape(pFLAIR[ii], (1,pT1.shape[1],pT1.shape[2],pT1.shape[3], 1))
				pT1=np.concatenate((pT1,pFLAIR), axis=4)


				if Multiscale == 1:
					fileSEG2mm="stack_seg2mm_1mm_"+str(i)+".npy"
					pSEG2mm =np.load(fileSEG2mm, mmap_mode='r')
					pSEG2mm = np.reshape(pSEG2mm[ii], (1,pSEG2mm.shape[1],pSEG2mm.shape[2],pSEG2mm.shape[3], 1))
					pT1=np.concatenate((pT1,pSEG2mm), axis=4) # Concatenate T1 and Seg from 2mm interpolated at 1mm

				if AtlasPrior > 0:
					fileAtlas="stack_atlas_1mm_"+str(i)+".npy"
					pAtlas =np.load(fileAtlas, mmap_mode='r')
					pAtlas = np.reshape(pAtlas[ii], (1,pAtlas.shape[1],pAtlas.shape[2],pAtlas.shape[3], 1))
					pT1=np.concatenate((pT1,pAtlas), axis=4) # Concatenate T1 and Seg from 2mm interpolated at 1mm

				#output
				for iii in lista:
					ind = np.where(pLABb == iii)
					pLAB[ind] = pLABb[ind]

				x[cpt, :, :, :, :] = pT1
				y[cpt, :, :, :, :] = pLAB
				cpt = cpt + 1


		print("sizeLAB=",y.shape)
		print("Number of used images = ",cpt," ( ", cpt/numfiles*100, " %) ")
		#x = x.astype('float32')
		#y = y.astype('uint16')

		#filex="tiles_img_1mm_"+str(ii)+".npy"
		#filey="tiles_lab_1mm_"+str(ii)+".npy"
		#filelist="list_lab_1mm_"+str(ii)+".npy"

		filex="tiles_img_1mm_"+str(ii)+"_.npy"
		filey="tiles_lab_1mm_"+str(ii)+"_.npy"
		filelist="list_lab_1mm_"+str(ii)+"_.npy"

		np.save(filex, x)
		np.save(filey, y)
		np.save(filelist,lista)

		#x = None
		#y = None

	os.chdir(path)

	return path,listaT1,lista

def save_Tile_2mm(nbNN,ps, lib_path, AtlasPrior, FineTuning = 0, path_libteacher = None):

	path=os.getcwd()

	#params
	ps1=ps[0]
	ps2=ps[1]
	ps3=ps[2]

	os.chdir(lib_path)
	listaT1 = sorted(glob.glob("DLB*t1*.nii"))
	listaFLAIR = sorted(glob.glob("DLB*flair*.nii"))
	listaLAB = sorted(glob.glob("DLB*lesion*.nii"))
	listaMASK = sorted(glob.glob("DLB*mask*.nii"))


	numfiles=len(listaT1)

	#if (AtlasPrior == 1):
		# Atlas prior obtained by warping MICCAI atlas on the subject
		#listaatlasprior = sorted(glob.glob("wlab_n_mmni_*.nii"))
	if (AtlasPrior == 1):
		listaatlasprior = sorted(glob.glob("wlab_*.nii*"))
	if (AtlasPrior == 2):
		listaatlasprior = sorted(glob.glob("Assembly_seg_1mm_n_mmni_*.nii*"))


	for i in range(0,numfiles):

		print(" ")
		print("Images")
		print(str(i+1))

		print(listaT1[i])
		print(listaFLAIR[i])
		print(listaMASK[i])
		print(listaLAB[i])

		T1_img = nii.load(listaT1[i])
		FLAIR_img = nii.load(listaFLAIR[i])
		LAB_img = nii.load(listaLAB[i])
		MASK_img = nii.load(listaMASK[i])

		T1=T1_img.get_data()
		T1=T1.astype('float32')
		T1 =T1[::2,::2,::2] # down-smapling

		FLAIR=FLAIR_img.get_data()
		FLAIR=FLAIR.astype('float32')
		FLAIR =FLAIR[::2,::2,::2] # down-smapling

		LABb=LAB_img.get_data()
		LABb=LABb.astype('int')

		MASK = MASK_img.get_data()
		MASK=MASK.astype('int')
		MASK = MASK[::2,::2,::2] # down-smapling
		ind=np.where(MASK>0)
		indbg=np.where(MASK==0)

		if (AtlasPrior > 0):
			print(listaatlasprior[i])
			Atlas_img = nii.load(listaatlasprior[i])
			Atlas=Atlas_img.get_data()
			Atlas=Atlas.astype('float32')
			Atlas = Atlas[::2,::2,::2]
			#remove incomplete labels (not in all cases)
			Atlasc = np.zeros(Atlas.shape)
			for indexlab, lab in enumerate(labels_SLANT):
				indl=np.where(Atlas==lab)
				Atlasc[indl] = Atlas[indl]
			print('Removed inconsistent labels in Atlas: ', list(set(np.unique(Atlas)) - set(np.unique(Atlasc))))
			Atlas = Atlasc
			Atlas[indbg] = 0 # masked atlas priors
			#inda=np.where(Atlas>0)
			m1=np.mean(Atlas[ind])
			s1=np.std(Atlas[ind])
			Atlas=(Atlas-m1)/s1

		T1[indbg] = 0 #masked image
		m1=np.mean(T1[ind])
		s1=np.std(T1[ind])
		T1=(T1-m1)/s1

		FLAIR[indbg] = 0
		m1=np.mean(FLAIR[ind])
		s1=np.std(FLAIR[ind])
		FLAIR=(FLAIR-m1)/s1

		#subvolume extraction
		# We read the N subvolumes
		# First we estimate the overlap
		crop_bg = 2 # To crop null border.
		overlap1 = np.floor((nbNN[0]*ps1 - (T1.shape[0]-2*crop_bg)) / (nbNN[0]-1))
		offset1 = ps1 - overlap1.astype('int')
		#print("image size1",T1.shape[0])
		#print("overlap1=",overlap1.astype('int'))
		#print("offset1=",offset1)
		overlap2 = np.floor((nbNN[1]*ps2 - (T1.shape[1]-2*crop_bg)) / (nbNN[1]-1))
		offset2 = ps2 - overlap2.astype('int')
		#print("image size2",T1.shape[1])
		#print("overlap2=",overlap2.astype('int'))
		#print("offset2=",offset2)
		overlap3 = np.floor((nbNN[2]*ps3 - (T1.shape[2]-2*crop_bg)) / (nbNN[2]-1))
		offset3 = ps3 - overlap3.astype('int')
		#print("image size3",T1.shape[2])
		#print("overlap3=",overlap3.astype('int'))
		#print("offset3=",offset3)

		#pT1=patch_extraction.patch_extract_3D(T1,(ps1,ps2,ps3),off1,off2,off3)
		pT1=patch_extraction.patch_extract_3D_v2(T1,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		pFLAIR=patch_extraction.patch_extract_3D_v2(FLAIR,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		pLABb=patch_extraction.patch_extract_3D_v2(LABb,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
		# We select only the ii location (to be optimized)

		fileT1="stack_t1_2mm_"+str(i)+".npy"
		fileFLAIR="stack_flair_2mm_"+str(i)+".npy"
		fileLAB="stack_lab_2mm_"+str(i)+".npy"
		pT1 = pT1.astype('float32')
		pFLAIR = pFLAIR.astype('float32')
		pLABb = pLABb.astype('uint16')
		np.save(fileT1, pT1)
		np.save(fileLAB, pLABb)
		np.save(fileFLAIR, pFLAIR)

		if AtlasPrior > 0:
			pAtlas=patch_extraction.patch_extract_3D_v2(Atlas,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
			fileAtlas="stack_atlas_2mm_"+str(i)+".npy"
			pAtlas = pAtlas.astype('float32')
			np.save(fileAtlas, pAtlas)

	numLabels=2
	lista = np.array([0, 1])

	for ii in range(0,nbNN[0]*nbNN[1]*nbNN[2]):


		print(" ")
		print("Networks")
		print(str(ii+1))

		pLAB=np.empty([1, ps1, ps2, ps3, 1], dtype='uint8')

		pT1LastDim=2
		if AtlasPrior > 0:
			pT1LastDim+=1

		cpt = 0
		for i in range(0,numfiles):

			fileLAB= "stack_lab_2mm_"+str(i)+".npy"
			pLABb = np.load(fileLAB, mmap_mode='r')
			pLABb = np.reshape(pLABb[ii], (1,pLABb.shape[1],pLABb.shape[2],pLABb.shape[3], 1))
			if np.sum(pLABb) > 0 : #to better balance the traning, at least one lesion has to be present in the tile
				#print("Size of lesions in image ",i," for the tile ",ii," = ", np.sum(pLABb))
				cpt = cpt +1

		x=np.empty([cpt, ps1, ps2, ps3, pT1LastDim], dtype='float32')
		y=np.empty([cpt, ps1, ps2, ps3, 1], dtype='uint8')

		cpt = 0
		for i in range(0,numfiles):

			fileLAB="stack_lab_2mm_"+str(i)+".npy"
			pLABb =np.load(fileLAB, mmap_mode='r')
			pLABb = np.reshape(pLABb[ii], (1,pLABb.shape[1],pLABb.shape[2],pLABb.shape[3], 1))

			if np.sum(pLABb) > 0 : #to better balance the traning, at least one lesion has to be present in the tile
				#print("Size of lesions in image ",i," for the tile ",ii," = ", np.sum(pLABb))
				fileT1="stack_t1_2mm_"+str(i)+".npy"
				pT1 =np.load(fileT1, mmap_mode='r')
				pT1 = np.reshape(pT1[ii], (1,pT1.shape[1],pT1.shape[2],pT1.shape[3], 1))
				fileFLAIR="stack_flair_2mm_"+str(i)+".npy"
				pFLAIR =np.load(fileFLAIR, mmap_mode='r')
				pFLAIR = np.reshape(pFLAIR[ii], (1,pT1.shape[1],pT1.shape[2],pT1.shape[3], 1))
				pT1=np.concatenate((pT1,pFLAIR), axis=4)

				if AtlasPrior > 0:
					fileAtlas="stack_atlas_2mm_"+str(i)+".npy"
					pAtlas =np.load(fileAtlas, mmap_mode='r')
					pAtlas = np.reshape(pAtlas[ii], (1,pAtlas.shape[1],pAtlas.shape[2],pAtlas.shape[3], 1))
					pT1=np.concatenate((pT1,pAtlas), axis=4) # Concatenate T1 and Seg from 2mm interpolated at 1mm

				for iii in lista:
					ind = np.where(pLABb == iii)
					pLAB[ind] = pLABb[ind]

				x[cpt, :, :, :, :] = pT1
				y[cpt, :, :, :, :] = pLAB
				cpt = cpt + 1


		print("sizeLAB=",y.shape)
		print("Number of used images = ",cpt," ( ", cpt/numfiles*100, " %) ")
		#x = x.astype('float32')
		#y = y.astype('uint16')

		filex="tiles_img_2mm_"+str(ii)+".npy"
		filey="tiles_lab_2mm_"+str(ii)+".npy"
		filelist="list_lab_2mm_"+str(ii)+".npy"
		# filemap="map_lab_2mm_"+str(ii)+".npy"

		np.save(filex, x)
		np.save(filey, y)
		np.save(filelist,lista)
		# np.save(filemap,map)

		#x = None
		#y = None

	os.chdir(path)

	return path,listaT1,lista
