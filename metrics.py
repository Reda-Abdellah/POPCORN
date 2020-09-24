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

from keras import backend as K
import tensorflow as tf
import numpy as np
import keras
import statsmodels.api as sm
from scipy.signal import argrelextrema
from collections import OrderedDict, defaultdict
from skimage import measure
from scipy.stats import pearsonr
from keras.callbacks import ModelCheckpoint

class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self,filepath,validation_data=(), monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        #super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        super(CustomModelCheckpoint, self).__init__(filepath,monitor=monitor, verbose=verbose,
                     save_best_only=save_best_only, save_weights_only=save_weights_only,
                     mode=mode, period=period)
        self.X_val, self.y_val = validation_data
        self.y_val =np.argmax(self.y_val, axis=-1)
    def pred_val_score(self):
        y_pred = self.model.predict(self.X_val, batch_size=1,verbose=0)
        y_pred = np.argmax(y_pred, axis=-1)
        return score_batch( y_pred,self.y_val)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:

                current =self.pred_val_score()
                print('score of validation:'+str(current))
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)




def seg_metrics(seg_vol, truth_vol, output_errors=False):
    time_start = time.time()
    seg_total = np.sum(seg_vol)
    truth_total = np.sum(truth_vol)
    tp = np.sum(seg_vol[truth_vol == 1])
    dice = 2 * tp / (seg_total + truth_total)
    ppv = tp / (seg_total + 0.001)
    tpr = tp / (truth_total + 0.001)
    vd = abs(seg_total - truth_total) / truth_total

    # calculate LFPR
    seg_labels, seg_num = measure.label(seg_vol, return_num=True, connectivity=2)
    lfp_cnt = 0
    tmp_cnt = 0
    for label in range(1, seg_num + 1):
        tmp_cnt += np.sum(seg_vol[seg_labels == label])
        if np.sum(truth_vol[seg_labels == label]) == 0:
            lfp_cnt += 1
    lfpr = lfp_cnt / (seg_num + 0.001)

    # calculate LTPR
    truth_labels, truth_num = measure.label(truth_vol, return_num=True, connectivity=2)
    ltp_cnt = 0
    for label in range(1, truth_num + 1):
        if np.sum(seg_vol[truth_labels == label]) > 0:
            ltp_cnt += 1
    ltpr = ltp_cnt / truth_num

    # calculate Pearson's correlation coefficient
    corr = pearsonr(seg_vol.flatten(), truth_vol.flatten())[0]
    # print("Timed used calculating metrics: ", time.time() - time_start)

    return OrderedDict([('dice', dice), ('ppv', ppv), ('tpr', tpr), ('lfpr', lfpr),
                        ('ltpr', ltpr), ('vd', vd), ('corr', corr)])

def score_batch(seg_vol_batch, truth_vol_batch):
    mean=0
    #print(seg_vol_batch.shape)
    #print(truth_vol_batch.shape)
    for i in range(seg_vol_batch.shape[0]):
        mean=score(seg_vol_batch[i], truth_vol_batch[i])+mean
    return mean/seg_vol_batch.shape[0]


def score(seg_vol, truth_vol):

    seg_total = np.sum(seg_vol)
    truth_total = np.sum(truth_vol)
    tp = np.sum(seg_vol[truth_vol == 1])
    dice = 2 * tp / (seg_total + truth_total)
    ppv = tp / (seg_total + 0.001)
    #tpr = tp / (truth_total + 0.001)
    #vd = abs(seg_total - truth_total) / truth_total

    # calculate LFPR
    seg_labels, seg_num = measure.label(seg_vol, return_num=True, connectivity=2)
    lfp_cnt = 0
    tmp_cnt = 0
    for label in range(1, seg_num + 1):
        tmp_cnt += np.sum(seg_vol[seg_labels == label])
        if np.sum(truth_vol[seg_labels == label]) == 0:
            lfp_cnt += 1
    lfpr = lfp_cnt / (seg_num + 0.001)

    # calculate LTPR
    truth_labels, truth_num = measure.label(truth_vol, return_num=True, connectivity=2)
    ltp_cnt = 0
    for label in range(1, truth_num + 1):
        if np.sum(seg_vol[truth_labels == label]) > 0:
            ltp_cnt += 1
    ltpr = ltp_cnt / truth_num
    corr = pearsonr(seg_vol.flatten(), truth_vol.flatten())[0]
    return ((dice+ppv)/8)+((corr+ltpr-lfpr+1)/4)


def mdice(y_pred, y_true):
		acu=0
		size=y_true.get_shape().as_list()
		epsilon=0.00000000001
		for i in range(0,size[4]):
			a=y_true[:,:,:,:,i]
			b=y_pred[:,:,:,:,i]
			y_int = a[:]*b[:]
			acu=acu+(2*K.sum(y_int[:]) / (K.sum(a[:]) + K.sum(b[:]) + epsilon) )
		acu=acu/(size[4])
		return acu
def mdice_ssl_gt(y_pred, y_true):
	return mdice(y_pred[0::2,:,:,:,:], y_true[0::2,:,:,:,:])
def mdice_ssl_vb(y_pred, y_true):
	return mdice(y_pred[1::2,:,:,:,:], y_true[1::2,:,:,:,:])
def mdice_seg(y_pred, y_true):
	return mdice(y_pred[:,:,:,:,:2], y_true[:,:,:,:,:2])
def mdice_recon(y_pred, y_true):
	return mdice(y_pred[:,:,:,:,2:], y_true[:,:,:,:,2:])
def mse_recon(y_pred, y_true):
	return  keras.losses.mean_squared_error(y_pred[:,:,:,:,2:], y_true[:,:,:,:,2:])

def mdice_fc(y_pred, y_true):
	return mdice(y_pred[0::2,:,:,:,:], y_true[0::2,:,:,:,:])
def mse_fc(y_pred, y_true):
	return  keras.losses.mean_squared_error(y_pred[1::2,:,:,:,:], y_true[1::2,:,:,:,:])


def mean_dice_withoutbg(y_pred, y_true):
		acu=0
		n=0
		lista=np.unique(y_true)
		for i in lista:
			if(i==0):
				continue #avoid background
			a=(y_true==i)*1.0
			b=(y_pred==i)*1.0
			y_int = a[:]*b[:]
			acu=acu+(2*np.sum(y_int[:]) / (np.sum(a[:]) + np.sum(b[:])))
			n=n+1
		acu=acu/n
		return acu


def calculate_dice(pred_seg, gt_lbl):
    union_correct = pred_seg * gt_lbl
    tp_num = np.sum(union_correct)
    gt_pos_num = np.sum(gt_lbl)
    dice = (2.0 * tp_num) / (np.sum(pred_seg) + gt_pos_num) if gt_pos_num != 0 else -1
    return dice




def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if (y_actual[i]==y_hat[i]==1):
           TP += 1
        if( y_hat[i]==1 and y_actual[i]!=y_hat[i]):
           FP += 1
        if (y_actual[i]==y_hat[i]==0):
           TN += 1
        if( y_hat[i]==0 and y_actual[i]!=y_hat[i]):
           FN += 1

    return(TP, FP, TN, FN)


def mean_dice_withoutbg_mask(y_pred, y_true, mask_ind):
		acu=0
		n=0
		lista=np.unique(y_true)
		for i in lista:
			if(i==0):
				continue #avoid background
			a=(y_true[mask_ind]==i)*1.0
			b=(y_pred[mask_ind]==i)*1.0
			y_int = a[:]*b[:]
			acu=acu+(2*np.sum(y_int[:]) / (np.sum(a[:]) + np.sum(b[:])))
			n=n+1
		acu=acu/n
		return acu

def GDL(y_pred, y_true): #Generalized dice
	num=0
	den=0
	epsilon=0.00000000001
	size=y_true.get_shape().as_list()
	for i in range(0,size[4]):
		R=y_true[:,:,:,:,i]
		P=y_pred[:,:,:,:,i]
		W=1/K.sum(R[:])
		num = num + W*K.sum(R[:]*P[:])
		den = den + W*((K.sum(R[:]+P[:])))
	if(den==0):
		den=epsilon
	return 2*(num/den)

def GJLsmooth(y_pred, y_true): #Generalized Jaccard
	num=0 #https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
	den=0
	smooth =1.
	size=y_true.get_shape().as_list()
	for i in range(0,size[4]):
		R=y_true[:,:,:,:,i]
		P=y_pred[:,:,:,:,i]
		inter = K.sum(K.abs(R[:]*P[:]))
		W=1/K.sum(K.abs(R[:]))
		num = num + W*inter
		den = den + W*((K.sum(K.abs(R[:])+K.abs(P[:])))-inter)
	jac = (num+smooth) / (den+smooth)
	return jac

def focal_loss(y_true, y_pred):
    gamma=2.
    alpha=.25
    pt_1 = tf.where(K.equal(y_true, 1), y_pred, K.ones_like(y_pred))
    pt_0 = tf.where(K.equal(y_true, 0), y_pred, K.zeros_like(y_pred))
    return K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
