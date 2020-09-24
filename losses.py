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
import math
#import cc3d
from tensorflow import ones_like, equal, log
from tensorflow.python import mul
import keras

"""
def BottleneckRegularized(y_true, y_pred):
   return keras.losses.mean_squared_error(y_pred[0],y_pred[1])
"""
def BottleneckRegularized(y_true, y_pred):
    inter_dice=y_true[0]
    #latent_distance=keras.losses.mean_squared_error(y_pred[0],y_pred[1])
    latent_distance= 2*K.sum(K.square(y_pred[0]-y_pred[1]))/(K.mean(K.square(y_pred[0]))+K.mean(K.square(y_pred[1])))
    do=0.2
    h=1.0
    loss1= latent_distance*inter_dice - tf.math.minimum(0.0, (latent_distance-h)* (1-inter_dice))
    #loss2= tf.math.maximum(0.0, (latent_distance-h)* ( inter_dice-do)  )
    #latent_distance= 2*K.sum(K.square(y_pred[0]-y_pred[1]))/(K.mean(K.square(y_pred[0]))+K.mean(K.square(y_pred[1])))
    #segmentation_distance= 2*K.sum(K.square(y_true[0]-y_true[1]))/(K.sum(y_true[0])+K.sum(y_true[1]))
    #segmentation_distance=y_true[0]
    #loss3= latent_distance* K.exp(-segmentation_distance)
    return loss1

def newGDL(y_true, y_pred):
    acu=0
    W=[]
    tot=0
    eps=0
    size=y_pred.get_shape().as_list()
    for i in range(0,size[4]):
            a=y_true[:,:,:,:,i]
            b=1./tf.reduce_sum(a)
            W.append(b)
            tot=tot+b
    W=W/tot
    for i in range(0,size[4]):
            a=y_true[:,:,:,:,i]
            b=y_pred[:,:,:,:,i]
            y_int = a*b
            acu=acu+W[i]*(2*K.sum(y_int[:])+eps) / (K.sum(a[:]) + K.sum(b[:])+eps)
    return 1-acu

def newGJL(y_true, y_pred):

    acu=0
    W=[]
    tot=0
    eps=0
    size=y_pred.get_shape().as_list()
    for i in range(0,size[4]):

        a=y_true[:,:,:,:,i]
        b=1./tf.reduce_sum(a)
        W.append(b)
        tot=tot+b
    W=W/tot
    for i in range(0,size[4]):
        a=y_true[:,:,:,:,i]
        b=y_pred[:,:,:,:,i]
        y_int = a*b
        acu=acu+W[i]*(K.sum(y_int[:])+eps) / (K.sum(a[:]) + K.sum(b[:])+eps- y_int[:])
    return 1-acu


def JaccardLoss(y_true, y_pred):
    a=y_true
    b=y_pred
    y_int = a*b
    return 1-(K.sum(y_int[:])) / (K.sum(a[:]) + K.sum(b[:])+1- y_int[:])


def norm_mse(y_true,ypred):
    tot=K.sum(y_true)
    diff=K.sum(K.abs(y_true-ypred))
    return diff/tot

def newGtversky(alph=0.5,bet=None):
    if(bet==None):
        bet=1-alph

    def GtverskyLoss(y_true,y_pred):
        # 0.5 0.5 for dice
        # 1 1 for jaccard
        #Alpha = [0.2,0.8]
        #Beta= [0.8,0.2] #
        Alpha=[alph,bet]
        Beta=[bet,alph]
        W=[]
        tve=0.
        smooth =1.
        vol = 0.
        tot=0
        epsilon=1.
        size=y_pred.get_shape().as_list()
        for i in range(0,size[4]):

            a=y_true[:,:,:,:,i]
            b=1./tf.reduce_sum(a)
            W.append(b)
            tot=tot+b
        W=W/tot

        for i in range(0,size[4]):
            alpha= Alpha[i]
            beta= Beta[i]
            y_true_pos = K.flatten(y_true[:,:,:,:,i])
            y_pred_pos = K.flatten(y_pred[:,:,:,:,i])
            true_pos = K.sum(K.abs((y_true_pos * y_pred_pos)))
            false_neg = K.sum(K.abs(y_true_pos * (1-y_pred_pos)))
            false_pos = K.sum(K.abs((1-y_true_pos)*y_pred_pos))

            tve = tve+ (W[i]*true_pos +smooth) / (true_pos + alpha*false_neg + beta*false_pos  +smooth)
        return 1-tve
    return GtverskyLoss

def newGJL(y_true, y_pred):

    acu=0
    W=[]
    tot=0
    eps=0
    size=y_pred.get_shape().as_list()
    for i in range(0,size[4]):

        a=y_true[:,:,:,:,i]
        b=1./tf.reduce_sum(a)
        W.append(b)
        tot=tot+b
    W=W/tot
    for i in range(0,size[4]):
        a=y_true[:,:,:,:,i]
        b=y_pred[:,:,:,:,i]
        y_int = a*b
        acu=acu+W[i]*(K.sum(y_int[:])+eps) / (K.sum(a[:]) + K.sum(b[:])+eps- y_int[:])
    return 1-acu

def smooth_roud(x):
    x=x-0.5
    return 1 / (1 + tf.exp(-x*100))


def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


#"""
def weighted_binary_crossentropy(w1, w2):
    def loss(y_true, y_pred):

        # avoid absolute 0
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        ones = ones_like(y_true)
        msk = equal(y_true, ones)
        # tensor of booleans of length == y_true; true means that the true class is 1

        res, _ = tf.map_fn(lambda x: (mul(-log(x[0]), w1) if x[1] is True
                                      else mul(-log(1 - x[0]), w2), x[1]),
                           (y_pred, msk), dtype=(tf.float32, tf.bool))

        return res

    return loss


#"""

def mdice_loss(y_pred, y_true):
    acu=0
    size=y_true.get_shape().as_list()
    epsilon=0.00000000001
    for i in range(0,size[4]):
        a=y_true[:,:,:,:,i]
        b=y_pred[:,:,:,:,i]
        y_int = a[:]*b[:]
        acu=acu+(2*K.sum(y_int[:]) / (K.sum(a[:]) + K.sum(b[:])+ epsilon))
    acu=acu/(size[4])
    return 1-acu

def log_mdice_loss(y_pred, y_true):
    return K.log(mdice_loss(y_pred, y_true)+0.00000001)



def approx_round_grad(x, steepness=1):
    remainder = tf.mod(x, 1)
    sig = tf.sigmoid(steepness*(remainder - 0.5))
    return sig*(1 - sig)

"""
def LTPR(y_pred, y_true):
    y_pred=tf.reshape(y_pred[:,:,:,:,0],[108,120,108])
    y_true=tf.reshape(y_true[:,:,:,:,0],[108,120,108])
    labels_out = cc3d.connected_components(y_pred)
    count=0
    label_number=labels_out.max()+1
    for label in range(0,label_number):
        label_density=labels_out==label
            intersection=label_density*y_true
        if(intersection.max()>0):
            count=count+1

    return count/label_number
"""


def mix_loss(y_pred, y_true):
    r=0.99
    return r*GJLsmooth(y_pred, y_true)+(1-r)*fbeta(y_true, y_pred)



def fbeta(y_true, y_pred):
    #, threshold_shift=0
    beta = 1.5   #high Beta gives more importance to recall, if Beta=1 it becomes F1 score

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin =y_pred
    #y_pred_bin = smooth_roud(y_pred + threshold_shift)

    tp = K.sum(smooth_roud(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(smooth_roud(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(smooth_roud(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return 1- ( (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon()) )


def GDL(y_pred, y_true): #Generalized dice loss
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
    return 1-2*(num/den)

def GJLsmooth(y_pred, y_true): #Generalized Jaccard loss
    num=0. #https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    den=0.
    smooth =1.
    vol = 0.
    epsilon=1.
    size=y_true.get_shape().as_list()
    for i in range(0,size[4]):
        R=y_true[:,:,:,:,i]
        P=y_pred[:,:,:,:,i]
        inter = K.sum(K.abs(R[:]*P[:]))
        vol = K.sum(K.abs(R[:]))
        if(vol==0):
            W = 1
        else :
            W=1/(vol+epsilon)
        num = num + W*inter
        den = den + W*((K.sum(K.abs(R[:])+K.abs(P[:])))-inter)
    jac = (num+smooth) / (den+smooth)
    return 1-jac




def Gtversky(a=0.5,b=None):
    if(b==None):
        b=1-a

    def GtverskyLoss(y_pred, y_true):
        # 0.5 0.5 for dice
        # 1 1 for jaccard
        #Alpha = [0.2,0.8]
        #Beta= [0.8,0.2] #
        Alpha=[b,a]
        Beta=[a,b]
        num=0.
        den=0.
        smooth =1.
        vol = 0.
        epsilon=1.
        size=y_true.get_shape().as_list()
        for i in range(0,size[4]):
            alpha= Alpha[i]
            beta= Beta[i]
            y_true_pos = K.flatten(y_true[:,:,:,:,i])
            y_pred_pos = K.flatten(y_pred[:,:,:,:,i])
            true_pos = K.sum(K.abs((y_true_pos * y_pred_pos)))
            false_neg = K.sum(K.abs(y_true_pos * (1-y_pred_pos)))
            false_pos = K.sum(K.abs((1-y_true_pos)*y_pred_pos))

            vol = K.sum(K.abs(y_true[:,:,:,:,i]))
            W=1/(vol+epsilon)
            num = num + W*true_pos
            den = den + W*(true_pos + alpha*false_neg + beta*false_pos )
        tve = (num+smooth) / (den+smooth)
        return 1-tve
    return GtverskyLoss



def GJLsmooth_ssl(alpha=0.15):
    #y[0] contains GT, y[1] is volbrain segmentation
    def loss(y_pred, y_true):
        loss_gt=GJLsmooth(y_pred[0::2,:,:,:,:], y_true[0::2,:,:,:,:])
        loss_vb=GJLsmooth(y_pred[1::2,:,:,:,:], y_true[1::2,:,:,:,:])
        loss=alpha*loss_vb+(1-alpha)*loss_gt
        return loss
    return loss
def GJLsmooth_ict_coeff(coeff=2):
    #y[0] contains GT, y[1] is volbrain segmentation
    def loss(y_pred, y_true):
        loss_gt=GJLsmooth(y_pred[0::2,:,:,:,:], y_true[0::2,:,:,:,:])
        #loss_vb=GJLsmooth(y_pred[1::2,:,:,:,:], y_true[1::2,:,:,:,:])
        loss_vb= keras.losses.mean_squared_error(y_pred[1::2,:,:,:,:], y_true[1::2,:,:,:,:])
        loss=coeff*loss_vb+loss_gt
        return loss
    return loss
def GJLsmooth_seg_recon_coeff(coeff=2):
    #y[0] contains GT, y[1] is volbrain segmentation
    def loss(y_pred, y_true):
        #print(y_pred[:,:,:,:,0:2].shape)
        loss_gt=GJLsmooth(y_pred[:,:,:,:,0:2], y_true[:,:,:,:,0:2])
        #loss_vb=GJLsmooth(y_pred[1::2,:,:,:,:], y_true[1::2,:,:,:,:])
        loss_vb= 0.5*keras.losses.mean_squared_error(y_pred[:,:,:,:,2], y_true[:,:,:,:,2])+0.5*keras.losses.mean_squared_error(y_pred[:,:,:,:,3], y_true[:,:,:,:,3])
        loss=coeff*loss_vb+loss_gt
        return loss
    return loss

def GJLsmooth_segmentation(y_pred, y_true):
    return GJLsmooth(y_pred[0::2,:,:,:,:], y_true[0::2,:,:,:,:])

def feature_consistency(y_pred, y_true):
    #pred=K.reshape(y_pred[1::2,:,:,:,:],(125,-1))   #K.mean(y_pred[1::2,:,:,:,:],axis=4)
    #true=K.reshape(y_true[1::2,:,:,:,:],(125,-1))   #K.mean(y_true[1::2,:,:,:,:],axis=4)
    pred=K.reshape(y_pred,(125,-1))
    true=K.reshape(y_true,(125,-1))
    #true=y_true[1::2,:,:,:,:]
    #pred=y_pred[1::2,:,:,:,:]
    max_true=K.max(true,axis=0)
    max_pred=K.max(pred,axis=0)
    #mean_true=K.mean(true,axis=0)
    #mean_pred=K.mean(pred,axis=0)
    #return keras.losses.mean_squared_error(K.concatenate((max_true,mean_true)),K.concatenate((max_pred,mean_pred)))/(K.sum(max_pred)+K.sum(mean_pred)+K.sum(mean_true)+K.sum(max_true))
    return keras.losses.mean_squared_error( max_true, max_pred )/(K.sum(max_pred)+K.sum(max_true))

def mdice2(y_pred, y_true):
    acu=0
    epsilon=0.01
    size=y_true.get_shape().as_list()
    for i in range(0,size[4]):
        a = y_true[:,:,:,:,i]
        T = np.where(y_true == i)
        P = np.where(y_pred == i)
        b = y_pred[:,:,:,:,i]
        y_int = a[:]*b[:]
        acu=acu + (len(T)/len(P)+1) * (2*K.sum(y_int[:]) / (K.sum(a[:])+K.sum(b[:])+epsilon))
    acu=acu/(size[4])
    return 1-acu


def T1_FLAIR_mixed(y_pred, y_true):
    #print(y_true.get_shape().as_list())
    #vox_les= K.sum( y_true[:,:,:,:,2])
    #vox_bg=  y_true[:,:,:,:,2].get_shape().as_list()
    #vox_bg= vox_bg[0]*vox_bg[1]*vox_bg[2]*vox_bg[3] - vox_les
    #vox_bg= 80*80*96 - vox_les
    #loss= (vox_les+vox_bg)*((keras.losses.mean_squared_error(y_pred[:,:,:,:,0]* y_true[:,:,:,:,2] , y_true[:,:,:,:,0]) / vox_les )+ (keras.losses.mean_squared_error(y_pred[:,:,:,:,0]* (1-y_true[:,:,:,:,2]) , y_true[:,:,:,:,1]) / vox_bg ))
    loss_1=keras.losses.mean_absolute_error(y_pred[:,:,:,:,0:2],y_true[:,:,:,:,0:2])
    loss_2= GJLsmooth(y_pred[:,:,:,:,2:4], y_true[:,:,:,:,2:4])
    loss=loss_1+loss_2
    return loss

def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def confusion(y_true, y_pred):
    smooth=1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    prec = (tp + smooth)/(tp+fp+smooth)
    recall = (tp+smooth)/(tp+fn+smooth)
    return prec, recall
epsilon = 1e-5
smooth=1
def tp(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    tp = (K.sum(y_pos * y_pred_pos) + smooth)/ (K.sum(y_pos) + smooth)
    return tp

def tn(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth )
    return tn

def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.5 #0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)



def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)
def bce(y_true, y_pred):
    loss=keras.losses.binary_crossentropy(y_true[...,0], y_pred[...,0])+keras.losses.binary_crossentropy(y_true[...,1], y_pred[...,1])
    loss=loss/2
    return loss
def focal_tversky(y_true,y_pred):
    #pt_1 = tversky(y_true[...,0], y_pred[...,0]) + tversky(y_true[...,1], y_pred[...,1])
    #pt_1 =pt_1 /2
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed


def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """


        #y_true=y_true[...,1]
        #y_pred=y_pred[...,1]

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed

def focal(y_true, y_pred):
    gamma=2.
    alph
