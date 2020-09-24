from keras.layers import Input, Conv3D , Dropout, Concatenate, BatchNormalization,Conv3DTranspose,GlobalAveragePooling3D, Add,MaxPooling3D, UpSampling3D,Flatten,Dense,Activation,SpatialDropout3D,Reshape, Lambda,GlobalMaxPooling3D
from keras.models import Model , load_model
from keras.activations import softmax
from keras import backend as K
from keras import optimizers
from keras.regularizers import l2
from keras.layers.merge import concatenate
from LinearResizeLayer import LinearResizeLayer
import tensorflow as tf
from NonlocalBlock import non_local_block
#import segmentation_models as sm
#import classification_models
from GroupNormalization import GroupNormalization
#from crf_as_rnn_keras_layer import CRF_RNN_Layer
#from crfrnn3d.src import high_dim_filter_loader
#from crfrnn3d.src.crfrnn_layer import CrfRnnLayer3D
#import CRFasRNNLayer.lattice_filter_op_loader
#from CRFasRNNLayer.crf_rnn_layer import crf_rnn_layer
#from CRFasRNNLayer.crf_as_rnn_keras_layer import CRF_RNN_Layer
#from CRFasRNNLayer_Conv1.crf_as_rnn_keras_layer import CRF_RNN_Layer as up_crf

from keras.legacy import interfaces
import keras.backend as K
from keras.optimizers import Optimizer
#import sys
#sys.path.insert(1,'./crfasrnn_keras/src')
#from crfrnn_layer import CrfRnnLayer3D


def load_UNET3D_bottleneck_regularized(ps1,ps2,ps3,ch,nc=4,nf=24,drop=0.5,groups=8,final_act='softmax'):   #3 levels + linear upsampling
    # See https://arxiv.org/pdf/1806.00546.pdf
    #nc: number of output classes
    #nf: number of filter
    #ch: number of channels
    G=groups
    # model UNET 3D
    pool_size=[2,2,2]

    input_img = Input(shape=(ps1, ps2, ps3, ch))     # adapt this if using `channels_first` image data format

    conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(input_img)
    conv1 = GroupNormalization(group=G)(conv1)
    #conv1 = WeightNorm(conv1)
    conv1 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv1)
    #conv1 = BatchNormalization()(conv1)
    #conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size)(conv1)
    if(drop>0):
        pool1 = Dropout(drop)(pool1)

    conv2 = GroupNormalization(group=G)(pool1)
    #conv2 = WeightNorm(pool1)
    conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    conv2 = GroupNormalization(group=G)(conv2)
    #conv2 = WeightNorm(conv2)

    conv2 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv2)
    #conv2 = BatchNormalization()(conv2)
    #conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size)(conv2)
    if(drop>0):
        pool2 = Dropout(drop)(pool2)

    conv3 = GroupNormalization(group=G)(pool2)
    #conv3 = WeightNorm(pool2)

    conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    conv3 = GroupNormalization(group=G)(conv3)
    #conv3 = WeightNorm(conv3)

    conv3 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv3)
    #conv3 = BatchNormalization()(conv3)
    #conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=pool_size)(conv3)
    if(drop>0):
            pool3 = Dropout(drop)(pool3)

    conv4 = GroupNormalization(group=G)(pool3)
    #conv4 = WeightNorm(pool3)

    conv4 = Conv3D(nf*16, (3, 3, 3), activation='relu', name='bottleneck' ,padding='same')(conv4) #,
    #conv4 = BatchNormalization()(conv4)
    #conv4 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv4)
    #conv4 = BatchNormalization()(conv4)
    #conv4 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv4)


    #up5 = UpSampling3D()(conv4)
    new_shape = conv3.shape.as_list()[1:-1]
    up5  = LinearResizeLayer(new_shape,name='up5')(conv4)

    up5 = concatenate([up5, conv3], axis=4) # up5 = 512 + conv3 = 256
    up5 = GroupNormalization(group=G)(up5)
    #up5 = WeightNorm(up5)

    conv5 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(up5)
    #conv5 = BatchNormalization()(conv5)
    #conv5 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv5)
    #conv5 = BatchNormalization()(conv5)
    #conv5 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv5)


    #up6 = UpSampling3D()(conv5)
    new_shape = conv2.shape.as_list()[1:-1]
    up6  = LinearResizeLayer(new_shape,name='up6')(conv5)

    up6 = concatenate([up6, conv2], axis=4) # up6 = 256 + conv2 = 128
    up6 = GroupNormalization(group=G)(up6)
    #up6 = WeightNorm(up6)

    conv6 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up6)
    #conv6 = BatchNormalization()(conv6)
    #conv6 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv6)
    #conv6 = BatchNormalization()(conv6)
    #conv6 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv6)


    #up7 = UpSampling3D()(conv6)
    new_shape = conv1.shape.as_list()[1:-1]
    up7   = LinearResizeLayer(new_shape,name='up7')(conv6)

    up7 = concatenate([up7, conv1], axis=4) # up7 = 128 + conv1 = 64
    up7 = GroupNormalization(group=G)(up7)
    #up7 = WeightNorm(up7)

    conv7 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up7)
    #conv7 = BatchNormalization()(conv7)
    #conv7 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv7)
    #conv7 = BatchNormalization()(conv7)
    #conv7 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv7)


    output = Conv3D(nc, (3, 3, 3), activation=final_act, padding='same')(conv7)
    bottleneck_reduced=GlobalAveragePooling3D()(conv4)
    model = Model(input_img, [output,bottleneck_reduced])
    #model = Model(input_img, [output,conv4])

    return model

class Adam_lr_mult(Optimizer):
    """Adam optimizer.
    Adam optimizer, with learning rate multipliers built on Keras implementation
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)

    AUTHOR: Erik Brorson
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False,
                 multipliers=None, debug_verbose=False,**kwargs):
        super(Adam_lr_mult, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.multipliers = multipliers
        self.debug_verbose = debug_verbose

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):

            # Learning rate multipliers
            if self.multipliers:
                multiplier = [mult for mult in self.multipliers if mult in p.name]
            else:
                multiplier = None
            if multiplier:
                new_lr_t = lr_t * self.multipliers[multiplier[0]]
                if self.debug_verbose:
                    print('Setting {} to learning rate {}'.format(multiplier[0], new_lr_t))
                    print(K.get_value(new_lr_t))
            else:
                new_lr_t = lr_t
                if self.debug_verbose:
                    print('No change in learning rate {}'.format(p.name))
                    print(K.get_value(new_lr_t))
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - new_lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - new_lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad,
                  'multipliers':self.multipliers}
        base_config = super(Adam_lr_mult, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))






def filter_size(init,mult,exp):
    i=int(init*(mult)**exp)
    if((i%8 )== 0):

        return i
    else:
        if((i%8)>3):

            i= 8*(1+ i//8)
        else:

            i= 8*(i//8)

        return int(i)


    #conv2 = GroupNormalization(group=G)(pool1)


def Dense3D_Unet(ps1,ps2,ps3,ch,nc=2,G=8,drop=0.5,mult=1.2):
    init=32

    inputs = Input((ps1, ps2, ps3, ch ))
    conv11 = Conv3D(filter_size(init,mult,0), (3, 3, 3), activation='relu', padding='same')(inputs)
    conc11 = concatenate([inputs, conv11], axis=4)
    #conc11 = GroupNormalization(group=G)(conc11)
    conv12 = Conv3D(filter_size(init,mult,1), (3, 3, 3), activation='relu', padding='same')(conc11)
    conc12 = concatenate([inputs, conv12], axis=4)
    #conc12 = GroupNormalization(group=G)(conc12)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conc12)
    if(drop>0):
        pool1 = Dropout(drop)(pool1)

    conv21 = Conv3D(filter_size(init,mult,2), (3, 3, 3), activation='relu', padding='same')(pool1)
    conc21 = concatenate([pool1, conv21], axis=4)
    #conc21 = GroupNormalization(group=G)(conc21)
    conv22 = Conv3D(filter_size(init,mult,3), (3, 3, 3), activation='relu', padding='same')(conc21)
    conc22 = concatenate([pool1, conv22], axis=4)
    #conc22 = GroupNormalization(group=G)(conc22)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conc22)
    if(drop>0):
        pool2 = Dropout(drop)(pool2)

    conv31 = Conv3D(filter_size(init,mult,4), (3, 3, 3), activation='relu', padding='same')(pool2)
    conc31 = concatenate([pool2, conv31], axis=4)
    #conc31 = GroupNormalization(group=G)(conc31)
    conv32 = Conv3D(filter_size(init,mult,5), (3, 3, 3), activation='relu', padding='same')(conc31)
    conc32 = concatenate([pool2, conv32], axis=4)
    #conc32 = GroupNormalization(group=G)(conc32)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conc32)
    if(drop>0):
        pool3 = Dropout(drop)(pool3)

    conv41 = Conv3D(filter_size(init,mult,6), (3, 3, 3), activation='relu', padding='same')(pool3)
    conc41 = concatenate([pool3, conv41], axis=4)
    #conc41 = GroupNormalization(group=G)(conc41)
    conv42 = Conv3D(filter_size(init,mult,6), (3, 3, 3), activation='relu', padding='same')(conc41)
    conc42 = concatenate([pool3, conv42], axis=4)
    #conc42 = GroupNormalization(group=G)(conc42)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conc42)
    if(drop>0):
        pool4 = Dropout(drop)(pool4)

    conv51 = Conv3D(filter_size(init,mult,7), (3, 3, 3), activation='relu', padding='same')(pool4)
    conc51 = concatenate([pool4, conv51], axis=4)
    #conc51 = GroupNormalization(group=G)(conc51)
    conv52 = Conv3D(filter_size(init,mult,7), (3, 3, 3), activation='relu', padding='same')(conc51)
    conc52 = concatenate([pool4, conv52], axis=4)
    #conc52 = GroupNormalization(group=G)(conc52)

    #up6 = concatenate([Conv3DTranspose(filter_size(init,mult,6), (2, 2, 2), strides=(2, 2, 2), padding='same')(conc52), conc42], axis=4)
    new_shape = conc42.shape.as_list()[1:-1]
    up6  = LinearResizeLayer(new_shape,name='up6')(conv52)
    up6 = concatenate([up6, conc42], axis=4)

    conv61 = Conv3D(filter_size(init,mult,6), (3, 3, 3), activation='relu', padding='same')(up6)
    conc61 = concatenate([up6, conv61], axis=4)
    #conc61 = GroupNormalization(group=G)(conc61)

    conv62 = Conv3D(filter_size(init,mult,6), (3, 3, 3), activation='relu', padding='same')(conc61)
    conc62 = concatenate([up6, conv62], axis=4)
    #conc62 = GroupNormalization(group=G)(conc62)

    #up7 = concatenate([Conv3DTranspose(filter_size(init,mult,5), (2, 2, 2), strides=(2, 2, 2), padding='same')(conc62), conv32], axis=4)
    new_shape = conc32.shape.as_list()[1:-1]
    up7  = LinearResizeLayer(new_shape,name='up7')(conv62)
    up7 = concatenate([up7, conc32], axis=4)

    conv71 = Conv3D(filter_size(init,mult,5), (3, 3, 3), activation='relu', padding='same')(up7)
    conc71 = concatenate([up7, conv71], axis=4)
    #conc71 = GroupNormalization(group=G)(conc71)
    conv72 = Conv3D(filter_size(init,mult,5), (3, 3, 3), activation='relu', padding='same')(conc71)
    conc72 = concatenate([up7, conv72], axis=4)
    #conc72 = GroupNormalization(group=G)(conc72)

    #up8 = concatenate([Conv3DTranspose(filter_size(init,mult,3), (2, 2, 2), strides=(2, 2, 2), padding='same')(conc72), conv22], axis=4)
    new_shape = conc22.shape.as_list()[1:-1]
    up8  = LinearResizeLayer(new_shape,name='up8')(conv72)
    up8 = concatenate([up8, conc22], axis=4)
    conv81 = Conv3D(filter_size(init,mult,3), (3, 3, 3), activation='relu', padding='same')(up8)
    conc81 = concatenate([up8, conv81], axis=4)
    #conc81 = GroupNormalization(group=G)(conc81)
    conv82 = Conv3D(filter_size(init,mult,3), (3, 3, 3), activation='relu', padding='same')(conc81)
    conc82 = concatenate([up8, conv82], axis=4)
    #conc82 = GroupNormalization(group=G)(conc82)

    #up9 = concatenate([Conv3DTranspose(filter_size(init,mult,2), (2, 2, 2), strides=(2, 2, 2), padding='same')(conc82), conv12], axis=4)
    new_shape = conc12.shape.as_list()[1:-1]
    up9  = LinearResizeLayer(new_shape,name='up9')(conv82)
    up9 = concatenate([up9, conc12], axis=4)
    conv91 = Conv3D(filter_size(init,mult,2), (3, 3, 3), activation='relu', padding='same')(up9)
    conc91 = concatenate([up9, conv91], axis=4)
    #conc91 = GroupNormalization(group=G)(conc91)
    conv92 = Conv3D(filter_size(init,mult,2), (3, 3, 3), activation='relu', padding='same')(conc91)
    conc92 = concatenate([up9, conv92], axis=4)
    #conc92 = GroupNormalization(group=G)(conc92)

    conv10 = Conv3D(nc, (1, 1, 1), activation='softmax')(conc92)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.summary()

    return model












def unext(ps1,ps2,ps3,ch,nc=4,nf=24,drop=0.5):
    model = sm.Unet('resnext503D', input_shape=(ps1,ps2,ps3,2),encoder_weights='None',weights=None,classes=2,decoder_use_batchnorm='None')
    return model


def block_25D(nf,input_img,name=None):
    conv_0 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(input_img)
    conv_1 = Conv3D(nf, (3, 3, 1), activation='relu', padding='same')(input_img)
    conv_2 = Conv3D(nf, (3, 1, 3), activation='relu', padding='same')(input_img)
    conv_3 = Conv3D(nf, (1, 3, 3), activation='relu', padding='same')(input_img)
    if(not name==None):
        conv=concatenate([conv_0,conv_1,conv_2,conv_3],name=name, axis=4)
    else:
        conv=concatenate([conv_0,conv_1,conv_2,conv_3], axis=4)
    return conv

def unet25D(ps1,ps2,ps3,ch,nc=4,nf=24//4 ,drop=0.5):   #3 levels + linear upsampling
    # See https://arxiv.org/pdf/1806.00546.pdf
    #nc: number of output classes
    #nf: number of filter
    #ch: number of channels
    G=8
    # model UNET 3D
    pool_size=[2,2,2]

    input_img = Input(shape=(ps1, ps2, ps3, ch))     # adapt this if using `channels_first` image data format

    conv1 = block_25D(nf,input_img)
    conv1 = GroupNormalization(group=G)(conv1)
    conv1 = block_25D(nf*2,conv1)

    pool1 = MaxPooling3D(pool_size=pool_size)(conv1)
    if(drop>0):
        pool1 = Dropout(drop)(pool1)

    conv2 = GroupNormalization(group=G)(pool1)
    conv2 = block_25D(nf*2,conv2)
    conv2 = GroupNormalization(group=G)(conv2)
    conv2 = block_25D(nf*4,conv2)


    pool2 = MaxPooling3D(pool_size=pool_size)(conv2)
    if(drop>0):
        pool2 = Dropout(drop)(pool2)

    conv3 = GroupNormalization(group=G)(pool2)
    conv3 = block_25D(nf*4,conv3)
    conv3 = GroupNormalization(group=G)(conv3)
    conv3 = block_25D(nf*8,conv3)

    pool3 = MaxPooling3D(pool_size=pool_size)(conv3)
    if(drop>0):
            pool3 = Dropout(drop)(pool3)

    conv4 = GroupNormalization(group=G)(pool3)
    conv4 = block_25D(nf*16,conv4,name='bottleneck') #,name='bottleneck'


    new_shape = conv3.shape.as_list()[1:-1]
    up5  = LinearResizeLayer(new_shape,name='up5')(conv4)

    up5 = concatenate([up5, conv3], axis=4) # up5 = 512 + conv3 = 256
    up5 = GroupNormalization(group=G)(up5)
    conv5 = block_25D(nf*8,up5)

    new_shape = conv2.shape.as_list()[1:-1]
    up6  = LinearResizeLayer(new_shape,name='up6')(conv5)

    up6 = concatenate([up6, conv2], axis=4) # up6 = 256 + conv2 = 128
    up6 = GroupNormalization(group=G)(up6)
    conv6 = block_25D(nf*4,up6)

    #up7 = UpSampling3D()(conv6)
    new_shape = conv1.shape.as_list()[1:-1]
    up7   = LinearResizeLayer(new_shape,name='up7')(conv6)

    up7 = concatenate([up7, conv1], axis=4) # up7 = 128 + conv1 = 64
    up7 = GroupNormalization(group=G)(up7)
    conv7 = block_25D(nf*4,up7)

    output = Conv3D(nc, (3, 3, 3), activation='softmax', padding='same')(conv7)

    model = Model(input_img, output)

    return model



def double_conv_layer(x, size, dropout=0.0, batch_norm=True):
    axis = 4
    conv = Conv3D(size, (3,3,3), padding='same')(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    conv = Conv3D(size, (3,3,3), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = SpatialDropout3D(dropout)(conv)
    return conv


def redgen_224(ps1,ps2,ps3,dropout_val=0.2, channels=2,filters=32):
    inputs = Input((ps1, ps2, ps3, channels))
    axis = 4
    filters = filters

    conv_224 = double_conv_layer(inputs, filters)
    pool_112 = MaxPooling3D(pool_size=(2,2, 2))(conv_224)

    conv_112 = double_conv_layer(pool_112, 2*filters)
    pool_56 = MaxPooling3D(pool_size=(2,2, 2))(conv_112)

    conv_56 = double_conv_layer(pool_56, 4*filters)
    pool_28 = MaxPooling3D(pool_size=(2,2, 2))(conv_56)

    conv_28 = double_conv_layer(pool_28, 8*filters)
    pool_14 = MaxPooling3D(pool_size=(2,2, 2))(conv_28)

    conv_14 = double_conv_layer(pool_14, 16*filters)
    pool_7 = MaxPooling3D(pool_size=(2,2, 2))(conv_14)

    conv_7 = double_conv_layer(pool_7, 32*filters)

    up_14 = concatenate([UpSampling3D(size=(2,2, 2))(conv_7), conv_14], axis=axis)
    up_conv_14 = double_conv_layer(up_14, 16*filters)

    up_28 = concatenate([UpSampling3D(size=(2,2, 2))(up_conv_14), conv_28], axis=axis)
    up_conv_28 = double_conv_layer(up_28, 8*filters)

    up_56 = concatenate([UpSampling3D(size=(2, 2,2))(up_conv_28), conv_56], axis=axis)
    up_conv_56 = double_conv_layer(up_56, 4*filters)

    up_112 = concatenate([UpSampling3D(size=(2, 2,2))(up_conv_56), conv_112], axis=axis)
    up_conv_112 = double_conv_layer(up_112, 2*filters)

    up_224 = concatenate([UpSampling3D(size=(2,2, 2))(up_conv_112), conv_224], axis=axis)
    up_conv_224 = double_conv_layer(up_224, filters, dropout_val)

    conv_final = Conv3D(2, (1,1, 1))(up_conv_224)
    #conv_final = Activation('sigmoid')(conv_final)
    model = Model(inputs, conv_final, name="redgen")
    return model



def load_UNET3D_SLANT27(ps1,ps2,ps3,ch,nc=4,nf=24,drop=0.5):   #3 levels
    # See https://arxiv.org/pdf/1806.00546.pdf
    #nc: number of output classes
    #nf: number of filter
    #ch: number of channels

    # model UNET 3D
    pool_size=[2,2,2]

    input_img = Input(shape=(ps1, ps2, ps3, ch))     # adapt this if using `channels_first` image data format

    conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(input_img)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv1)
    #conv1 = BatchNormalization()(conv1)
    #conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size)(conv1)
    if(drop>0):
        pool1 = Dropout(drop)(pool1)

    conv2 = BatchNormalization()(pool1)
    conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv2)
    #conv2 = BatchNormalization()(conv2)
    #conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size)(conv2)
    if(drop>0):
        pool2 = Dropout(drop)(pool2)

    conv3 = BatchNormalization()(pool2)
    conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv3)
    #conv3 = BatchNormalization()(conv3)
    #conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=pool_size)(conv3)
    if(drop>0):
            pool3 = Dropout(drop)(pool3)

    conv4 = BatchNormalization()(pool3)
    conv4 = Conv3D(nf*16, (3, 3, 3), activation='relu',name='bottleneck', padding='same')(conv4)
    #conv4 = BatchNormalization()(conv4)
    #conv4 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv4)
    #conv4 = BatchNormalization()(conv4)
    #conv4 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv4)


    up5 = UpSampling3D()(conv4)
    up5 = concatenate([up5, conv3], axis=4) # up5 = 512 + conv3 = 256
    up5 = BatchNormalization()(up5)
    conv5 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(up5)
    #conv5 = BatchNormalization()(conv5)
    #conv5 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv5)
    #conv5 = BatchNormalization()(conv5)
    #conv5 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv5)


    up6 = UpSampling3D()(conv5)
    up6 = concatenate([up6, conv2], axis=4) # up6 = 256 + conv2 = 128
    up6 = BatchNormalization()(up6)
    conv6 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up6)
    #conv6 = BatchNormalization()(conv6)
    #conv6 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv6)
    #conv6 = BatchNormalization()(conv6)
    #conv6 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv6)


    up7 = UpSampling3D()(conv6)
    up7 = concatenate([up7, conv1], axis=4) # up7 = 128 + conv1 = 64
    up7 = BatchNormalization()(up7)
    conv7 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up7)
    #conv7 = BatchNormalization()(conv7)
    #conv7 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv7)
    #conv7 = BatchNormalization()(conv7)
    #conv7 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv7)

    output = Conv3D(nc, (3, 3, 3), activation='softmax', padding='same')(conv7)

    model = Model(input_img, output)

    return model


def dilated(ps1,ps2,ps3,ch,nc=4,nf=24,drop=0.5):   #3 levels + linear upsampling
    # See https://arxiv.org/pdf/1806.00546.pdf
    #nc: number of output classes
    #nf: number of filter
    #ch: number of channels
    G=8
    # model UNET 3D
    pool_size=[2,2,2]

    input_img = Input(shape=(ps1, ps2, ps3, ch))     # adapt this if using `channels_first` image data format

    conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(input_img)
    conv1 = GroupNormalization(group=G)(conv1)
    conv1 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv1)
    #pool1 = MaxPooling3D(pool_size=pool_size)(conv1)
    if(drop>0):
        pool1 = Dropout(drop)(conv1)

    conv2 = GroupNormalization(group=G)(pool1)
    conv2 = Conv3D(nf*2, (3, 3, 3), dilation_rate=(2,2,2), activation='relu', padding='same')(conv2)
    conv2 = GroupNormalization(group=G)(conv2)

    conv2 = Conv3D(nf*4, (3, 3, 3), dilation_rate=(2,2,2) , activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size)(conv2)
    if(drop>0):
        pool2 = Dropout(drop)(pool2)

    conv3 = GroupNormalization(group=G)(pool2)

    conv3 = Conv3D(nf*4, (3, 3, 3), dilation_rate=(2,2,2), activation='relu', padding='same')(conv3)
    conv3 = GroupNormalization(group=G)(conv3)

    conv3 = Conv3D(nf*6, (3, 3, 3), dilation_rate=(2,2,2), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=pool_size)(conv3)
    if(drop>0):
            pool3 = Dropout(drop)(pool3)

    conv4 = GroupNormalization(group=G)(pool3)

    conv4 = Conv3D(nf*8, (3, 3, 3), dilation_rate=(2,2,2), activation='relu', name='bottleneck' ,padding='same')(conv4) #,
    new_shape = conv3.shape.as_list()[1:-1]
    up5  = LinearResizeLayer(new_shape,name='up5')(conv4)

    up5 = concatenate([up5, conv3], axis=4) # up5 = 512 + conv3 = 256
    up5 = GroupNormalization(group=G)(up5)

    conv5 = Conv3D(nf*6, (3, 3, 3), dilation_rate=(2,2,2), activation='relu', padding='same')(up5)
    new_shape = conv2.shape.as_list()[1:-1]
    up6  = LinearResizeLayer(new_shape,name='up6')(conv5)

    up6 = concatenate([up6, conv2], axis=4) # up6 = 256 + conv2 = 128
    up6 = GroupNormalization(group=G)(up6)

    conv6 = Conv3D(nf*4, (3, 3, 3), dilation_rate=(2,2,2), activation='relu', padding='same')(up6)
    #new_shape = conv1.shape.as_list()[1:-1]
    #up7   = LinearResizeLayer(new_shape,name='up7')(conv6)

    up7 = concatenate([conv6, conv1], axis=4) # up7 = 128 + conv1 = 64
    up7 = GroupNormalization(group=G)(up7)

    conv7 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(up7)
    output = Conv3D(nc, (3, 3, 3), activation='softmax', padding='same')(conv7)

    model = Model(input_img, output)

    return model

    #conv6 = Conv3D(nf*4, (3, 3, 3), dilation_rate=(2,2,2), activation='relu', padding='same')(up6)



def YNET(ps1,ps2,ps3,ch,nc=4,nf=24,drop=0.5,groups=8):   #3 levels + linear upsampling
    # See https://arxiv.org/pdf/1806.00546.pdf
    #nc: number of output classes
    #nf: number of filter
    #ch: number of channels
    G=groups
    # model UNET 3D
    pool_size=[2,2,2]

    input_img = Input(shape=(ps1, ps2, ps3, ch))     # adapt this if using `channels_first` image data format
    in_1= Lambda(lambda x: x[:,:,:,:,0:1])(input_img) #, output_shape=(ps1, ps2, ps3, 1)
    in_2= Lambda(lambda x: x[:,:,:,:,1:2])(input_img) #, output_shape=(ps1, ps2, ps3, 1)
    in_3= Lambda(lambda x: x[:,:,:,:,2:3])(input_img)
    in_4= Lambda(lambda x: x[:,:,:,:,3:4])(input_img)

    #print(input_img)
    #print(in_1)
    #print(in_2)

    conv1 = Conv3D(nf//2, (3, 3, 3), activation='relu', padding='same')(in_1)
    conv1 = GroupNormalization(group=G//2)(conv1)
    #conv1 = WeightNorm(conv1)
    conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv1)
    #conv1 = BatchNormalization()(conv1)
    #conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size)(conv1)
    if(drop>0):
        pool1 = Dropout(drop)(pool1)

    conv2 = GroupNormalization(group=G)(pool1)
    #conv2 = WeightNorm(pool1)
    conv2 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv2)
    conv2 = GroupNormalization(group=G)(conv2)
    #conv2 = WeightNorm(conv2)

    conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    #conv2 = BatchNormalization()(conv2)
    #conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size)(conv2)
    if(drop>0):
        pool2 = Dropout(drop)(pool2)

    conv3 = GroupNormalization(group=G)(pool2)
    #conv3 = WeightNorm(pool2)

    conv3 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv3)
    conv3 = GroupNormalization(group=G)(conv3)
    #conv3 = WeightNorm(conv3)

    conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    #conv3 = BatchNormalization()(conv3)
    #conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=pool_size)(conv3)
    if(drop>0):
            pool3 = Dropout(drop)(pool3)

    conv4 = GroupNormalization(group=G)(pool3)
    #conv4 = WeightNorm(pool3)

    conv4 = Conv3D(nf*8, (3, 3, 3), activation='relu',padding='same')(conv4) #,



    conv1_ = Conv3D(nf//2, (3, 3, 3), activation='relu', padding='same')(in_2)
    conv1_ = GroupNormalization(group=G//2)(conv1_)
    #conv1 = WeightNorm(conv1)
    conv1_ = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv1_)
    #conv1 = BatchNormalization()(conv1)
    #conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1_ = MaxPooling3D(pool_size=pool_size)(conv1_)
    if(drop>0):
        pool1_ = Dropout(drop)(pool1_)

    conv2_ = GroupNormalization(group=G)(pool1_)
    #conv2 = WeightNorm(pool1)
    conv2_ = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv2_)
    conv2_ = GroupNormalization(group=G)(conv2_)
    #conv2 = WeightNorm(conv2)

    conv2_ = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2_)
    #conv2 = BatchNormalization()(conv2)
    #conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2_ = MaxPooling3D(pool_size=pool_size)(conv2_)
    if(drop>0):
        pool2_ = Dropout(drop)(pool2_)

    conv3_ = GroupNormalization(group=G)(pool2_)
    #conv3 = WeightNorm(pool2)

    conv3_ = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv3_)
    conv3_ = GroupNormalization(group=G)(conv3_)
    #conv3 = WeightNorm(conv3)

    conv3_ = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3_)
    #conv3 = BatchNormalization()(conv3)
    #conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3_ = MaxPooling3D(pool_size=pool_size)(conv3_)
    if(drop>0):
            pool3_ = Dropout(drop)(pool3_)

    conv4_ = GroupNormalization(group=G)(pool3_)
    #conv4 = WeightNorm(pool3)

    conv4_ = Conv3D(nf*8, (3, 3, 3), activation='relu' ,padding='same')(conv4_)
    #conv4 = BatchNormalization()(conv4)
    #conv4 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv4)
    #conv4 = BatchNormalization()(conv4)
    #conv4 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv4)




    conv1__ = Conv3D(nf//2, (3, 3, 3), activation='relu', padding='same')(in_3)
    conv1__ = GroupNormalization(group=G//2)(conv1__)
    #conv1 = WeightNorm(conv1)
    conv1__ = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv1__)
    #conv1 = BatchNormalization()(conv1)
    #conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1__ = MaxPooling3D(pool_size=pool_size)(conv1__)
    if(drop>0):
        pool1__ = Dropout(drop)(pool1__)

    conv2__ = GroupNormalization(group=G)(pool1__)
    #conv2 = WeightNorm(pool1)
    conv2__ = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv2__)
    conv2__ = GroupNormalization(group=G)(conv2__)
    #conv2 = WeightNorm(conv2)

    conv2__ = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2__)
    #conv2 = BatchNormalization()(conv2)
    #conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2__ = MaxPooling3D(pool_size=pool_size)(conv2__)
    if(drop>0):
        pool2__ = Dropout(drop)(pool2__)

    conv3__ = GroupNormalization(group=G)(pool2__)
    #conv3 = WeightNorm(pool2)

    conv3__ = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv3__)
    conv3__ = GroupNormalization(group=G)(conv3__)
    #conv3 = WeightNorm(conv3)

    conv3__ = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3__)
    #conv3 = BatchNormalization()(conv3)
    #conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3__ = MaxPooling3D(pool_size=pool_size)(conv3__)
    if(drop>0):
            pool3__ = Dropout(drop)(pool3__)

    conv4__ = GroupNormalization(group=G)(pool3__)
    #conv4 = WeightNorm(pool3)

    conv4__ = Conv3D(nf*8, (3, 3, 3), activation='relu' ,padding='same')(conv4__)
    #conv4 = BatchNormalization()(conv4)
    #conv4 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv4)
    #conv4 = BatchNormalization()(conv4)
    #conv4 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv4)









    conv1___ = Conv3D(nf//2, (3, 3, 3), activation='relu', padding='same')(in_4)
    conv1___ = GroupNormalization(group=G//2)(conv1___)
    #conv1 = WeightNorm(conv1)
    conv1___ = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv1___)
    #conv1 = BatchNormalization()(conv1)
    #conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1___ = MaxPooling3D(pool_size=pool_size)(conv1___)
    if(drop>0):
        pool1___ = Dropout(drop)(pool1___)

    conv2___ = GroupNormalization(group=G)(pool1___)
    #conv2 = WeightNorm(pool1)
    conv2___ = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv2___)
    conv2___ = GroupNormalization(group=G)(conv2___)
    #conv2 = WeightNorm(conv2)

    conv2___ = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2___)
    #conv2 = BatchNormalization()(conv2)
    #conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2___ = MaxPooling3D(pool_size=pool_size)(conv2___)
    if(drop>0):
        pool2___ = Dropout(drop)(pool2___)

    conv3___ = GroupNormalization(group=G)(pool2___)
    #conv3 = WeightNorm(pool2)

    conv3___= Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv3___)
    conv3___ = GroupNormalization(group=G)(conv3___)
    #conv3 = WeightNorm(conv3)

    conv3___ = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3___)
    #conv3 = BatchNormalization()(conv3)
    #conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3___ = MaxPooling3D(pool_size=pool_size)(conv3___)
    if(drop>0):
            pool3_ = Dropout(drop)(pool3___)

    conv4___ = GroupNormalization(group=G)(pool3___)
    #conv4 = WeightNorm(pool3)

    conv4___ = Conv3D(nf*8, (3, 3, 3), activation='relu' ,padding='same')(conv4___)
    #conv4 = BatchNormalization()(conv4)
    #conv4 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv4)
    #conv4 = BatchNormalization()(conv4)
    #conv4 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv4)




    conv4__=concatenate([conv4, conv4_,conv4__,conv4___], axis=4, name='bottleneck')
    conv3__=concatenate([conv3, conv3_,conv3__,conv3___], axis=4)
    conv2__=concatenate([conv2, conv2_,conv2__,conv2___], axis=4)
    conv1__=concatenate([conv1, conv1_,conv1__,conv1___], axis=4)

    #up5 = UpSampling3D()(conv4)
    new_shape = conv3__.shape.as_list()[1:-1]
    up5  = LinearResizeLayer(new_shape,name='up5')(conv4__)

    up5 = concatenate([up5, conv3__], axis=4) # up5 = 512 + conv3 = 256
    up5 = GroupNormalization(group=G)(up5)
    #up5 = WeightNorm(up5)

    conv5 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(up5)
    #conv5 = BatchNormalization()(conv5)
    #conv5 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv5)
    #conv5 = BatchNormalization()(conv5)
    #conv5 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv5)


    #up6 = UpSampling3D()(conv5)
    new_shape = conv2__.shape.as_list()[1:-1]
    up6  = LinearResizeLayer(new_shape,name='up6')(conv5)

    up6 = concatenate([up6, conv2__], axis=4) # up6 = 256 + conv2 = 128
    up6 = GroupNormalization(group=G)(up6)
    #up6 = WeightNorm(up6)

    conv6 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up6)
    #conv6 = BatchNormalization()(conv6)
    #conv6 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv6)
    #conv6 = BatchNormalization()(conv6)
    #conv6 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv6)


    #up7 = UpSampling3D()(conv6)
    new_shape = conv1__.shape.as_list()[1:-1]
    up7   = LinearResizeLayer(new_shape,name='up7')(conv6)

    up7 = concatenate([up7, conv1__], axis=4) # up7 = 128 + conv1 = 64
    up7 = GroupNormalization(group=G)(up7)
    #up7 = WeightNorm(up7)

    conv7 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up7)
    #conv7 = BatchNormalization()(conv7)
    #conv7 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv7)
    #conv7 = BatchNormalization()(conv7)
    #conv7 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv7)

    output = Conv3D(nc, (3, 3, 3), activation='softmax', padding='same')(conv7)

    model = Model(input_img, output)

    return model



def load_UNET3D_SLANT27_v2_groupNorm(ps1,ps2,ps3,ch,nc=4,nf=24,drop=0.5,groups=8,final_act='softmax'):   #3 levels + linear upsampling
    # See https://arxiv.org/pdf/1806.00546.pdf
    #nc: number of output classes
    #nf: number of filter
    #ch: number of channels
    G=groups
    # model UNET 3D
    pool_size=[2,2,2]

    input_img = Input(shape=(ps1, ps2, ps3, ch))     # adapt this if using `channels_first` image data format

    conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(input_img)
    conv1 = GroupNormalization(group=G)(conv1)
    #conv1 = WeightNorm(conv1)
    conv1 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv1)
    #conv1 = BatchNormalization()(conv1)
    #conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size)(conv1)
    if(drop>0):
        pool1 = Dropout(drop)(pool1)

    conv2 = GroupNormalization(group=G)(pool1)
    #conv2 = WeightNorm(pool1)
    conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    conv2 = GroupNormalization(group=G)(conv2)
    #conv2 = WeightNorm(conv2)

    conv2 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv2)
    #conv2 = BatchNormalization()(conv2)
    #conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size)(conv2)
    if(drop>0):
        pool2 = Dropout(drop)(pool2)

    conv3 = GroupNormalization(group=G)(pool2)
    #conv3 = WeightNorm(pool2)

    conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    conv3 = GroupNormalization(group=G)(conv3)
    #conv3 = WeightNorm(conv3)

    conv3 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv3)
    #conv3 = BatchNormalization()(conv3)
    #conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=pool_size)(conv3)
    if(drop>0):
            pool3 = Dropout(drop)(pool3)

    conv4 = GroupNormalization(group=G)(pool3)
    #conv4 = WeightNorm(pool3)

    conv4 = Conv3D(nf*16, (3, 3, 3), activation='relu', name='bottleneck' ,padding='same')(conv4) #,
    #conv4 = BatchNormalization()(conv4)
    #conv4 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv4)
    #conv4 = BatchNormalization()(conv4)
    #conv4 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv4)


    #up5 = UpSampling3D()(conv4)
    new_shape = conv3.shape.as_list()[1:-1]
    up5  = LinearResizeLayer(new_shape,name='up5')(conv4)

    up5 = concatenate([up5, conv3], axis=4) # up5 = 512 + conv3 = 256
    up5 = GroupNormalization(group=G)(up5)
    #up5 = WeightNorm(up5)

    conv5 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(up5)
    #conv5 = BatchNormalization()(conv5)
    #conv5 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv5)
    #conv5 = BatchNormalization()(conv5)
    #conv5 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv5)


    #up6 = UpSampling3D()(conv5)
    new_shape = conv2.shape.as_list()[1:-1]
    up6  = LinearResizeLayer(new_shape,name='up6')(conv5)

    up6 = concatenate([up6, conv2], axis=4) # up6 = 256 + conv2 = 128
    up6 = GroupNormalization(group=G)(up6)
    #up6 = WeightNorm(up6)

    conv6 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up6)
    #conv6 = BatchNormalization()(conv6)
    #conv6 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv6)
    #conv6 = BatchNormalization()(conv6)
    #conv6 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv6)


    #up7 = UpSampling3D()(conv6)
    new_shape = conv1.shape.as_list()[1:-1]
    up7   = LinearResizeLayer(new_shape,name='up7')(conv6)

    up7 = concatenate([up7, conv1], axis=4) # up7 = 128 + conv1 = 64
    up7 = GroupNormalization(group=G)(up7)
    #up7 = WeightNorm(up7)

    conv7 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up7)
    #conv7 = BatchNormalization()(conv7)
    #conv7 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv7)
    #conv7 = BatchNormalization()(conv7)
    #conv7 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv7)


    output = Conv3D(nc, (3, 3, 3), activation=final_act, padding='same')(conv7)

    model = Model(input_img, output)

    return model





def discriminator(ps1,ps2,ps3,ch,nc=4,nf=24,drop=0.5,groups=8):   #3 levels + linear upsampling

    G=groups

    pool_size=[2,2,2]

    input_img = Input(shape=(ps1, ps2, ps3, ch))

    conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(input_img)
    #conv1 = GroupNormalization(group=G)(conv1)
    conv1=BatchNormalization()(conv1)
    conv1 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv1)

    pool1 = MaxPooling3D(pool_size=pool_size)(conv1)
    if(drop>0):
        pool1 = Dropout(drop)(pool1)

    #conv2 = GroupNormalization(group=G)(pool1)
    conv2=BatchNormalization()(pool1)

    conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    #conv2 = GroupNormalization(group=G)(conv2)

    conv2 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv2)

    pool2 = MaxPooling3D(pool_size=pool_size)(conv2)
    if(drop>0):
        pool2 = Dropout(drop)(pool2)

    #conv3 = GroupNormalization(group=G)(pool2)
    conv3=BatchNormalization()(pool2)

    conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    #conv3 = GroupNormalization(group=G)(conv3)

    conv3 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv3)

    pool3 = MaxPooling3D(pool_size=pool_size)(conv3)
    if(drop>0):
            pool3 = Dropout(drop)(pool3)


    pool1=MaxPooling3D(pool_size=pool_size)(input_img)
    pool1=MaxPooling3D(pool_size=pool_size)(pool1)
    pool1=MaxPooling3D(pool_size=pool_size)(pool1)
    #pool2=MaxPooling3D(pool_size=pool_size)(pool2)
    concat= concatenate([pool1,pool3])
    #conv4 = GroupNormalization(group=G)(concat)
    #conv4 = GroupNormalization(group=G)(pool3)
    #conv4=BatchNormalization()(pool3)
    #conv4 = Conv3D(nf*16, (3, 3, 3), activation='relu', name='bottleneck' ,padding='same')(conv4) #,

    flatten=MaxPooling3D(pool_size=pool_size)(concat)
    #flatten= GlobalMaxPooling3D()(conv4)
    flatten=Reshape((-1,))(flatten)
    output=Dense(1,activation='sigmoid')(flatten)
    model = Model(input_img, output)
    return model





def load_UNET3D_SLANT27_v2_groupNorm_lin(ps1,ps2,ps3,ch,nc=4,nf=24,drop=0.5,groups=8):   #3 levels + linear upsampling
    # See https://arxiv.org/pdf/1806.00546.pdf
    #nc: number of output classes
    #nf: number of filter
    #ch: number of channels
    G=groups
    # model UNET 3D
    pool_size=[2,2,2]

    input_img = Input(shape=(ps1, ps2, ps3, ch))     # adapt this if using `channels_first` image data format

    conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(input_img)
    #conv1 = GroupNormalization(group=G)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv1)
    #conv1 = BatchNormalization()(conv1)
    #conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size)(conv1)
    if(drop>0):
        pool1 = Dropout(drop)(pool1)

    #conv2 = GroupNormalization(group=G)(pool1)
    conv2 = BatchNormalization()(pool1)
    conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    conv2 = GroupNormalization(group=G)(conv2)
    #conv2 = WeightNorm(conv2)

    conv2 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv2)
    #conv2 = BatchNormalization()(conv2)
    #conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size)(conv2)
    if(drop>0):
        pool2 = Dropout(drop)(pool2)

    #conv3 = GroupNormalization(group=G)(pool2)
    conv3 = BatchNormalization()(pool2)

    conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    #conv3 = GroupNormalization(group=G)(conv3)
    conv3 = BatchNormalization()(conv3)

    conv3 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv3)
    #conv3 = BatchNormalization()(conv3)
    #conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=pool_size)(conv3)
    if(drop>0):
            pool3 = Dropout(drop)(pool3)

    #conv4 = GroupNormalization(group=G)(pool3)
    conv4 = BatchNormalization()(pool3)

    conv4 = Conv3D(nf*16, (3, 3, 3), activation='relu', name='bottleneck' ,padding='same')(conv4) #,
    #conv4 = BatchNormalization()(conv4)
    #conv4 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv4)
    #conv4 = BatchNormalization()(conv4)
    #conv4 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv4)


    #up5 = UpSampling3D()(conv4)
    new_shape = conv3.shape.as_list()[1:-1]
    up5  = LinearResizeLayer(new_shape,name='up5')(conv4)

    up5 = concatenate([up5, conv3], axis=4) # up5 = 512 + conv3 = 256
    #up5 = GroupNormalization(group=G)(up5)
    up5 = BatchNormalization()(up5)

    conv5 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(up5)
    #conv5 = BatchNormalization()(conv5)
    #conv5 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv5)
    #conv5 = BatchNormalization()(conv5)
    #conv5 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv5)


    #up6 = UpSampling3D()(conv5)
    new_shape = conv2.shape.as_list()[1:-1]
    up6  = LinearResizeLayer(new_shape,name='up6')(conv5)

    up6 = concatenate([up6, conv2], axis=4) # up6 = 256 + conv2 = 128
    #up6 = GroupNormalization(group=G)(up6)
    up6 = BatchNormalization()(up6)

    conv6 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up6)
    #conv6 = BatchNormalization()(conv6)
    #conv6 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv6)
    #conv6 = BatchNormalization()(conv6)
    #conv6 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv6)


    #up7 = UpSampling3D()(conv6)
    new_shape = conv1.shape.as_list()[1:-1]
    up7   = LinearResizeLayer(new_shape,name='up7')(conv6)

    up7 = concatenate([up7, conv1], axis=4) # up7 = 128 + conv1 = 64
    #up7 = GroupNormalization(group=G)(up7)
    up7 = BatchNormalization()(up7)

    conv7 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up7)
    #conv7 = BatchNormalization()(conv7)
    #conv7 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv7)
    #conv7 = BatchNormalization()(conv7)
    #conv7 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv7)

    output = Conv3D(nc, (3, 3, 3), padding='same')(conv7)

    model = Model(input_img, output)

    return model

def load_UNET3D_v3(ps1,ps2,ps3,ch,nc=4,nf=24,drop=0.5):   #3 levels + linear upsampling
    # See https://arxiv.org/pdf/1806.00546.pdf
    #nc: number of output classes
    #nf: number of filter
    #ch: number of channels

    # model UNET 3D
    pool_size=[2,2,2]

    input_img = Input(shape=(ps1, ps2, ps3, ch))     # adapt this if using `channels_first` image data format

    conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(input_img)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size)(conv1)
    if(drop>0):
        pool1 = Dropout(drop)(pool1)

    conv2 = BatchNormalization()(pool1)
    conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size)(conv2)
    if(drop>0):
        pool2 = Dropout(drop)(pool2)

    conv3 = BatchNormalization()(pool2)
    conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv3)

    #pool3 = MaxPooling3D(pool_size=pool_size)(conv3)
    #if(drop>0):
    #        pool3 = Dropout(drop)(pool3)

    #conv4 = BatchNormalization()(pool3)
    #conv4 = Conv3D(nf*16, (3, 3, 3), activation='relu', padding='same')(conv4)


    #new_shape = conv3.shape.as_list()[1:-1]
    #up5  = LinearResizeLayer(new_shape,name='up5')(conv4)

    #up5 = concatenate([up5, conv3], axis=4) # up5 = 512 + conv3 = 256
    #up5 = BatchNormalization()(up5)
    #conv5 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(up5)

    new_shape = conv2.shape.as_list()[1:-1]
    #up6  = LinearResizeLayer(new_shape,name='up6')(conv5)
    up6  = LinearResizeLayer(new_shape,name='up6')(conv3)


    up6 = concatenate([up6, conv2], axis=4) # up6 = 256 + conv2 = 128
    up6 = BatchNormalization()(up6)
    conv6 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up6)

    new_shape = conv1.shape.as_list()[1:-1]
    up7   = LinearResizeLayer(new_shape,name='up7')(conv6)

    up7 = concatenate([up7, conv1], axis=4) # up7 = 128 + conv1 = 64
    up7 = BatchNormalization()(up7)
    conv7 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up7)

    output = Conv3D(nc, (3, 3, 3), activation='softmax', padding='same')(conv7)

    model = Model(input_img, output)

    return model


def load_UNET3D_v3deep(ps1,ps2,ps3,ch,nc=4,nf=24,drop=0.5):   #3 levels + linear upsampling
    # See https://arxiv.org/pdf/1806.00546.pdf
    #nc: number of output classes
    #nf: number of filter
    #ch: number of channels

    # model UNET 3D
    pool_size=[2,2,2]

    input_img = Input(shape=(ps1, ps2, ps3, ch))     # adapt this if using `channels_first` image data format

    conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(input_img)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size)(conv1)
    if(drop>0):
        pool1 = Dropout(drop)(pool1)

    conv2 = BatchNormalization()(pool1)
    conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size)(conv2)
    if(drop>0):
        pool2 = Dropout(drop)(pool2)

    conv3 = BatchNormalization()(pool2)
    conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=pool_size)(conv3)
    if(drop>0):
            pool3 = Dropout(drop)(pool3)

    conv4 = BatchNormalization()(pool3)
    conv4 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv3D(nf*16, (3, 3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling3D(pool_size=pool_size)(conv4)
    if(drop>0):
            pool4 = Dropout(drop)(pool4)

    conv4b = BatchNormalization()(pool4)
    conv4b = Conv3D(nf*32, (3, 3, 3), activation='relu', padding='same')(conv4b)





    new_shape = conv4.shape.as_list()[1:-1]
    up5_  = LinearResizeLayer(new_shape,name='up5_')(conv4b)

    up5_ = concatenate([up5_, conv4], axis=4) # up5 = 512 + conv3 = 256
    up5_ = BatchNormalization()(up5_)
    conv5_ = Conv3D(nf*16, (3, 3, 3), activation='relu', padding='same')(up5_)


    new_shape = conv3.shape.as_list()[1:-1]
    up5  = LinearResizeLayer(new_shape,name='up5')(conv4)

    up5 = concatenate([up5, conv3], axis=4) # up5 = 512 + conv3 = 256
    up5 = BatchNormalization()(up5)
    conv5 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(up5)

    new_shape = conv2.shape.as_list()[1:-1]
    up6  = LinearResizeLayer(new_shape,name='up6')(conv5)


    up6 = concatenate([up6, conv2], axis=4) # up6 = 256 + conv2 = 128
    up6 = BatchNormalization()(up6)
    conv6 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up6)

    new_shape = conv1.shape.as_list()[1:-1]
    up7   = LinearResizeLayer(new_shape,name='up7')(conv6)

    up7 = concatenate([up7, conv1], axis=4) # up7 = 128 + conv1 = 64
    up7 = BatchNormalization()(up7)
    conv7 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up7)

    output = Conv3D(nc, (3, 3, 3), activation='softmax', padding='same')(conv7)

    model = Model(input_img, output)

    return model



def load_UNET3D_MULTITASK(ps1,ps2,ps3,ch,nc=4,nf=24,drop=0.5):   #3 levels + linear upsampling

    pool_size=[2,2,2]

    input_img = Input(shape=(ps1, ps2, ps3, ch))     # adapt this if using `channels_first` image data format

    conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(input_img)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv1)

    pool1 = MaxPooling3D(pool_size=pool_size)(conv1)
    if(drop>0):
        pool1 = Dropout(drop)(pool1)

    conv2 = BatchNormalization()(pool1)
    conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv2)

    pool2 = MaxPooling3D(pool_size=pool_size)(conv2)
    if(drop>0):
        pool2 = Dropout(drop)(pool2)

    conv3 = BatchNormalization()(pool2)
    conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv3)

    pool3 = MaxPooling3D(pool_size=pool_size)(conv3)
    if(drop>0):
            pool3 = Dropout(drop)(pool3)

    conv4 = BatchNormalization()(pool3)
    conv4 = Conv3D(nf*16, (3, 3, 3), activation='relu', padding='same')(conv4)


    new_shape = conv3.shape.as_list()[1:-1]
    up5  = LinearResizeLayer(new_shape,name='up5')(conv4)

    up5 = concatenate([up5, conv3], axis=4) # up5 = 512 + conv3 = 256
    up5 = BatchNormalization()(up5)
    conv5 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(up5)
    up5_  = LinearResizeLayer(new_shape,name='up5_')(conv4)
    up5_ = BatchNormalization()(up5_)
    conv5_ = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(up5_)


    new_shape = conv2.shape.as_list()[1:-1]
    up6  = LinearResizeLayer(new_shape,name='up6')(conv5)

    up6 = concatenate([up6, conv2], axis=4) # up6 = 256 + conv2 = 128
    up6 = BatchNormalization()(up6)
    conv6 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up6)
    up6_  = LinearResizeLayer(new_shape,name='up6_')(conv5_)
    up6_ = BatchNormalization()(up6_)
    conv6_ = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up6_)

    new_shape = conv1.shape.as_list()[1:-1]
    up7   = LinearResizeLayer(new_shape,name='up7')(conv6)

    up7 = concatenate([up7, conv1], axis=4) # up7 = 128 + conv1 = 64
    up7 = BatchNormalization()(up7)
    conv7 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up7)

    output = Conv3D(nc, (3, 3, 3), activation='softmax', padding='same')(conv7)

    up7_   = LinearResizeLayer(new_shape,name='up7_')(conv6_)
    up7_ = BatchNormalization()(up7_)
    conv7_ = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same',name='conv7_')(up7_)
    output_ = Conv3D(nc, (3, 3, 3), padding='same')(conv7_)
    final_output= concatenate([output,output_], axis=4)
    model = Model(input_img, final_output)

    return model



def load_UNET3D_SLANT27_v2(ps1,ps2,ps3,ch,nc=4,nf=24,drop=0.5,softmax=True):   #3 levels + linear upsampling
    # See https://arxiv.org/pdf/1806.00546.pdf
    #nc: number of output classes
    #nf: number of filter
    #ch: number of channels

    # model UNET 3D
    pool_size=[2,2,2]

    input_img = Input(shape=(ps1, ps2, ps3, ch))     # adapt this if using `channels_first` image data format

    conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(input_img)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv1)
    #conv1 = BatchNormalization()(conv1)
    #conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size)(conv1)
    if(drop>0):
        pool1 = Dropout(drop)(pool1)

    conv2 = BatchNormalization()(pool1)
    conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv2)
    #conv2 = BatchNormalization()(conv2)
    #conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size)(conv2)
    if(drop>0):
        pool2 = Dropout(drop)(pool2)

    conv3 = BatchNormalization()(pool2)
    conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv3)
    #conv3 = BatchNormalization()(conv3)
    #conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=pool_size)(conv3)
    if(drop>0):
            pool3 = Dropout(drop)(pool3)

    conv4 = BatchNormalization()(pool3)
    conv4 = Conv3D(nf*16, (3, 3, 3), activation='relu', padding='same')(conv4)
    #conv4 = BatchNormalization()(conv4)
    #conv4 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv4)
    #conv4 = BatchNormalization()(conv4)
    #conv4 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv4)


    #up5 = UpSampling3D()(conv4)
    new_shape = conv3.shape.as_list()[1:-1]
    up5  = LinearResizeLayer(new_shape,name='up5')(conv4)

    up5 = concatenate([up5, conv3], axis=4) # up5 = 512 + conv3 = 256
    up5 = BatchNormalization()(up5)
    conv5 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(up5)
    #conv5 = BatchNormalization()(conv5)
    #conv5 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv5)
    #conv5 = BatchNormalization()(conv5)
    #conv5 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv5)


    #up6 = UpSampling3D()(conv5)
    new_shape = conv2.shape.as_list()[1:-1]
    up6  = LinearResizeLayer(new_shape,name='up6')(conv5)

    up6 = concatenate([up6, conv2], axis=4) # up6 = 256 + conv2 = 128
    up6 = BatchNormalization()(up6)
    conv6 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up6)
    #conv6 = BatchNormalization()(conv6)
    #conv6 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv6)
    #conv6 = BatchNormalization()(conv6)
    #conv6 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv6)


    #up7 = UpSampling3D()(conv6)
    new_shape = conv1.shape.as_list()[1:-1]
    up7   = LinearResizeLayer(new_shape,name='up7')(conv6)

    up7 = concatenate([up7, conv1], axis=4) # up7 = 128 + conv1 = 64
    up7 = BatchNormalization()(up7)
    conv7 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up7)
    #conv7 = BatchNormalization()(conv7)
    #conv7 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv7)
    #conv7 = BatchNormalization()(conv7)
    #conv7 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv7)
    if(softmax):
        output = Conv3D(nc, (3, 3, 3), activation='softmax', padding='same')(conv7)
    else:
        output = Conv3D(nc, (3, 3, 3), padding='same')(conv7)
    model = Model(input_img, output)

    return model



def load_UNET_AE(ps1,ps2,ps3,ch,nc=4,nf=24,drop=0.5):   #3 levels + linear upsampling
    # See https://arxiv.org/pdf/1806.00546.pdf
    #nc: number of output classes
    #nf: number of filter
    #ch: number of channels
    G=8
    # model UNET 3D
    pool_size=[2,2,2]

    input_img = Input(shape=(ps1, ps2, ps3, ch))     # adapt this if using `channels_first` image data format

    conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(input_img)
    conv1 = GroupNormalization(group=G)(conv1)
    conv1 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv1)
    #conv1 = BatchNormalization()(conv1)
    #conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size)(conv1)
    if(drop>0):
        pool1 = Dropout(drop)(pool1)

    conv2 =GroupNormalization(group=G)(pool1)
    conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    conv2 = GroupNormalization(group=G)(conv2)
    conv2 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv2)
    #conv2 = BatchNormalization()(conv2)
    #conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size)(conv2)
    if(drop>0):
        pool2 = Dropout(drop)(pool2)

    conv3 = GroupNormalization(group=G)(pool2)
    conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    conv3 = GroupNormalization(group=G)(conv3)
    conv3 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv3)
    #conv3 = BatchNormalization()(conv3)
    #conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=pool_size)(conv3)
    if(drop>0):
            pool3 = Dropout(drop)(pool3)

    conv4 = GroupNormalization(group=G)(pool3)
    conv4 = Conv3D(nf*16, (3, 3, 3), activation='relu', name='bottleneck',padding='same')(conv4)
    #conv4 = BatchNormalization()(conv4)
    #conv4 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv4)
    #conv4 = BatchNormalization()(conv4)
    #conv4 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv4)


    #up5 = UpSampling3D()(conv4)
    new_shape = conv3.shape.as_list()[1:-1]
    up5  = LinearResizeLayer(new_shape,name='up5')(conv4)

    #up5 = concatenate([up5, conv3], axis=4) # up5 = 512 + conv3 = 256
    up5 = GroupNormalization(group=G)(up5)
    conv5 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(up5)
    #conv5 = BatchNormalization()(conv5)
    #conv5 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv5)
    #conv5 = BatchNormalization()(conv5)
    #conv5 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv5)


    #up6 = UpSampling3D()(conv5)
    new_shape = conv2.shape.as_list()[1:-1]
    up6  = LinearResizeLayer(new_shape,name='up6')(conv5)

    #up6 = concatenate([up6, conv2], axis=4) # up6 = 256 + conv2 = 128
    up6 = GroupNormalization(group=G)(up6)
    conv6 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up6)
    #conv6 = BatchNormalization()(conv6)
    #conv6 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv6)
    #conv6 = BatchNormalization()(conv6)
    #conv6 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv6)


    #up7 = UpSampling3D()(conv6)
    new_shape = conv1.shape.as_list()[1:-1]
    up7   = LinearResizeLayer(new_shape,name='up7')(conv6)

    #up7 = concatenate([up7, conv1], axis=4) # up7 = 128 + conv1 = 64
    up7 = GroupNormalization(group=G)(up7)
    conv7 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up7)
    #conv7 = BatchNormalization()(conv7)
    #conv7 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv7)
    #conv7 = BatchNormalization()(conv7)
    #conv7 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv7)

    output1 = Conv3D(nc//2, (3, 3, 3), padding='same')(conv7)
    output2 = Conv3D(nc//2, (3, 3, 3), activation='softmax', padding='same')(conv7)
    #bottleneck_reduced=MaxPooling3D(pool_size=pool_size ,name='bottleneck_reduced')(conv4)
    output= concatenate([output1,output2], axis=4)
    model = Model(input_img, output)

    return model

def load_UNET3D_CRFupsampling(ps1,ps2,ps3,ch,nc=4,nf=24,drop=0.5):   #3 levels + linear upsampling
    # See https://arxiv.org/pdf/1806.00546.pdf
    #nc: number of output classes
    #nf: number of filter
    #ch: number of channels

    # model UNET 3D
    pool_size=[2,2,2]

    #input_img = Input(shape=(ps1, ps2, ps3, ch))     # adapt this if using `channels_first` image data format
    input_img = Input(batch_shape=(1,ps1, ps2, ps3, ch))
    conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(input_img)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv1)
    #conv1 = BatchNormalization()(conv1)
    #conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size)(conv1)
    if(drop>0):
        pool1 = Dropout(drop)(pool1)

    conv2 = BatchNormalization()(pool1)
    conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv2)
    #conv2 = BatchNormalization()(conv2)
    #conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size)(conv2)
    if(drop>0):
        pool2 = Dropout(drop)(pool2)

    conv3 = BatchNormalization()(pool2)
    conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv3)
    #conv3 = BatchNormalization()(conv3)
    #conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=pool_size)(conv3)
    if(drop>0):
            pool3 = Dropout(drop)(pool3)

    conv4 = BatchNormalization()(pool3)
    conv4 = Conv3D(nf*16, (3, 3, 3), activation='relu', padding='same')(conv4)
    #conv4 = BatchNormalization()(conv4)
    #conv4 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv4)
    #conv4 = BatchNormalization()(conv4)
    #conv4 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv4)


    #up5 = UpSampling3D()(conv4)
    new_shape = conv3.shape.as_list()[1:-1]
    up5  = LinearResizeLayer(new_shape,name='up5')(conv4)
    up5  = up_crf(num_classes=2, theta_alpha=20,theta_beta=0.2,theta_gamma=20,num_iterations=5)([up5,input_img])

    up5 = concatenate([up5, conv3], axis=4) # up5 = 512 + conv3 = 256
    up5 = BatchNormalization()(up5)
    conv5 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(up5)
    #conv5 = BatchNormalization()(conv5)
    #conv5 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv5)
    #conv5 = BatchNormalization()(conv5)
    #conv5 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv5)


    #up6 = UpSampling3D()(conv5)
    new_shape = conv2.shape.as_list()[1:-1]
    up6  = LinearResizeLayer(new_shape,name='up6')(conv5)
    up6  = up_crf(num_classes=2, theta_alpha=20,theta_beta=0.2,theta_gamma=20,num_iterations=5)([up6,input_img])

    up6 = concatenate([up6, conv2], axis=4) # up6 = 256 + conv2 = 128
    up6 = BatchNormalization()(up6)
    conv6 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up6)
    #conv6 = BatchNormalization()(conv6)
    #conv6 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv6)
    #conv6 = BatchNormalization()(conv6)
    #conv6 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv6)


    #up7 = UpSampling3D()(conv6)
    new_shape = conv1.shape.as_list()[1:-1]
    up7   = LinearResizeLayer(new_shape,name='up7')(conv6)
    up7  = up_crf(num_classes=2, theta_alpha=20,theta_beta=0.2,theta_gamma=20,num_iterations=5)([up7,input_img])

    up7 = concatenate([up7, conv1], axis=4) # up7 = 128 + conv1 = 64
    up7 = BatchNormalization()(up7)
    conv7 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up7)
    #conv7 = BatchNormalization()(conv7)
    #conv7 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv7)
    #conv7 = BatchNormalization()(conv7)
    #conv7 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv7)

    output = Conv3D(nc, (3, 3, 3),padding='same')(conv7) # activation='softmax',
    #output= CRF_Layer(image_dims=(80,96,80) ,num_classes=2)(output) #([output,input_img])
    output = CRF_RNN_Layer(num_classes=2, theta_alpha=20,theta_beta=0.2,theta_gamma=20,num_iterations=5)([output,input_img])
    output= Activation(softmax)(output)
    model = Model(input_img, output)

    return model


def load_UNET3D_SLANT27_v2_withCRF(ps1,ps2,ps3,ch,nc=4,nf=24,drop=0.5):   #3 levels + linear upsampling
    # See https://arxiv.org/pdf/1806.00546.pdf
    #nc: number of output classes
    #nf: number of filter
    #ch: number of channels

    # model UNET 3D
    pool_size=[2,2,2]

    #input_img = Input(shape=(ps1, ps2, ps3, ch))     # adapt this if using `channels_first` image data format
    input_img = Input(batch_shape=(1,ps1, ps2, ps3, ch))
    conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(input_img)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv1)
    #conv1 = BatchNormalization()(conv1)
    #conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size)(conv1)
    if(drop>0):
        pool1 = Dropout(drop)(pool1)

    conv2 = BatchNormalization()(pool1)
    conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv2)
    #conv2 = BatchNormalization()(conv2)
    #conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size)(conv2)
    if(drop>0):
        pool2 = Dropout(drop)(pool2)

    conv3 = BatchNormalization()(pool2)
    conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv3)
    #conv3 = BatchNormalization()(conv3)
    #conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=pool_size)(conv3)
    if(drop>0):
            pool3 = Dropout(drop)(pool3)

    conv4 = BatchNormalization()(pool3)
    conv4 = Conv3D(nf*16, (3, 3, 3), activation='relu', padding='same')(conv4)
    #conv4 = BatchNormalization()(conv4)
    #conv4 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv4)
    #conv4 = BatchNormalization()(conv4)
    #conv4 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv4)


    #up5 = UpSampling3D()(conv4)
    new_shape = conv3.shape.as_list()[1:-1]
    up5  = LinearResizeLayer(new_shape,name='up5')(conv4)

    up5 = concatenate([up5, conv3], axis=4) # up5 = 512 + conv3 = 256
    up5 = BatchNormalization()(up5)
    conv5 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(up5)
    #conv5 = BatchNormalization()(conv5)
    #conv5 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv5)
    #conv5 = BatchNormalization()(conv5)
    #conv5 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv5)


    #up6 = UpSampling3D()(conv5)
    new_shape = conv2.shape.as_list()[1:-1]
    up6  = LinearResizeLayer(new_shape,name='up6')(conv5)

    up6 = concatenate([up6, conv2], axis=4) # up6 = 256 + conv2 = 128
    up6 = BatchNormalization()(up6)
    conv6 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up6)
    #conv6 = BatchNormalization()(conv6)
    #conv6 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv6)
    #conv6 = BatchNormalization()(conv6)
    #conv6 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv6)


    #up7 = UpSampling3D()(conv6)
    new_shape = conv1.shape.as_list()[1:-1]
    up7   = LinearResizeLayer(new_shape,name='up7')(conv6)

    up7 = concatenate([up7, conv1], axis=4) # up7 = 128 + conv1 = 64
    up7 = BatchNormalization()(up7)
    conv7 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up7)
    #conv7 = BatchNormalization()(conv7)
    #conv7 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv7)
    #conv7 = BatchNormalization()(conv7)
    #conv7 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv7)

    output = Conv3D(nc, (3, 3, 3),padding='same')(conv7) # activation='softmax',
    #output= CRF_Layer(image_dims=(80,96,80) ,num_classes=2)(output) #([output,input_img])
    output = CRF_RNN_Layer(num_classes=2, theta_alpha=20,theta_beta=0.2,theta_gamma=20,num_iterations=5)([output,input_img])
    output= Activation(softmax)(output)
    model = Model(input_img, output)

    return model

#def CRF_Layer(x):
#    return tf.nn.fractional_max_pool(x,p_ratio)[0]

def load_UNET3D_feature_consistency(ps1,ps2,ps3,ch,nc=4,nf=24,drop=0.5):   #3 levels + linear upsampling
    # See https://arxiv.org/pdf/1806.00546.pdf
    #nc: number of output classes
    #nf: number of filter
    #ch: number of channels

    # model UNET 3D
    pool_size=[2,2,2]

    input_img = Input(shape=(ps1, ps2, ps3, ch))     # adapt this if using `channels_first` image data format

    conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(input_img)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv1)
    #conv1 = BatchNormalization()(conv1)
    #conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size)(conv1)
    if(drop>0):
        pool1 = Dropout(drop)(pool1)

    conv2 = BatchNormalization()(pool1)
    conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv2)
    #conv2 = BatchNormalization()(conv2)
    #conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size)(conv2)
    if(drop>0):
        pool2 = Dropout(drop)(pool2)

    conv3 = BatchNormalization()(pool2)
    conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv3)

    #conv3 = BatchNormalization()(conv3)
    #conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=pool_size)(conv3)
    if(drop>0):
            pool3 = Dropout(drop)(pool3)

    conv4 = BatchNormalization()(pool3)
    conv4 = Conv3D(nf*16, (3, 3, 3), activation='relu',name='bottleneck', padding='same')(conv4)
    #conv4 = BatchNormalization()(conv4)
    #conv4 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv4)
    #conv4 = BatchNormalization()(conv4)
    #conv4 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv4)


    #up5 = UpSampling3D()(conv4)
    new_shape = conv3.shape.as_list()[1:-1]
    up5  = LinearResizeLayer(new_shape,name='up5')(conv4)

    up5 = concatenate([up5, conv3], axis=4) # up5 = 512 + conv3 = 256
    up5 = BatchNormalization()(up5)
    conv5 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(up5)
    #conv5 = BatchNormalization()(conv5)
    #conv5 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv5)
    #conv5 = BatchNormalization()(conv5)
    #conv5 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv5)


    #up6 = UpSampling3D()(conv5)
    new_shape = conv2.shape.as_list()[1:-1]
    up6  = LinearResizeLayer(new_shape,name='up6')(conv5)

    up6 = concatenate([up6, conv2], axis=4) # up6 = 256 + conv2 = 128
    up6 = BatchNormalization()(up6)
    conv6 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up6)
    #conv6 = BatchNormalization()(conv6)
    #conv6 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv6)
    #conv6 = BatchNormalization()(conv6)
    #conv6 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv6)


    #up7 = UpSampling3D()(conv6)
    new_shape = conv1.shape.as_list()[1:-1]
    up7   = LinearResizeLayer(new_shape,name='up7')(conv6)

    up7 = concatenate([up7, conv1], axis=4) # up7 = 128 + conv1 = 64
    up7 = BatchNormalization()(up7)
    conv7 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up7)
    #conv7 = BatchNormalization()(conv7)
    #conv7 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv7)
    #conv7 = BatchNormalization()(conv7)
    #conv7 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv7)

    output = Conv3D(nc, (3, 3, 3), activation='softmax',name='segmentation', padding='same')(conv7)
    bottleneck_reduced=MaxPooling3D(pool_size=pool_size,name='bottleneck_')(conv4)
    #bottleneck_reduced=MaxPooling3D(pool_size=pool_size)(conv4)
    model = Model(input_img, [output,bottleneck_reduced])

    return model
#name='bottleneck',
def load_UNET3D_SLANT27_v2_supervised(ps1,ps2,ps3,ch,nc=4,nf=24,drop=0.5):   #3 levels + linear upsampling + deep_supervision
    # See https://arxiv.org/pdf/1806.00546.pdf
    #nc: number of output classes
    #nf: number of filter
    #ch: number of channels

    # model UNET 3D
    pool_size=[2,2,2]

    input_img = Input(shape=(ps1, ps2, ps3, ch))     # adapt this if using `channels_first` image data format

    conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(input_img)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size)(conv1)
    if(drop>0):
        pool1 = Dropout(drop)(pool1)

    conv2 = BatchNormalization()(pool1)
    conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size)(conv2)
    if(drop>0):
        pool2 = Dropout(drop)(pool2)

    conv3 = BatchNormalization()(pool2)
    conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=pool_size)(conv3)
    if(drop>0):
            pool3 = Dropout(drop)(pool3)

    conv4 = BatchNormalization()(pool3)
    conv4 = Conv3D(nf*16, (3, 3, 3), activation='relu', padding='same')(conv4)

    #up5 = UpSampling3D()(conv4)
    new_shape = conv3.shape.as_list()[1:-1]
    up5  = LinearResizeLayer(new_shape,name='up5')(conv4)

    up5 = concatenate([up5, conv3], axis=4) # up5 = 512 + conv3 = 256
    up5 = BatchNormalization()(up5)
    conv5 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(up5)

    #up6 = UpSampling3D()(conv5)
    new_shape = conv2.shape.as_list()[1:-1]
    up6  = LinearResizeLayer(new_shape,name='up6')(conv5)

    up6 = concatenate([up6, conv2], axis=4) # up6 = 256 + conv2 = 128
    up6 = BatchNormalization()(up6)
    conv6 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up6)

    output0 = Conv3D(nc, (3, 3, 3), activation='softmax', padding='same')(conv6)


    #up7 = UpSampling3D()(conv6)
    new_shape = conv1.shape.as_list()[1:-1]
    up7   = LinearResizeLayer(new_shape,name='up7')(conv6)

    up7 = concatenate([up7, conv1], axis=4) # up7 = 128 + conv1 = 64
    up7 = BatchNormalization()(up7)
    conv7 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up7)

    output = Conv3D(nc, (3, 3, 3), activation='softmax', padding='same')(conv7)

    model = Model(input_img, [output0 ,output])

    # for train: call this net like this:
    # model=modelos.load_UNET3D_SLANT27_v2_supervised(ps[0],ps[1],ps[2],ch,nc,nf,drop)
    # model.compile(optimizer='adam', loss=losses.mdice_loss, loss_weights=[0.3,0.7],metrics=[metrics.mdice])
    # ....
    # result=model.fit(x_train, [y_train[:,::2,::2,::2,:],y_train], ....
    # for test: call this net like this:
    # a,res = model.predict(x)

    return model

def load_nonlocal_unet_3D(input_shape, n_classes, n_filters, dropout=None, upsampling='default'):
    """Simple 3D U-Net, with depth 4."""
    if upsampling not in ('default', 'linear'):
        raise ValueError('Wrong upsampling layer. Choose between `default` or `linear`.')

    conv_config = {'kernel_size': (3, 3, 3), 'padding': 'same', 'activation': 'relu'}
    pooling_size = (2, 2, 2)

    # Level 1
    with tf.variable_scope('level1'):
        input_layer = Input(input_shape)
        conv1 = Conv3D(n_filters, **conv_config)(input_layer)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv3D(n_filters * 2, **conv_config)(conv1)
        # conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling3D(pooling_size)(conv1)
        if dropout:
            pool1 = Dropout(dropout)(pool1)

    # Level 2
    with tf.variable_scope('level2'):
        conv2 = BatchNormalization()(pool1)
        conv2 = Conv3D(n_filters * 2, **conv_config)(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv3D(n_filters * 4, **conv_config)(conv2)
        # conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling3D(pooling_size)(conv2)
        if dropout:
            pool2 = Dropout(dropout)(pool2)

    # Level 3
    with tf.variable_scope('level3'):
        conv3 = BatchNormalization()(pool2)
        conv3 = Conv3D(n_filters * 4, **conv_config)(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv3D(n_filters * 8, **conv_config)(conv3)
        # conv3 = BatchNormalization()(conv3)
        # conv3 = non_local_block(conv3,intermediate_dim=None, compression=4,
        #                             mode='embedded', add_residual=True)
        # NLconv3 = BatchNormalization()(conv3)
        # NLconv3 = non_local_block(NLconv3)

        # NLconv3 = BatchNormalization()(NLconv3)
        pool3 = MaxPooling3D(pooling_size)(conv3)
        if dropout:
            pool3 = Dropout(dropout)(pool3)

    # Level 4
    with tf.variable_scope('level4'):
        # conv4 = Conv3D(n_filters * 8, **conv_config)(pool3)
        conv4 = BatchNormalization()(pool3)
        conv4 = Conv3D(n_filters * 16, **conv_config)(conv4)
        #conv4 = BatchNormalization()(conv4)
        # conv4 = BatchNormalization()(conv4)
        # conv4 = non_local_block(conv4,intermediate_dim=None, compression=4,
        #                             mode='embedded', add_residual=True)
        NLconv4 = BatchNormalization()(conv4)
        NLconv4 = non_local_block(NLconv4)
        conv4 = Concatenate()([conv4, NLconv4])

    # Level 3 Up
    with tf.variable_scope('level3_up'):
        if upsampling == 'default':
            up3 = UpSampling3D()(conv4)
        elif upsampling == 'linear':
            new_size = conv3.shape.as_list()[1:-1]
            up3 = LinearResizeLayer(new_size, name='up3')(conv4)
        concat3 = Concatenate()([up3, conv3])
        upconv3 = BatchNormalization()(concat3)
        upconv3 = Conv3D(n_filters * 8, **conv_config)(upconv3)
        upconv3 = BatchNormalization()(upconv3)

    # Level 2 Up
    with tf.variable_scope('level2_up'):
        if upsampling == 'default':
            up2 = UpSampling3D()(upconv3)
        elif upsampling == 'linear':
            new_size = conv2.shape.as_list()[1:-1]
            up2 = LinearResizeLayer(new_size, name='up2')(upconv3)
        concat2 = Concatenate()([up2, conv2])
        upconv2 = BatchNormalization()(concat2)
        upconv2 = Conv3D(n_filters * 4, **conv_config)(upconv2)
        upconv2 = BatchNormalization()(upconv2)

    # Level 1 Up
    with tf.variable_scope('level1_up'):
        if upsampling == 'default':
            up1 = UpSampling3D()(upconv2)
        elif upsampling == 'linear':
            new_size = conv1.shape.as_list()[1:-1]
            up1 = LinearResizeLayer(new_size, name='up1')(upconv2)
        concat1 = Concatenate()([up1, conv1])
        upconv1 = BatchNormalization()(concat1)
        upconv1 = Conv3D(n_filters * 4, **conv_config)(upconv1)
        upconv1 = BatchNormalization()(upconv1)

    # Softmax layer
    output_layer = Conv3D(n_classes, (1, 1, 1), padding='same', activation='softmax')(upconv1)

    # Connect layers and return Model
    unet_model = Model(input_layer, output_layer)
    return unet_model
