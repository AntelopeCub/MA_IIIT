from copy import deepcopy

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dropout,
                                     GlobalAveragePooling2D, MaxPooling2D,
                                     ReLU, Softmax)
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

from .f_membership_loss import cat_crossentropy_from_logits
from .Q_ActivationNormLayers import (ActivationLayer_relu,
                                     ActivationLayer_signed)
from .Q_Conv2dNorm import Conv2dNorm
from .Q_RegularizersNorm import discretize_reg, max_weights_reg

"""
Build VGG-Net hier werden nur Anzahl der Faltungsschichten gezaehlt. VGG-16 hat
somit den Schluessel/ eine Anzahl der Layer von 13. ("Ueberall miunus 3")
"""

CUSTOM_OBJ = {'Conv2dNorm':Conv2dNorm, 'max_weights_reg':max_weights_reg,
              'ActivationLayer_signed':ActivationLayer_signed, 'ActivationLayer_relu':ActivationLayer_relu,
              'discretize_reg': discretize_reg, 'cat_crossentropy_from_logits': cat_crossentropy_from_logits}

dist_kern = { '7': [64, 64, 128, 128, 256, 256, 256],
              '8': [64,     128,      256, 256,           512, 512, 512, 512] ,
             '10': [64, 64, 128, 128, 256, 256,           512, 512, 512, 512] ,
             '13': [64, 64, 128, 128, 256, 256, 256,      512, 512, 512, 512, 512, 512],
             '16': [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
            }

dist_pool = { '7': [1, 3, 6], 
              '8': [0, 1, 3,  5,  7],
             '10': [1, 3, 5,  7,  9],
             '13': [1, 3, 6,  9, 12],
             '16': [1, 3, 7, 11, 15]
             }

def get_kern_pool(layers, inputs, verbose=True):
    # Lade Konfigureation
    if layers in dist_kern.keys():
        nbr_kern = dist_kern[layers]
        pos_pool = dist_pool[layers]
    elif int(layers) < 13:
        if verbose==True:
            print(layers + ' ist ungueltig - verwende die ersten ' + layers + ' Schichten von VGG-16' )
        nbr_kern = dist_kern['13'][:int(layers)]
        pos_pool = dist_pool['13'][:int(layers)]
    elif int(layers) < 16:
        if verbose==True:
            print(layers + ' ist ungueltig - verwende die ersten ' + layers + ' Schichten von VGG-19' )
        nbr_kern = dist_kern['16'][:int(layers)]
        pos_pool = dist_pool['16'][:int(layers)]
    else:
        if verbose==True:
            print(layers + ' ist ungueltig - verwende VGG-16' )
        nbr_kern = dist_kern['13']
        pos_pool = dist_pool['13']
    # Passe Max-Pooling-Layer an falls Eingabe zu klein
    if np.minimum(inputs[0], inputs[1]) < 128:        
        if np.minimum(inputs[0], inputs[1]) < 32:
            pos_pool = pos_pool[:2]
            del_pool = 3
        elif np.minimum(inputs[0], inputs[1]) < 64:
            pos_pool = pos_pool[:3]
            del_pool = 2
        else:
            pos_pool = pos_pool[:4]
            del_pool = 1
        if verbose==True:
            print('Eingabe zu klein! Entferne %i Max-Pooling-Schicht(en)' % ( del_pool ) )
        
    return nbr_kern, pos_pool

"""
BASELINE MODEL
"""
def f_build_vgg_con(inputs, l2_norm, outputs, layers='13', L_A=[None, None], verbose=True) :
    if verbose==True:
        print('Build VGG-Net')
    # Lade Konfigureation
    nbr_kern, pos_pool = get_kern_pool(layers, inputs, verbose=verbose)
        
    # Baue Modell
    model = Sequential()   
    model.add(Conv2D(nbr_kern[0], 
                         kernel_size=(3, 3), 
                         padding='same',
                         kernel_initializer = "he_normal",
                         kernel_regularizer=l2(l2_norm),
                         bias_regularizer=l2(l2_norm),
                         use_bias = False,
                         input_shape=inputs))
    # if L_A[1]:
    #     model.add(ActivationLayer_signed(L_A=[L_A[0], L_A[1]]))  
    model.add(BatchNormalization())
    model.add(ReLU())
    if 0 in pos_pool:
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    for i1 in range(1, len(nbr_kern)):
        model.add(Conv2D(nbr_kern[i1], 
                         kernel_size=(3, 3), 
                         padding='same',
                         kernel_initializer = "he_normal",
                         kernel_regularizer=l2(l2_norm),
                         bias_regularizer=l2(l2_norm),
                         use_bias = False))
        # if L_A[1]:
        #     model.add(ActivationLayer_signed(L_A=[L_A[0], L_A[1]]))  
        model.add(BatchNormalization())
        model.add(ReLU())
        if i1 in pos_pool:
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    # model.add(Dropout(rate=0.5))
    model.add(Conv2D(outputs, name='c1x1_2d',
                        kernel_size=(1,1), 
                        kernel_regularizer=l2(l2_norm),
                        bias_regularizer=l2(l2_norm),
                        kernel_initializer = "he_normal"))
    model.add(GlobalAveragePooling2D())
    # model.add(Softmax())
    return model
    


"""
Q MODEL
"""
def f_build_vgg_qua(inputs, l2_norm, outputs, L_A, L_W, layers='13',verbose=False) :
    nbr_kern, pos_pool = get_kern_pool(layers, inputs, verbose=verbose)
        
    # Baue Modell
    model = Sequential()   
    model.add(Conv2dNorm(nbr_kern[0], 
                         kernel_size=(3, 3), 
                         padding='same',
                         bias_initializer = "he_normal",
                         kernel_initializer = "he_normal",
                         kernel_regularizer=l2(l2_norm),
                         bias_regularizer=l2(l2_norm),
                         L_A = L_A,  L_W=L_W,
                         input_shape=inputs))
    if 0 in pos_pool:
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    for i1 in range(1, len(nbr_kern)):
        model.add(Conv2dNorm(nbr_kern[i1], 
                         kernel_size=(3, 3), 
                         padding='same',
                         bias_initializer = "he_normal",
                         kernel_initializer = "he_normal",
                         kernel_regularizer=l2(l2_norm),
                         bias_regularizer=l2(l2_norm),
                         L_A = L_A,  L_W=L_W))
        if i1 in pos_pool:
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    # model.add(Dropout(rate=0.5))
    model.add(Conv2D(outputs, name='c1x1_2d',
                        kernel_size=(1,1), 
                        bias_initializer = "he_normal",
                        kernel_initializer = "he_normal",
                        kernel_regularizer=max_weights_reg(L_W = L_W, l2= 0),
                        bias_regularizer=max_weights_reg(L_W = L_W, l2= 0) ))
    
    model.add(ActivationLayer_signed(L_A=[L_A[0], L_A[1]]))   
    model.add(GlobalAveragePooling2D())
    # model.add(Softmax())
    return model
    

def f_convert_model(model, optimizer, L_W, L_A, custom_obj, verbose=False):
    cfg = model.get_config()
    cfg_reg = {'class_name': 'discretize_reg', 'config': {"FL": L_W, "lg": 1e-4, "l2":0}}
    weights_new = []
    idx1, idx2 = 0, 0
    while cfg['layers'][idx1]['class_name'] != 'GlobalAveragePooling2D':
        if cfg['layers'][idx1]['class_name'] == 'Conv2dNorm':
            cfg_conv2d   = deepcopy(cfg['layers'][idx1])
            cfg_conv2d['config']['kernel_regularizer'] = cfg_reg
            cfg_conv2d['config']['bias_regularizer'] = cfg_reg
            cfg_conv2d['class_name'] = 'Conv2D'
            cfg_conv2d['config']['name'] = 'conv2d_new_' + str(idx2)
            del cfg_conv2d['config']['momentum'], cfg_conv2d['config']['max_scale'], cfg_conv2d['config']['L_W'], cfg_conv2d['config']['L_A']
            cfg_aktiv = {'class_name': 'ActivationLayer_relu', 'config': { "L_A": L_A} }
            
            del cfg['layers'][idx1]
            cfg['layers'].insert(idx1, cfg_aktiv)
            cfg['layers'].insert(idx1, cfg_conv2d)
            
            weights_tmp = model.layers[idx2].get_weights()
            weights_new.append(weights_tmp[0] * weights_tmp[2])
            weights_new.append(weights_tmp[1] * weights_tmp[2])
            idx1 += 1
            
        elif cfg['layers'][idx1]['class_name'] == 'Conv2D':
            cfg['layers'][idx1]['config']['kernel_regularizer'] = cfg_reg
            cfg['layers'][idx1]['config']['bias_regularizer'] = cfg_reg
            weights_tmp = model.layers[idx2].get_weights()
            for wei in weights_tmp:
                weights_new.append(wei)
        elif cfg['layers'][idx1]['class_name'] == 'SepConv2DNorm':
            cfg_sepcon   = deepcopy(cfg['layers'][idx1]['config'])
            cfg_depthw   = {'class_name': 'DepthwiseConv2D', 'config': { "depthwise_initializer": cfg_sepcon['depthwise_initializer'],
                           'depthwise_regularizer': cfg_reg, 'depthwise_constraint': cfg_sepcon['depthwise_constraint'], 'name': 'sepconv2d_new_' + str(idx2), 
                           'strides': cfg_sepcon['strides'], 'trainable': True, 'kernel_size': (3,3), 'use_bias': False, 'padding': 'same',} }
            cfg_aktiv    = {'class_name': 'ActivationLayer_relu', 'config': { "L_A": L_A} }
            cfg_conv2d   = {'class_name': 'Conv2D', 'config': { "kernel_initializer": cfg_sepcon['kernel_initializer'], "bias_initializer": cfg_sepcon['bias_initializer'],
                           'kernel_constraint': cfg_sepcon['kernel_constraint'], 'bias_constraint': cfg_sepcon['bias_constraint'],
                           'kernel_regularizer': cfg_reg, 'bias_regularizer': cfg_reg,  'name': 'conv2d_new_' + str(idx2),
                           'filters': cfg_sepcon['filters'], 'kernel_size': (1,1), 'padding': 'same',
                           'strides': cfg_sepcon['strides'], 'trainable': True,  'use_bias': True} }
            del cfg['layers'][idx1]
            cfg['layers'].insert(idx1, cfg_aktiv)
            cfg['layers'].insert(idx1, cfg_conv2d)
            cfg['layers'].insert(idx1, cfg_aktiv)
            cfg['layers'].insert(idx1, cfg_depthw)
     
            weights_tmp = model.layers[idx2].get_weights()
            weights_new.append(weights_tmp[0])
            weights_new.append(weights_tmp[1] * weights_tmp[3])
            weights_new.append(weights_tmp[2] * weights_tmp[3])
            idx1 += 3
        else:
            weights_tmp = model.layers[idx2].get_weights()
            for wei in weights_tmp:
                weights_new.append(wei)
        idx1 += 1
        idx2 += 1
        
    model_conv = Sequential.from_config(cfg, custom_objects=custom_obj)
    model_conv.set_weights(weights_new)
    model_conv.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc', cat_crossentropy_from_logits]) 
    if verbose == True:
        model_conv.summary()
    return model_conv
