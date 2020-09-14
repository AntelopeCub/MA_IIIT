"""
Funktion fuer die Diskretisierung der Gewichte eines Keras-Modells:
    - Faktoren der 8-Bit-Normalisierung in Gewichte
    - Clippen und Diskretisieren der CONV-Gewichte 
    - Gewichte: signed 8 Bit (VZ= 1, IL=0, FL=7 -> -1...1)
    - Aktivierung: unsigned 8 Bit (VZ=0, IL=1, FL=7 -> 0...2)
    
Gute Parameter:
    - L2-Norm: 3e-4     (nur fuer Kernel und Biases der Conv-Layer)
    - Min-Max-Norm: +/-1 (fuer alle lernbaren Paramter)
    - 8-Bit-Normalisierung: Momentum 0.93, Puffer 0.96, Max. Skalierung 5

"""
import numpy as np
from copy import deepcopy

BASE = 2
layer_names = ['InputLayer', 'Reshape', 'Lambda',
               'MaxPooling2D', 'GlobalAveragePooling2D', 
               'SpatialDropout2D', 'Dropout', 'BatchNormalization',
               'Activation', 'ReLU', 'Softmax', 
               'ActivationLayer_relu', 'ActivationLayer_signed']

def weight_discretization(model, L_CONV=(1, 7), L_FC=(1, 7), L_BN=(2, 6)):
    # Minimal und Maximal erlaubter Wert
    max_value_c  = BASE**(L_CONV[0]-1) - (BASE**-L_CONV[1]) 
    min_value_c  = -BASE**(L_CONV[0]-1)
    max_value_fc = BASE**(L_FC[0]-1) - (BASE**-L_FC[1]) 
    min_value_fc = -BASE**(L_FC[0]-1)
    max_value_bn = BASE**(L_BN[0]-1) - (BASE**-L_FC[1]) 
    min_value_bn = -BASE**(L_BN[0]-1)
    # Config aus Modell einlesen
    cfg = model.get_config()
    

    def f_discrete(model, idx, cfg):
        name = cfg['layers'][idx]['class_name']
        if name == 'Conv2dNorm': # Conv2dNorm hat immer 3 Gewichte!!!
            weights = model.layers[idx].get_weights()
            weights[0] *= weights[2]
            weights[1] *= weights[2]
            weights[2] = np.ones_like(weights[2]) 
    
            weights[0]   = np.clip(weights[0], min_value_c, max_value_c)
            weights[1]   = np.clip(weights[1], min_value_c, max_value_c)
            
            weights[0]   = np.round(weights[0] * BASE**L_CONV[1]) * BASE**-L_CONV[1]
            weights[1]   = np.round(weights[1] * BASE**L_CONV[1]) * BASE**-L_CONV[1]
            model.layers[idx].set_weights(deepcopy(weights))

        elif name == 'Conv2D' or name == 'DepthwiseConv2D':
            weights = model.layers[idx].get_weights()
            for i1 in range(len(weights)):
                weights[i1]   = np.clip(weights[i1], min_value_c, max_value_c)
                weights[i1]   = np.round(weights[i1] * BASE**(L_CONV[1])) * BASE**(-L_CONV[1])
            model.layers[idx].set_weights(deepcopy(weights))
            
        elif name == 'Dense': # gleich wie Conv2D, aber anderes Bitschema verwendet
            weights = model.layers[i1].get_weights()
            for i1 in range(len(weights)):
                weights[i1]   = np.clip(weights[i1], min_value_fc, max_value_fc)
                weights[i1]   = np.round(weights[i1] * BASE**L_FC[1]) * BASE**-L_FC[1]
            model.layers[idx].set_weights(deepcopy(weights))

#        elif name == 'BatchNormalization':
#            weights = model.layers[i1].get_weights()
#            for i1 in range(len(weights)):
#                weights[i1]   = np.clip(weights[i1], min_value_bn, max_value_bn)
#                weights[i1]   = np.round(weights[i1] * BASE**L_BN[1]) * BASE**-L_BN[1]
#            model.layers[idx].set_weights(deepcopy(weights))

        elif name == 'Model':
            for idx2 in range(len(cfg['layers'][idx]['config']['layers'])):
                model.layers[idx] = f_discrete(model.layers[idx], idx2, cfg['layers'][idx]['config'])
        elif name in layer_names:
            pass
        else:
            print('Fehler: Kenne ' + name + ' nicht.')
        return model

    for i1 in range(len(cfg['layers'])):
        model = f_discrete(model, i1, cfg)
           
            
    return model


