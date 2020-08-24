import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers


class max_weights_reg(regularizers.Regularizer):
    def __init__(self, L_W=[2,6], lm=1e2, l2=0, l1=0):
        self.L_W = L_W
        self.w_max = 2**(L_W[0]-1)- 2**(-L_W[1]-1)
        self.l2 = l2
        self.l1 = l1
        self.lm = lm

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, inputs):
        l2_loss = self.l2*K.sum(K.square(inputs))
        l1_loss = self.l1*K.sum(K.abs(inputs))
        lm_loss = self.lm*K.sum( K.relu(K.abs(inputs) -self.w_max ))
        return l2_loss + l1_loss + lm_loss

    def get_config(self):
        return {"l2": self.l2,
                "l1": self.l1,
                "lm": self.lm,
                "L_W": self.L_W}
        
        
class papr_reg(regularizers.Regularizer):
    def __init__(self, lp=1e-2):
        self.lp = lp

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, inputs):
        lp_loss = self.lp * K.sum( tf.math.divide(K.max(K.square(inputs), axis=(0,1,2)),  K.mean(K.abs(inputs), axis=(0,1,2)) ))
        return lp_loss

    def get_config(self):
        return {"lp": self.lp}
    
"""
Regularisierung fuer die Quantisierung:
    - Abstand von Quantisierungsstufen fuer x < 4*delta
    - l1-Norm
    - Max-Norm: Gewichte nicht groesser als durch Q.-Schema vorgegeben
"""
class discretize_reg(regularizers.Regularizer):
    def __init__(self, FL=[2,6], lg=1e-5, l2=1e-5):
        self.FL = FL
        self.lg = lg 
        self.l2 = l2
        
    def set_layer(self, layer):
        self.layer = layer
        
    def __call__(self, inputs):    
#        c_weights =  K.clip(inputs, -4*self.delta, 4*self.delta)
        q_weights =  K.round(inputs / 2**-self.FL[1]) * 2**-self.FL[1]
        lg_loss = self.lg * K.sum(-K.log(K.abs(inputs)+1e-6) * K.abs(inputs-K.stop_gradient(q_weights))) 
        l2_loss = self.l2*K.sum(K.square(inputs))
        lm_loss = 1e2 * K.sum( K.relu(K.abs(inputs) - (2**(self.FL[0]-1) - 2**(-self.FL[1]-1) ) ))
        return lg_loss + l2_loss + lm_loss
    
    def get_config(self):
        return {"FL": self.FL,
                "lg": self.lg,
                "l2": self.l2}
