import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


class ActivationLayer_relu(Layer):
    def __init__(self,
                 axis=-1,
                 L_A = [3, 5],    # Bitlaenge Laenge Aktivierung (signed)
                 **kwargs):
        super(ActivationLayer_relu, self).__init__(**kwargs)
        self.max_activity = 2**(L_A[0]) - 2**(-L_A[1])
        self.supports_masking = True
        self.axis = axis
        self.L_A = L_A

        

    def build(self, input_shape):   
        super(ActivationLayer_relu, self).build(input_shape)

    def set_LA(self, L_A):
        self.L_A = L_A
        self.max_activity = 2**(L_A[0]) - 2**(-L_A[1]-1)

    def call(self, inputs, training=None):
        def inference_phase():
            outputs = K.clip(inputs, min_value=0, max_value=self.max_activity)
            return K.round(outputs * 2**self.L_A[1]) * 2**-self.L_A[1] # naechster Nachbar

        
        def training_phase():   
            factor = 0.9*self.max_activity
            outputs = K.minimum(inputs, 0.1*inputs+factor)
            outputs = tf.where(outputs<=2**(-self.L_A[1]-1), tf.zeros_like(outputs), outputs)
            return outputs
            
        return K.in_train_phase(training_phase(), inference_phase(), training=training)


    def get_config(self):
        config = {
            'axis': self.axis,
            'L_A': self.L_A
        }
        base_config = super(ActivationLayer_relu, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




class ActivationLayer_signed(Layer):
    def __init__(self,
                 axis=-1,
                 L_A = [3, 5],    # Bitlaenge Laenge Aktivierung (signed)
                 **kwargs):
        super(ActivationLayer_signed, self).__init__(**kwargs)
        self.max_activity = 2**(L_A[0]-1) - 2**(-L_A[1])
        self.supports_masking = True
        self.axis = axis
        self.L_A = L_A

        

    def build(self, input_shape):   
        super(ActivationLayer_signed, self).build(input_shape)

    def set_LA(self, L_A):
        self.L_A = L_A
        self.max_activity = 2**(L_A[0]-1) - 2**(-L_A[1]-1)

    def call(self, inputs, training=None):
        def inference_phase():
            outputs = K.clip(inputs, min_value=-self.max_activity, max_value=self.max_activity)
            return K.round(outputs * 2**self.L_A[1]) * 2**-self.L_A[1] # naechster Nachbar

        
        def training_phase():   
            factor = 0.9*self.max_activity
            outputs = K.minimum(inputs, 0.1*inputs+factor)
            outputs = K.maximum(outputs, 0.1*outputs-factor)
            return outputs
            
        return K.in_train_phase(training_phase(), inference_phase(), training=training)


    def get_config(self):
        config = {
            'axis': self.axis,
            'L_A': self.L_A
        }
        base_config = super(ActivationLayer_signed, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
