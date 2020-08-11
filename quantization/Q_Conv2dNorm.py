import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import constraints, initializers, regularizers
from tensorflow.keras.layers import InputSpec, Layer


class Conv2dNorm(Layer):
    def __init__(self,
                 filters,
                 kernel_size = (3,3),
                 strides=1,
                 padding='same',
                 dilation_rate=1,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 bias_initializer='zeros',
                 kernel_constraint = None,
                 bias_constraint = None,
                 momentum=0.99,     # 0.98
                 puffer = 1.0, # 0.95  # Aktivierung darf xx % des Maximalen Wertes nicht ueberschreiten 
                 L_A = [3, 5],    # Integer Laenge Aktivierung (unsigned)
                 L_W = [1, 7],    # Integer Laenge Aktivierung (unsigned)
                 max_scale = 8,
                 **kwargs):
        super(Conv2dNorm, self).__init__(**kwargs)
        self.filters            = filters
        self.kernel_size        = kernel_size
        self.strides            = strides
        self.padding            = padding
        self.data_format        = 'channels_last'
        self.dilation_rate      = dilation_rate
        self.use_bias           = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer   = initializers.get(bias_initializer)  
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.bias_regularizer   = regularizers.get(bias_regularizer)
        self.kernel_constraint  = constraints.get(kernel_constraint)
        self.bias_constraint   = constraints.get(bias_constraint)
        self.L_W = L_W
        self.L_A = L_A
        self.momentum = momentum
        self.max_weight = 2**(L_W[0]-1) * puffer
        self.max_activity = 2**L_A[0] - 2**-L_A[1]
        self.max_activity_x = self.max_activity * puffer
        self.w_scale_initializer = initializers.Constant(value=1.)
        self.max_scale = max_scale
        self.input_spec = InputSpec(ndim=4)
        # self.factor = 0.9*self.max_activity
 
    
    def build(self, input_shape):
        self.channel_axis = -1
        input_dim = input_shape[self.channel_axis ]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      name='kernel')
        if self.use_bias:   
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        name='bias')
        else:
            self.bias = None
        self.w_scale = self.add_weight(shape=(self.filters,),
                                    name='w_scale',
                                    initializer=self.w_scale_initializer,
                                    trainable=False)   
  
        
        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={self.channel_axis: input_dim})
        self.built = True




    def call(self, inputs, training=None): 
        def normalize_inference():
            convs = K.conv2d(
                inputs,
                self.kernel * self.w_scale,
                strides=self.strides,
                padding=self.padding,
                data_format='channels_last',
                dilation_rate=self.dilation_rate)

            if self.use_bias:
                if self.data_format == 'channels_last':
                    convs = K.bias_add(
                        convs,
                        self.bias * self.w_scale,
                        data_format='channels_last')

            outputs = K.clip(convs, min_value=0, max_value=self.max_activity)
            return K.round(outputs * 2**self.L_A[1]) * 2**-self.L_A[1] # naechster Nachbar

        
        def training_phase():   
            convs = K.conv2d(
                inputs,
                self.kernel,
                data_format='channels_last',
                strides=self.strides,
                padding=self.padding,
                dilation_rate=self.dilation_rate)

            if self.use_bias:
                if self.data_format == 'channels_last':
                    convs = K.bias_add(
                        convs,
                        self.bias,
                        data_format='channels_last')
                scale2 =  self.max_weight / (K.maximum(tf.abs(self.bias), tf.reduce_max(tf.abs(self.kernel), axis=(0,1,2)))+1e-6)
            else:
                scale2 =  self.max_weight / (tf.reduce_max(tf.abs(self.kernel), axis=(0,1,2))+1e-6)

            indizes = K.greater(K.max(convs, axis=(0,1,2)), 0.01)
            scale1 = self.w_scale * tf.cast(~indizes, tf.float32) + tf.cast(indizes, tf.float32) * K.abs(self.max_activity_x / (K.max(convs, axis=(0,1,2))+1e-6))
                 
            scale  = K.minimum(K.minimum(scale1, scale2), self.max_scale)
            

            self.add_update(K.moving_average_update(self.w_scale, scale, self.momentum))

            outputs = convs * self.w_scale
            if self.data_format == 'channels_last':
                outputs = tf.transpose(outputs, [0, 3, 1, 2])
                outputs = tf.where(outputs<=2**(-self.L_A[1]-1), tf.zeros_like(outputs), outputs)
                outputs = tf.transpose(outputs, [0, 2, 3, 1])
            else:
                outputs = tf.where(outputs<=2**(-self.L_A[1]-1), tf.zeros_like(outputs), outputs)
            outputs = K.minimum(outputs, self.max_activity) #0.1*outputs+self.factor)
            return outputs
            
        return K.in_train_phase(training_phase(), normalize_inference(), training=training)


    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
        return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)


    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),           
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'L_W': self.L_W,
            'L_A': self.L_A,
            'momentum': self.momentum,
            'max_scale': self.max_scale
            }
        base_config = super(Conv2dNorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
    
    
    
    
"""
_______________________________________________________________________________
SEP-CONV
_______________________________________________________________________________
"""
    
    
    
class SepConv2DNorm(Layer):
    def __init__(self,
                  filters,
                  kernel_size = (3,3),
                  strides=1,
                  padding='same',
                  dilation_rate=(1, 1),
                  kernel_initializer='glorot_uniform',
                  kernel_regularizer=None,
                  depthwise_initializer='glorot_uniform',
                  depthwise_regularizer=None,
                  depthwise_constraint = None,
                  bias_regularizer=None,
                  activity_regularizer=None,
                  bias_initializer='zeros',
                  kernel_constraint = None,
                  bias_constraint = None,
                  momentum=0.99,     # 0.98
                  puffer = 0.95, # 0.95  # Aktivierung darf xx % des Maximalen Wertes nicht ueberschreiten 
                  L_A = [1, 7],    # Integer Laenge Aktivierung (unsigned)
                  L_W = [3, 5],    # Integer Laenge Aktivierung (unsigned)
                  max_scale = 6,
                  **kwargs):
        super(SepConv2DNorm, self).__init__(**kwargs)
        self.filters                = filters
        self.kernel_size            = kernel_size
        self.strides                = strides
        self.padding                = padding
        self.data_format            = 'channels_last'
        self.dilation_rate          =  dilation_rate
        self.depth_multiplier       = 1    
        self.kernel_initializer     = initializers.get(kernel_initializer)
        self.bias_initializer       = initializers.get(bias_initializer)  #// siehe unten weight*scale wird reg.
        self.depthwise_initializer  = initializers.get(depthwise_initializer)
        self.kernel_regularizer     = regularizers.get(kernel_regularizer)   
        self.bias_regularizer       = regularizers.get(bias_regularizer)
        self.depthwise_regularizer  = regularizers.get(depthwise_regularizer)
        self.kernel_constraint      = constraints.get(kernel_constraint)
        self.bias_constraint        = constraints.get(bias_constraint)
        self.depthwise_constraint   = constraints.get(depthwise_constraint)
        self.activity_regularizer   = regularizers.get(activity_regularizer)
        self.L_W = L_W
        self.L_A = L_A
        self.momentum = momentum
        self.max_weight = 2**(L_W[0]-1) * puffer
        self.max_activity = 2**L_A[0] - 2**-L_A[1]
        self.max_activity_signed = 2**(L_A[0]-1) - 2**-L_A[1]
        self.max_activity_x = self.max_activity_signed * puffer
        self.w_scale_initializer = initializers.Constant(value=1.)
        self.max_scale = max_scale
        self.input_spec = InputSpec(ndim=4)
    
    
    
    def build(self, input_shape):
        self.channel_axis = -1
        if input_shape[self.channel_axis] is None:
          raise ValueError('The channel dimension of the inputs to '
                            '`DepthwiseConv2D` '
                            'should be defined. Found `None`.')
        input_dim = int(input_shape[self.channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
        kernel_shape = (1, 1, input_dim, self.filters)
        
        self.depthwise_kernel = self.add_weight(shape=depthwise_kernel_shape,
                                initializer=self.depthwise_initializer,
                                name='depthwise_kernel',
                                regularizer=self.depthwise_regularizer,
                                constraint=self.depthwise_constraint)
         
        self.kernel = self.add_weight(shape=kernel_shape,
                              initializer=self.kernel_initializer,
                              regularizer=self.kernel_regularizer,
                              constraint=self.kernel_constraint,
                              name='kernel')   
        
        self.bias = self.add_weight(shape=(self.filters,),
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    name='bias')
       
        self.w_scale = self.add_weight(shape=(self.filters,),
                            name='w_scale',
                            initializer=self.w_scale_initializer,
                            trainable=False)   
          # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={self.channel_axis: input_dim})
        self.built = True



    def call(self, inputs, training=None):
        def normalize_inference():
            dconvs = K.depthwise_conv2d(
                inputs,
                self.depthwise_kernel,
                strides=self.strides,
                padding=self.padding,
                data_format='channels_last')
#            dconvs = K.clip(dconvs, min_value=0, max_value=self.max_activity)
            dconvs = K.clip(dconvs, min_value=-self.max_activity_signed, max_value=self.max_activity_signed)
            
            convs = K.conv2d(
                dconvs,
                self.kernel * self.w_scale,
                strides=(1,1),
                padding=self.padding,
                data_format='channels_last',
                dilation_rate=self.dilation_rate)
            convs = K.bias_add(
                convs,
                self.bias * self.w_scale,
                data_format='channels_last')
        
#            outputs = convs
#            outputs = K.clip(convs, min_value=-self.max_activity_signed, max_value=self.max_activity_signed)
            outputs = K.clip(convs, min_value=0, max_value=self.max_activity)
            return K.round(outputs * 2**self.L_A[1]) * 2**-self.L_A[1] # naechster Nachbar
        
        def training_phase(): 
            # Depthwise-Conv mit Soft-Relu
            dconvs = K.depthwise_conv2d(
                        inputs,
                        self.depthwise_kernel,
                        strides=self.strides,
                        padding=self.padding,
                        data_format='channels_last')
            
#            dconvs = tf.where(dconvs<=2**(-self.L_A[1]-1), tf.zeros_like(dconvs), dconvs)
#            factor2 = 0.9*self.max_activity
#            dconvs = K.minimum(dconvs, 0.1*dconvs+factor2)
            factor2 = 0.9*self.max_activity_signed
            dconvs = K.minimum(dconvs, 0.1*dconvs+factor2)
            dconvs = K.maximum(dconvs, 0.1*dconvs-factor2)
            
            # Pointwise-Conv 
            convs = K.conv2d(
                        dconvs,
                        self.kernel,
                        strides=(1,1),
                        padding=self.padding,
                        data_format='channels_last',
                        dilation_rate=self.dilation_rate)
            convs = K.bias_add(
                        convs,
                        self.bias, 
                        data_format='channels_last')
            
            # Skalierung
            scale1 = K.abs(self.max_activity_x / (K.max(K.abs(convs), axis=(0,1,2))+1e-6))
            indizes = K.greater(scale1, self.max_scale)
            scale1 = self.w_scale * tf.to_float(indizes) + tf.to_float(~indizes) * scale1
                 
            scale2 =  self.max_weight / (K.maximum(tf.abs(self.bias), tf.reduce_max(tf.abs(self.kernel), axis=(0,1,2)))+1e-6)
            scale  = K.minimum(scale1, scale2)

            self.add_update(K.moving_average_update(self.w_scale, scale, self.momentum), inputs)
            # Softclipped-linear
            outputs = convs * self.w_scale
#            outputs = K.clip(outputs, min_value=-self.max_activity_signed, max_value=self.max_activity_signed)
            outputs = tf.where(outputs<=2**(-self.L_A[1]-1), tf.zeros_like(outputs), outputs)
            outputs = K.minimum(outputs, 0.1*outputs+factor2)
            return outputs
            
        return K.in_train_phase(training_phase(), normalize_inference(), training=training)
        

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
        return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)
        

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'depthwise_initializer': initializers.serialize(self.depthwise_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),    
            'depthwise_regularizer': regularizers.serialize(self.depthwise_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'depthwise_constraint': constraints.serialize(self.depthwise_constraint),
            'L_W': self.L_W,
            'L_A': self.L_A,
            'momentum': self.momentum,
            'max_scale': self.max_scale
            }
        base_config = super(SepConv2DNorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
  
    
        
    
# """
# _______________________________________________________________________________
# RES-CONV
# _______________________________________________________________________________
# """
    

# class Res2dNorm(Layer):
#     def __init__(self,
#                  kernel_size = (3,3),
#                  strides=1,
#                  padding='same',
#                  dilation_rate=1,
#                  kernel_initializer='glorot_uniform',
#                  kernel_regularizer=None,
#                  bias_regularizer=None,
#                  activity_regularizer=None,
#                  bias_initializer='zeros',
#                  kernel_constraint = None,
#                  bias_constraint = None,
#                  momentum=0.99,     # 0.98
#                  puffer = 0.95, # 0.95  # Aktivierung darf xx % des Maximalen Wertes nicht ueberschreiten 
#                  L_A = [3, 5],    # Integer Laenge Aktivierung (unsigned)
#                  L_W = [1, 7],    # Integer Laenge Aktivierung (unsigned)
#                  max_scale = 6,
#                  **kwargs):
#         super(Res2dNorm, self).__init__(**kwargs)
#         self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
#         self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
#         self.padding = conv_utils.normalize_padding(padding)
#         self.data_format='channels_last'
#         self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')
#         self.kernel_initializer = initializers.get(kernel_initializer)
#         self.bias_initializer = initializers.get(bias_initializer)  #// siehe unten weight*scale wird reg.
#         self.kernel_regularizer = regularizers.get(kernel_regularizer)
#         self.activity_regularizer = activity_regularizer
#         self.bias_regularizer = regularizers.get(bias_regularizer)
#         self.kernel_constraint = constraints.get(kernel_constraint)
#         self.bias_constraint = constraints.get(bias_constraint)
#         self.L_W = L_W
#         self.L_A = L_A
#         self.momentum = momentum
#         self.max_weight = 2**(L_W[0]-1) * puffer
#         self.max_activity = 2**L_A[0] - 2**-L_A[1]
#         self.max_activity_x = self.max_activity * puffer
#         self.w_scale_initializer = initializers.Constant(value=1.)
#         self.max_scale = max_scale
#         self.input_spec = InputSpec(ndim=4)
#         self.factor = 0.9*self.max_activity
 
    
#     def build(self, input_shape):
#         self.channel_axis = -1
#         input_dim = input_shape[self.channel_axis ]
#         kernel_shape = self.kernel_size + (input_dim, input_dim)

#         self.kernel = self.add_weight(shape=kernel_shape,
#                                       initializer=self.kernel_initializer,
#                                       regularizer=self.kernel_regularizer,
#                                       name='kernel')   
#         self.bias = self.add_weight(shape=(input_dim,),
#                                     initializer=self.bias_initializer,
#                                     regularizer=self.bias_regularizer,
#                                     name='bias')
#         self.w_scale = self.add_weight(shape=(input_dim,),
#                                     name='w_scale',
#                                     initializer=self.w_scale_initializer,
#                                     trainable=False)   
  
        
#         # Set input spec.
#         self.input_spec = InputSpec(ndim=4, axes={self.channel_axis: input_dim})
#         self.built = True




#     def call(self, inputs, training=None): 
#         def normalize_inference():
#             convs = K.conv2d(
#                 inputs,
#                 self.kernel * self.w_scale,
#                 strides=self.strides,
#                 padding=self.padding,
#                 data_format='channels_last',
#                 dilation_rate=self.dilation_rate)
#             convs = K.bias_add(
#                 convs,
#                 self.bias * self.w_scale,
#                 data_format='channels_last')
#             convs = K.relu(convs)
#             outputs = K.clip(convs+0.5*inputs, min_value=0, max_value=self.max_activity)
#             return K.round(outputs * 2**self.L_A[1]) * 2**-self.L_A[1] # naechster Nachbar

        
#         def training_phase():   
#             convs = K.conv2d(
#                 inputs,
#                 self.kernel,
#                 data_format='channels_last',
#                 strides=self.strides,
#                 padding=self.padding,
#                 dilation_rate=self.dilation_rate)
#             convs = K.bias_add(
#                 convs,
#                 self.bias,
#                 data_format='channels_last')
#             convs = K.relu(convs)
            
#             max_activity_tmp = self.max_activity_x - 0.5 * K.max(inputs, axis=(0,1,2))
            

#             scale1 = K.abs(max_activity_tmp / (K.max(convs, axis=(0,1,2))+1e-6))
#             indizes = K.greater(scale1, self.max_scale)
#             scale1 = self.w_scale * tf.to_float(indizes) + tf.to_float(~indizes) * scale1
                 
#             scale2 =  self.max_weight / (K.maximum(tf.abs(self.bias), tf.reduce_max(tf.abs(self.kernel), axis=(0,1,2)))+1e-6)
#             scale  = K.minimum(scale1, scale2)
            

#             self.add_update(K.moving_average_update(self.w_scale, scale, self.momentum), inputs)

#             outputs = convs * self.w_scale + 0.5 * inputs
#             outputs = tf.where(outputs<=2**(-self.L_A[1]-1), tf.zeros_like(outputs), outputs)
#             outputs = K.minimum(outputs, self.max_activity) #0.1*outputs+self.factor)
#             return outputs
            
#         return K.in_train_phase(training_phase(), normalize_inference(), training=training)


#     def compute_output_shape(self, input_shape):
#         if self.data_format == 'channels_last':
#             space = input_shape[1:-1]
#             new_space = []
#             for i in range(len(space)):
#                 new_dim = conv_utils.conv_output_length(
#                     space[i],
#                     self.kernel_size[i],
#                     padding=self.padding,
#                     stride=self.strides[i],
#                     dilation=self.dilation_rate[i])
#                 new_space.append(new_dim)
#         return (input_shape[0],) + tuple(new_space) + (input_shape[-1],)
#         if self.data_format == 'channels_first':
#             space = input_shape[2:]
#             new_space = []
#             for i in range(len(space)):
#                 new_dim = conv_utils.conv_output_length(
#                     space[i],
#                     self.kernel_size[i],
#                     padding=self.padding,
#                     stride=self.strides[i],
#                     dilation=self.dilation_rate[i])
#                 new_space.append(new_dim)
#             return (input_shape[0], input_shape[-1]) + tuple(new_space)


#     def get_config(self):
#         config = {
#             'kernel_size': self.kernel_size,
#             'strides': self.strides,
#             'padding': self.padding,
#             'dilation_rate': self.dilation_rate,
#             'kernel_initializer': initializers.serialize(self.kernel_initializer),
#             'bias_initializer': initializers.serialize(self.bias_initializer),
#             'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
#             'bias_regularizer': regularizers.serialize(self.bias_regularizer),           
#             'kernel_constraint': constraints.serialize(self.kernel_constraint),
#             'bias_constraint': constraints.serialize(self.bias_constraint),
#             'L_W': self.L_W,
#             'L_A': self.L_A,
#             'momentum': self.momentum,
#             'max_scale': self.max_scale
#             }
#         base_config = super(Res2dNorm, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
