import tensorflow.keras.backend as K


def combined_membership_loss(y_true, y_pred):
    loss_m = K.mean( K.binary_crossentropy(y_true, K.sigmoid(y_pred)))
    loss_c = K.categorical_crossentropy(y_true, y_pred, from_logits=True)
    
    return loss_m+loss_c

def cat_crossentropy_from_logits(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)
    