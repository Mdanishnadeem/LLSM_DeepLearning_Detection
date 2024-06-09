import tensorflow.keras.backend as K
import tensorflow as tf

def dice_coef(y_true, y_pred, smooth=1.):
    # y_true_f = K.flatten(y_true)
    y_true_f = tf.keras.layers.Flatten()(y_true)
    # y_pred_f = K.flatten(y_pred)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(smooth=1.):
    def dice(y_true, y_pred):
        return -dice_coef(y_true, y_pred, smooth)
    return dice
