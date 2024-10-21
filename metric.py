import tensorflow as tf

def correlation_coefficient(y_true, y_pred):
    x = tf.reshape(y_true, [-1])
    y = tf.reshape(y_pred, [-1])
    mx = tf.reduce_mean(x)
    my = tf.reduce_mean(y)
    xm, ym = x - mx, y - my
    r_num = tf.reduce_sum(tf.multiply(xm, ym))
    r_den = tf.sqrt(tf.multiply(tf.reduce_sum(tf.square(xm)), tf.reduce_sum(tf.square(ym)))) + 1e-7  # Add epsilon
    r = tf.divide(r_num, r_den)
    return r
