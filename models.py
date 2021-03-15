import tensorflow as tf
conv_filter_size = 3


def conv_block_out(x, in_channels, out_channels):
    with tf.variable_scope('dense_out'):
        conv_dense_out = tf.keras.layers.Dense(out_channels, name='dense_out')(x)
    return conv_dense_out
    
def conv_block_dense(x, in_channels, out_channels, keep_prob):
    with tf.variable_scope('conv_dense'):
        conv_dense = tf.keras.layers.Dense(out_channels, activation='relu', name='dense')(x)
    return conv_dense
    

def conv_block(x, in_channels, out_channels, keep_prob, dil_factor=1):
    stddev = tf.cast(tf.sqrt(tf.divide(3,((conv_filter_size**3 * in_channels)))), tf.float32)
    with tf.variable_scope('conv1'):
        shape = [conv_filter_size, conv_filter_size, conv_filter_size, in_channels, out_channels]
        w1 = tf.Variable(tf.truncated_normal(shape, stddev=stddev, name='weights'))
        b = tf.Variable(tf.zeros([out_channels]), name = 'bias')
        conv_n = tf.keras.layers.Conv3D(out_channels, conv_filter_size, strides=(1, 1, 1), dilation_rate=(dil_factor, dil_factor, dil_factor), padding='same', activation='relu')(x)
        conv_n_dropped = tf.nn.dropout(conv_n, keep_prob)
    return conv_n_dropped


def conv_block_in(x, in_channels, out_channels, keep_prob, input_shape):
    print(input_shape)
    with tf.variable_scope('conv_in'):
        conv_in = tf.keras.layers.Conv3D(out_channels, conv_filter_size, activation='relu', padding='same', input_shape=input_shape)(x)
        conv_in_dropped = tf.nn.dropout(conv_in, keep_prob)
    return conv_in_dropped


def inference(image_batch, mode, keep, k, dil_lst, input_shape):
    thresh = tf.Variable(2.0, dtype=tf.float32)
    
    f1 = lambda: keep
    f2 = lambda: tf.constant(1.0)
    keep_prob = tf.case([(tf.less(mode, thresh), f1)], default=f2)
    
    with tf.variable_scope('input_layer'): 
        conv_in = conv_block_in(image_batch, 3, k, keep_prob, input_shape)
        
    with tf.variable_scope('global_path'):
        conv_glob_1 = conv_block(conv_in, k, 2*k, keep_prob)
        conv_glob_2 = conv_block(conv_glob_1, 2*k, 3*k, keep_prob)
        conv_glob_3 = conv_block(conv_glob_2, 3*k, 4*k, keep_prob)
        conv_glob_4 = conv_block(conv_glob_3, 4*k, 4*k, keep_prob)
               
    with tf.variable_scope('local_path'):
        
        conv_loc_1 = conv_block(conv_in, k, 2*k, keep_prob, dil_lst[0])
        conv_loc_2 = conv_block(conv_loc_1, 2*k, 3*k, keep_prob, dil_lst[1])
        conv_loc_3 = conv_block(conv_loc_2, 3*k, 4*k, keep_prob, dil_lst[2])    
        conv_loc_4 = conv_block(conv_loc_3, 4*k, 4*k, keep_prob, dil_lst[3])
    
    with tf.variable_scope('concat'):
        concat = tf.concat([conv_glob_4,conv_loc_4], 4, name='concat')     
        
    with tf.variable_scope('fc_layer1'):
        conv_dense_1 = conv_block_dense(concat, 8*k, 256, keep_prob)
        
    with tf.variable_scope('fc_layer2'):
        conv_dense_2 = conv_block_dense(conv_dense_1, 256, 128, keep_prob)
        
    with tf.variable_scope('fc_layer3'):
        conv_dense_3 = conv_block_dense(conv_dense_2, 128, 3, keep_prob)

    with tf.variable_scope('out'):
        out = conv_block_out(conv_dense_3, 3, 3)

    return out

def loss(reference, prediction, mask, cost='MAE'):
    
    if cost == 'MSE':
        loss = tf.losses.mean_squared_error(reference, prediction)
    else:
        loss = tf.losses.absolute_difference(reference, prediction)
    
    loss = tf.reduce_mean(loss)
    return loss

def loss_sum(loss):
    loss_sum = tf.summary.scalar('loss', loss)
    return loss_sum
    
def training(loss, learning_rate):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step)
    return train_op