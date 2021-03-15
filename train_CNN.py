import os
import tensorflow as tf
import numpy as np
import scipy.io as sio
import shutil
import time

from data_loader import load_data, test_patches_extract, test_patches_assemble, train_patches_extract, permute_data
from models import inference, loss, training, loss_sum


exp_dir = './data'
out_dir = './CNN_training'

subj_lst = ['001', '002', '003', '004']

test_lst = ['001']
val_lst = ['002']
train_lst = list(set(subj_lst)-set(val_lst)-set(test_lst))

# Parameters
train_samples = 100#00
val_samples = 30#00
patch_size = [24,24,24]
Nx,Ny,Nz = patch_size
batch_size = 20
dil_lst = [2,4,8,12]
learning_rate = 1e-4
k = 24
loss_func = 'MAE'
keep_rate = 1.
epochs = 100

# Prepare data
qmap_mot_train, qmap_res_train, mask_train = load_data(exp_dir, train_lst, test=False)
qmap_mot_val, qmap_res_val, mask_val = load_data(exp_dir, val_lst, test=False)

patch_mot_val, patch_res_val, patch_mask_val = train_patches_extract(patch_size, val_samples, qmap_mot_val, qmap_res_val, mask_val)
patch_mot_train, patch_res_train, patch_mask_train = train_patches_extract(patch_size, train_samples, qmap_mot_train, qmap_res_train, mask_train)



val_batch_size = batch_size
steps_per_epoch = patch_mot_train.shape[0] // batch_size
val_steps_per_epoch =  patch_mot_val.shape[0] // val_batch_size

# Save checkpoints
ckpt_dir = os.path.join(out_dir, 'checkpoints')

if os.path.isdir(out_dir):
    shutil.rmtree(out_dir)
    print('Removing old checkpoint dir')
    time.sleep(20)

best_val_loss = 1e10
best_epoch = 0
improve_cut = 20
last_improve = 0

#Create a network instance
with tf.Graph().as_default():

    # Placeholders for input data (shape: [batch size, image size, image size, channels])
    input_pl = tf.placeholder(tf.float32, shape=(None, Nx, Ny, Nz, 3))
    output_pl = tf.placeholder(tf.float32, shape=(None, Nx, Ny, Nz, 3))
    mask_pl = tf.placeholder(tf.float32, shape=(None, Nx, Ny, Nz))

    input_shape = (batch_size, patch_size[0], patch_size[1], patch_size[2], 3)

    # Placeholder for keep_prob in dropout (mode: train:1.0 or test:2.0)
    mode_pl = tf.placeholder(tf.float32)

    # Operations
    prediction_op = inference(input_pl, mode_pl, keep_rate, k, dil_lst, input_shape)
        
    loss_op = loss(output_pl, prediction_op, mask_pl, loss_func)
    train_op = training(loss_op, learning_rate)
    loss_sum_op = loss_sum(loss_op)

    # Create saver for saving variables
    saver = tf.train.Saver(max_to_keep=epochs)
    # Initialize the variables
    init = tf.global_variables_initializer()
    # Create session
    sess = tf.InteractiveSession()
    sess.run(init)

    # Save train + val losses
    train_loss_summary = tf.summary.scalar('training_loss', loss_op) 
    train_loss_writer = tf.summary.FileWriter(os.path.join(out_dir, 'train_loss_summary'), sess.graph)
    val_loss_summary = tf.summary.scalar('validation_loss', loss_op)
    val_loss_writer = tf.summary.FileWriter(os.path.join(out_dir, 'val_loss_summary'), sess.graph)

    # Training
    for ep in range(epochs):
        print('----> epoch', ep)
        total_train_loss = 0
        total_val_loss = 0

        train_loss_list = []
        val_loss_list = []

        train_input, train_res, train_mask = permute_data(patch_mot_train, patch_res_train, patch_mask_train)

        val_input, val_res, val_mask = permute_data(patch_mot_val, patch_res_val, patch_mask_val)

        train_random_idxs = np.random.permutation(train_input.shape[0])
        val_random_idxs = np.random.permutation(val_input.shape[0])
        for step in range(steps_per_epoch):
            # Get batches 
            batch_start_idx = (step % (train_input.shape[0] // batch_size)) * batch_size
            batch_idx = train_random_idxs[batch_start_idx:batch_start_idx+batch_size]
            im_batch = train_input[batch_idx,:,:,:,:]
            out_batch = train_res[batch_idx,:,:,:,:]
            mask_batch = train_mask[batch_idx,:,:,:]
            # Create input dictionary to feed network
            input_dict = {input_pl: im_batch, output_pl: out_batch, mask_pl: mask_batch, mode_pl:1.0}
            # Run operations
            activations, pred, train_loss, train_loss_sum = sess.run([train_op, prediction_op, loss_op, train_loss_summary], feed_dict=input_dict)

            train_loss_writer.add_summary(train_loss_sum, ep*steps_per_epoch + step)
            train_loss_list.append(train_loss)
        # Train loss
        total_train_loss = np.mean(train_loss_list)

        for val_step in range(val_steps_per_epoch):

            # Get batches
            batch_start_idx = (val_step % (patch_mot_val.shape[0] // batch_size)) * batch_size
            val_batch_idx = val_random_idxs[batch_start_idx:batch_start_idx+batch_size]
            val_im_batch = val_input[val_batch_idx,:,:,:,:]
            val_out_batch = val_res[val_batch_idx,:,:,:,:]
            val_mask_batch = val_mask[val_batch_idx,:,:,:]
            # Create input dictionary to feed network
            val_input_dict = {input_pl: val_im_batch, output_pl: val_out_batch, mask_pl: val_mask_batch, mode_pl:2.0}
            # Run operations
            val_pred, val_loss, val_loss_sum = sess.run([prediction_op, loss_op, val_loss_summary], feed_dict=val_input_dict)
            val_loss_list.append(val_loss)

            val_loss_writer.add_summary(val_loss_sum, ep*val_steps_per_epoch + val_step)
        
        # Val loss
        total_val_loss = np.mean(val_loss_list)

        if ep%10==0:
            checkpoint_file = '{}/CNN_model_{}.ckpt'.format(ckpt_dir, str(ep))
            saver.save(sess, checkpoint_file)

        #save checkpoint if val_loss decreases
        if total_val_loss < best_val_loss:
            last_improve = 0
            best_val_loss = total_val_loss.copy()
            print('New best checkpoint - epoch', str(ep))
            
            #delete previously saved best checkpoint
            old_ckpt_files = os.listdir(ckpt_dir)
            for file in old_ckpt_files:
                if file.startswith('CNN_best_model'):
                    if os.path.isfile('{}/{}'.format(ckpt_dir, file)):
                        os.remove('{}/{}'.format(ckpt_dir, file))

            #save new best checkpoint
            checkpoint_file = '{}/CNN_best_model_{}.ckpt'.format(ckpt_dir, str(ep))
            saver.save(sess, checkpoint_file)
            best_epoch = ep

        else:
            last_improve = last_improve + 1
            if last_improve >= improve_cut:
                print('Early stopping')
                break
sess.close()    
