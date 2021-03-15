import os
import tensorflow as tf
import numpy as np
import scipy.io as sio

from data_loader import load_data, test_patches_extract, test_patches_assemble
from models import inference

#data folder
exp_dir = './data'
#folder where to store CNN-corrected data (creates a subfolder with corrected .mat files)
out_dir = './data/CNN_application'
#folder with checkpoints
project_dir = './CNN_training'

subj = '001'

#CNN specs
train_samples = 10000
val_samples = 3000
patch_size = [24,24,24]
batch_size = 20
dil_lst = [2,4,8,12]
learning_rate = 1e-4
keep_rate = 1.
k = 24
loss_func = 'MAE'

Nx,Ny,Nz = patch_size
reps = 6 
epoch = 'best'  
batch_size_calc = 20


total_qmap_pred_out = np.zeros([200,200,200, 3]) #DL-corrected maps
total_qmap_mot_out = np.zeros([200,200,200, 3]) #Navigator-corrected maps
total_qmap_nocorr_out = np.zeros([200,200,200, 3]) #Non-corrected maps
total_mask_out = np.zeros([200,200,200, 1])

# Load test data
qmap_mot_test, qmap_nocorr_test, mask_test = load_data(exp_dir, subj, test=True)

# Apply CNN
ckpt_dir = os.path.join(project_dir, 'checkpoints')
ckpt_list = os.listdir(ckpt_dir)

######################### run inference

if epoch=='best':
    for file in ckpt_list:
        if 'CNN_best_model' in file:
            eval_ckpt_file = file.split('.ckpt')[0] + '.ckpt'

else:

    eval_ckpt_file = 'CNN_model_{}.ckpt'.format(str(epoch))
    
    
qmap_pred_out = np.zeros([200,200,200,3,reps])
qmap_mot_out = np.zeros([200,200,200,3,reps])
mask_out = np.zeros([200,200,200,1,reps])

# Correct motion... go go go
tf.reset_default_graph()

# Placeholders for input data (shape: [batch size, image size, image size, channels])
input_pl = tf.placeholder(tf.float32, shape=(None, Nx,Ny,Nz,3))
output_pl = tf.placeholder(tf.float32, shape=(None, Nx,Ny,Nz,3))
mask_pl = tf.placeholder(tf.float32, shape=(None, Nx, Ny,Nz))

# Placeholder for keep_prob in dropout (mode: train:1.0 or test:2.0)
mode_pl = tf.placeholder(tf.float32)

# Operations
input_shape = (batch_size, patch_size[0], patch_size[1], patch_size[2], 3)
prediction_op = inference(input_pl, mode_pl, keep_rate, k, dil_lst, input_shape)              
saver = tf.train.Saver()

stepsize = int(Nx/reps)
for rep in range(reps):
    
    r = rep*stepsize
    print('--->', r)
    patch_mot_test, patch_mask_test = test_patches_extract(patch_size, qmap_mot_test, mask_test, r)
    patch_pred_test = np.zeros_like(patch_mot_test)
    test_samples, x_patch, y_patch, z_patch, q_patch = patch_pred_test.shape
    test_steps = test_samples // batch_size_calc

    #Create a CNN instance
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(ckpt_dir, eval_ckpt_file))
        for test_step in range(test_steps):
            # Get batches of image and mask slices
            batch_start_idx = (test_step % test_samples) * batch_size# For 3 input channels
            test_im_batch = patch_mot_test[batch_start_idx:batch_start_idx+batch_size,:,:,:,:]
            test_mask_batch = patch_mask_test[batch_start_idx:batch_start_idx+batch_size,:,:,:]
            # Create input dictionary to feed network
            test_input_dict = {input_pl: test_im_batch, mask_pl: test_mask_batch, mode_pl:2.0}
            # Run operations
            patch_pred_test[batch_start_idx:batch_start_idx+batch_size,:,:,:,:] = sess.run(prediction_op, feed_dict=test_input_dict)

    qmap_pred_out[r:,r:,r:,:,rep], qmap_mot_out[r:,r:,r:,:,rep], mask_out[r:,r:,r:,:,rep] = test_patches_assemble(patch_mot_test, patch_pred_test, patch_mask_test, r)


qmap_pred_out = np.mean(qmap_pred_out, axis=4)
qmap_pred_out[qmap_pred_out<0] = 0
total_qmap_pred_out = qmap_pred_out.copy()
total_qmap_mot_out = qmap_mot_out[:,:,:,:,0].copy()
total_mask_out = mask_out[:,:,:,:,0].copy()
total_qmap_nocorr_out  = qmap_nocorr_test.copy()
out = {'qmap': qmap_pred_out[:,:,:,0:2]/1e3, 'pd':qmap_pred_out[:,:,:,2]}

# save corrected qmaps
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
out_path = os.path.join(out_dir, 'subj' + subj + '_CNNcorr.mat')
sio.savemat(out_path, {'out':out})

###############################

