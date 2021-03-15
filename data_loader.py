import nibabel as nib
import numpy as np
import scipy.io as sio
import os

max_array = [5000., 500., 1.]

def load_data(exp_dir, subj_lst, test=False): 
    
    if test:
        subj = subj_lst
        print('----> Testing')
        mask = np.ones([200,200,200,1])
        qmap_mot = np.zeros([200,200,200,3])
        qmap_nocorr = np.zeros([200,200,200,3])

        # load qmap_mot (motion data after navigator-based correction)
        mot_dir = os.path.join(exp_dir, 'motion_recon')

        try: 
            mot_file = [f for f in os.listdir(mot_dir) if subj in f]
            
            data = sio.loadmat(os.path.join(mot_dir, mot_file[0]))   
            qmap_mot[:,:,:,0:2] = data['out']['qmap'][0,0]*1e3 #T1, T2 in ms
            qmap_mot[:,:,:,2] = np.abs(data['out']['pd'][0,0])/np.max(np.abs(data['out']['pd'][0,0]))
                
        except: 
            print('No qmap_mot file found! Giving back zeros')

        # load qmap_nocorr (motion data without any correction)
        nocorr_dir =  os.path.join(exp_dir, 'nocorr_recon')

        try: 
            nocorr_file = [f for f in os.listdir(nocorr_dir) if subj in f]
            
            data = sio.loadmat(os.path.join(nocorr_dir, nocorr_file[0]))   
            qmap_nocorr[:,:,:,0:2] = data['out']['qmap'][0,0]*1e3 #T1, T2 in ms
            qmap_nocorr[:,:,:,2] = np.abs(data['out']['pd'][0,0])/np.max(np.abs(data['out']['pd'][0,0]))
                
        except: 
            print('No qmap_nocorr file found! Giving back zeros')

        return qmap_mot, qmap_nocorr, mask
    
    else: # training
        print('----> Training')
        qmap_mot = np.zeros([200,200,200,len(subj_lst),3])
        qmap_hq = np.zeros([200,200,200,len(subj_lst),3])
        qmap_res = np.zeros([200,200,200,len(subj_lst),3])
        mask = np.ones([200,200,200,len(subj_lst),1])
        for s,subj in enumerate(subj_lst):
            print(subj)

            # load qmap_mot (simulated motion data after navigator-based correction)
            mot_dir = os.path.join(exp_dir, 'simulated_motion_recon')

            try: 
                mot_file = [f for f in os.listdir(mot_dir) if subj in f]
                                
                data = sio.loadmat(os.path.join(mot_dir, mot_file[0]))   
                qmap_mot[:,:,:,s,0:2] = data['out']['qmap'][0,0]*1e3 #T1, T2 in ms
                qmap_mot[:,:,:,s,2] = np.abs(data['out']['pd'][0,0])/np.max(np.abs(data['out']['pd'][0,0]))
                    
            except: 
                print('No qmap_mot file found! Giving back zeros')

            # load qmap_hq (high quality, motion-free data)
            hq_dir =  os.path.join(exp_dir, 'hq_recon')

            try: 
                hq_file = [f for f in os.listdir(hq_dir) if subj in f]
                
                data = sio.loadmat(os.path.join(hq_dir, hq_file[0]))   
                qmap_hq[:,:,:,s,0:2] = data['out']['qmap'][0,0]*1e3 #T1, T2 in ms
                qmap_hq[:,:,:,s,2] = np.abs(data['out']['pd'][0,0])/np.max(np.abs(data['out']['pd'][0,0]))
                    
            except: 
                print('No qmap_hq file found! Giving back zeros')

            # residuum
            qmap_res = qmap_hq - qmap_mot
        
        return qmap_mot, qmap_res, mask


def train_patches_extract(patch_size, max_patches, qmap_mot, qmap_res, mask, max_array = max_array):
    
    Nx, Ny, Nz, Ns, Nq = qmap_mot.shape
    
    px = np.random.randint(0, Nx-patch_size[0], np.int(max_patches/Ns))
    py = np.random.randint(0, Ny-patch_size[1], np.int(max_patches/Ns))
    pz = np.random.randint(0, Nz-patch_size[2], np.int(max_patches/Ns))
    
    qmap_mot_out = np.zeros([max_patches, patch_size[0], patch_size[1], patch_size[2], 3])
    qmap_res_out = np.zeros([max_patches, patch_size[0], patch_size[1], patch_size[2], 3])
    mask_out = np.zeros([max_patches, patch_size[0], patch_size[1], patch_size[2]])

    i=0
    for s in range(Ns):   
        for p in range(len(px)):
            qmap_mot_out[i,:,:,:,:] = qmap_mot[px[p]:px[p]+patch_size[0], py[p]:py[p]+patch_size[1], pz[p]:pz[p]+patch_size[2], s, :].copy()
            qmap_res_out[i,:,:,:,:] = qmap_res[px[p]:px[p]+patch_size[0], py[p]:py[p]+patch_size[1], pz[p]:pz[p]+patch_size[2], s, :].copy()
            mask_out[i,:,:,:] = mask[px[p]:px[p]+patch_size[0], py[p]:py[p]+patch_size[1], pz[p]:pz[p]+patch_size[2], s, 0].copy()
            i = i+1
        print(i)
    qmap_mot_out = qmap_mot_out[:i, :, :, :, :]
    qmap_res_out = qmap_res_out[:i, :, :, :, :]
    mask_out = mask_out[:i, :, :, :]

    #normalize
    qmap_mot_out = qmap_mot_out / max_array
    qmap_res_out = qmap_res_out / max_array
    
    print(qmap_mot_out.shape)
    return qmap_mot_out, qmap_res_out, mask_out   
    


def test_patches_extract(patch_size, qmap_mot_orig, mask_orig, r, max_array=max_array):
    
    qmap_mot = qmap_mot_orig[r:, r:, r:,:].copy()
    mask = mask_orig[r:, r:, r:].copy()
    
    Nx, Ny, Nz, Nq = qmap_mot.shape
    num_patches_x = int(np.ceil(Nx / patch_size[0]))
    num_patches_y = int(np.ceil(Ny / patch_size[1]))
    num_patches_z = int(np.ceil(Nz / patch_size[2]))
    num_patches = num_patches_x * num_patches_y * num_patches_z
    
    qmap_mot_out = np.zeros([num_patches, patch_size[0], patch_size[1], patch_size[2], 3])
    mask_out = np.zeros([num_patches, patch_size[0], patch_size[1], patch_size[2]])
    
    px = range(0,Nx,patch_size[0])
    py = range(0,Ny,patch_size[1])
    pz = range(0,Nz,patch_size[2])
    
    i = 0
    for z in pz:
        for y in py:
            for x in px:
                try:
                    qmap_mot_out[i,:,:,:,:] = qmap_mot[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2],:].copy()
                    mask_out[i,:,:,:] = mask[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2],0].copy()
                except:
                    try:
                        nxx = qmap_mot[x:, y:y+patch_size[1], z:z+patch_size[2],:].shape[0]
                        qmap_mot_out[i,:nxx,:,:,:] = qmap_mot[x:, y:y+patch_size[1], z:z+patch_size[2],:].copy()
                        mask_out[i,:nxx,:,:] = mask[x:, y:y+patch_size[1], z:z+patch_size[2],0].copy()
                    except:
                        try:
                            nyy = qmap_mot[x:x+patch_size[0], y:, z:z+patch_size[2],:].shape[1]
                            qmap_mot_out[i,:,:nyy,:,:] = qmap_mot[x:x+patch_size[0],y:, z:z+patch_size[2],:].copy()
                            mask_out[i,:,:nyy,:] = mask[x:x+patch_size[0],y:, z:z+patch_size[2],0].copy()
                        
                        except:
                            try:
                                nzz = qmap_mot[x:x+patch_size[0], y:y+patch_size[1],z:,:].shape[2]
                                qmap_mot_out[i,:,:,:nzz,:] = qmap_mot[x:x+patch_size[0],y:y+patch_size[1], z:,:].copy()
                                mask_out[i,:,:,:nzz] = mask[x:x+patch_size[0],y:y+patch_size[1], z:,0].copy()
                            except: 
                                try: 
                                    nxx, nyy = qmap_mot[x:, y:, z:z+patch_size[2],:].shape[0:2]
                                    qmap_mot_out[i,:nxx,:nyy,:,:] = qmap_mot[x:,y:, z:z+patch_size[2],:].copy()
                                    mask_out[i,:nxx,:nyy,:] = mask[x:,y:, z:z+patch_size[2],0].copy()
                                    
                                except: 
                                    try: 
                                        nyy, nzz = qmap_mot[x:x+patch_size[0], y:, z:,:].shape[1:3]
                                        qmap_mot_out[i,:,:nyy,:nzz,:] = qmap_mot[x:x+patch_size[0],y:, z:,:].copy()
                                        mask_out[i,:,:nyy,:nzz] = mask[x:x+patch_size[0],y:, z:,0].copy()
                                        
                                    except:
                                        try: 
                                            nxx, nzz = [qmap_mot[x:, y:y+patch_size[1], z:,:].shape[i] for i in [0,2]]
                                            qmap_mot_out[i,:nxx,:,:nzz,:] = qmap_mot[x:,y:y+patch_size[1], z:,:].copy()
                                            mask_out[i,:nxx,:,:nzz] = mask[x:,y:y+patch_size[1], z:,0].copy()
 
                                        except:
                                            nxx,nyy, nzz = qmap_mot[x:, y:, z:,:].shape[0:3]
                                            qmap_mot_out[i,:nxx,:nyy,:nzz,:] = qmap_mot[x:,y:, z:,:].copy()
                                            mask_out[i,:nxx,:nyy,:nzz] = mask[x:,y:, z:,0].copy()

                i = i +1
               
    #normalize
    qmap_mot_out = qmap_mot_out / max_array
    
    return qmap_mot_out, mask_out
    

def test_patches_assemble(patch_mot, patch_res, patch_mask, r, max_array=max_array):
    
    patch_size = patch_mot.shape[1:4]
    
    Nx = 200-r
    Ny = 200-r
    Nz = 200-r
    
    #normalize
    patch_res = patch_res * max_array 
    patch_mot = patch_mot * max_array
    patch_pred = patch_mot + patch_res
              
    qmap_mot_out = np.zeros([Nx,Ny,Nz,3])
    qmap_res_out = np.zeros([Nx,Ny,Nz,3])
    qmap_pred_out = np.zeros([Nx,Ny,Nz,3])
    mask_out = np.zeros([Nx,Ny,Nz,1])
    
    px = range(0,Nx,patch_size[0])
    py = range(0,Ny,patch_size[1])
    pz = range(0,Nz,patch_size[2])
    
    i = 0
    for z in pz:
        for y in py:
            for x in px:
                try:
                    qmap_mot_out[x:x+patch_size[0], y:y+patch_size[1],z:z+patch_size[2],:] = patch_mot[i,:,:,:,:].copy()
                    qmap_res_out[x:x+patch_size[0], y:y+patch_size[1],z:z+patch_size[2],:] = patch_res[i,:,:,:,:].copy()
                    qmap_pred_out[x:x+patch_size[0], y:y+patch_size[1],z:z+patch_size[2],:] = patch_pred[i,:,:,:,:].copy()
                    mask_out[x:x+patch_size[0], y:y+patch_size[1],z:z+patch_size[2],0] = patch_mask[i,:,:,:].copy()
                except:
                    try:
                        nxx = qmap_mot_out[x:, y:y+patch_size[1], z:z+patch_size[2],:].shape[0]
                        qmap_mot_out[x:, y:y+patch_size[1], z:z+patch_size[2],:] = patch_mot[i,0:nxx,:,:,:].copy()
                        qmap_res_out[x:, y:y+patch_size[1], z:z+patch_size[2],:] = patch_res[i,0:nxx,:,:,:].copy()
                        qmap_pred_out[x:, y:y+patch_size[1], z:z+patch_size[2],:] = patch_pred[i,0:nxx,:,:,:].copy()
                        mask_out[x:, y:y+patch_size[1], z:z+patch_size[2],0] = patch_mask[i,0:nxx,:,:].copy()
                        
                    except:
                        try:
                            nyy = qmap_mot_out[x:x+patch_size[0],y:, z:z+patch_size[2],:].shape[1]
                            qmap_mot_out[x:x+patch_size[0], y:, z:z+patch_size[2],:] = patch_mot[i,:,0:nyy,:,:].copy()
                            qmap_res_out[x:x+patch_size[0], y:, z:z+patch_size[2],:] = patch_res[i,:,0:nyy,:,:].copy()
                            qmap_pred_out[x:x+patch_size[0], y:, z:z+patch_size[2],:] = patch_pred[i,:,0:nyy,:,:].copy()
                            mask_out[x:x+patch_size[0], y:, z:z+patch_size[2],0] = patch_mask[i,:,0:nyy,:].copy()

                        except:
                            try:
                                nzz = qmap_mot_out[x:x+patch_size[0],y:y+patch_size[1], z:,:].shape[2]
                                qmap_mot_out[x:x+patch_size[0], y:y+patch_size[1], z:,:] = patch_mot[i,:,:,0:nzz,:].copy()
                                qmap_res_out[x:x+patch_size[0], y:y+patch_size[1], z:,:] = patch_res[i,:,:,0:nzz,:].copy()
                                qmap_pred_out[x:x+patch_size[0], y:y+patch_size[1], z:,:] = patch_pred[i,:,:,0:nzz,:].copy()
                                mask_out[x:x+patch_size[0], y:y+patch_size[1], z:,0] = patch_mask[i,:,:,0:nzz].copy()

                            except: 
                                try:
                                    nxx,nyy = qmap_mot_out[x:, y:, z:z+patch_size[2],:].shape[0:2]
                                    qmap_mot_out[x:, y:, z:z+patch_size[2],:] = patch_mot[i,0:nxx, 0:nyy,:,:].copy()
                                    qmap_res_out[x:, y:, z:z+patch_size[2],:] = patch_res[i,0:nxx, 0:nyy,:,:].copy()
                                    qmap_pred_out[x:, y:, z:z+patch_size[2],:] = patch_pred[i,0:nxx, 0:nyy,:,:].copy()
                                    mask_out[x:, y:, z:z+patch_size[2],0] = patch_mask[i,0:nxx, 0:nyy,:].copy()   
                                
                                except: 
                                    try: 
                                        nyy, nzz = qmap_mot_out[x:x+patch_size[0], y:, z:,:].shape[1:3]
                                        qmap_mot_out[x:x+patch_size[0], y:, z:,:] = patch_mot[i,:, 0:nyy,0:nzz,:].copy()
                                        qmap_res_out[x:x+patch_size[0], y:, z:,:] = patch_res[i,:, 0:nyy,0:nzz,:].copy()
                                        qmap_pred_out[x:x+patch_size[0], y:, z:,:] = patch_pred[i,:, 0:nyy,0:nzz,:].copy()
                                        mask_out[x:x+patch_size[0], y:, z:,0] = patch_mask[i,:, 0:nyy,0:nzz].copy()   

                                    except:
                                        try: 
                                            nxx, nzz = [qmap_mot_out[x:, y:y+patch_size[1], z:,:].shape[i] for i in [0,2]]
                                            qmap_mot_out[x:, y:y+patch_size[1], z:,:] = patch_mot[i,0:nxx,:,0:nzz,:].copy()
                                            qmap_res_out[x:, y:y+patch_size[1], z:,:] = patch_res[i,0:nxx,:,0:nzz,:].copy()
                                            qmap_pred_out[x:, y:y+patch_size[1], z:,:] = patch_pred[i,0:nxx,:,0:nzz,:].copy()
                                            mask_out[x:, y:y+patch_size[1], z:,0] = patch_mask[i,0:nxx,:,0:nzz].copy()   
  
                                        except:
                                            nxx,nyy, nzz = qmap_mot_out[x:, y:, z:,:].shape[0:3]
                                            qmap_mot_out[x:, y:, z:,:] = patch_mot[i,0:nxx,0:nyy,0:nzz,:].copy()
                                            qmap_res_out[x:, y:, z:,:] = patch_res[i,0:nxx,0:nyy,0:nzz,:].copy()
                                            qmap_pred_out[x:, y:, z:,:] = patch_pred[i,0:nxx,0:nyy,0:nzz,:].copy()
                                            mask_out[x:, y:, z:,0] = patch_mask[i,0:nxx,0:nyy,0:nzz].copy()   
   
                i = i +1
   
    return qmap_pred_out, qmap_mot_out, mask_out   
    

def get_final_qmaps(qmap_mot, qmap_res, mask, max_array=max_array):
    
    out_mot = np.zeros(qmap_mot.shape)
    out_res = np.zeros(qmap_res.shape)
    out_qmap = np.zeros(qmap_mot.shape)

    out_res = qmap_res.copy() * max_array
    
    out_mot = qmap_mot.copy() * max_array
    out_qmap = out_mot.copy() + out_res

    return out_qmap, out_mot, out_res, mask




def permute_data(patch_mot, patch_res, patch_mask):
    
    rand_ind = np.random.permutation(patch_mot.shape[0])
    
    patch_mot = patch_mot[rand_ind,:,:,:,:]
    patch_res = patch_res[rand_ind,:,:,:,:]
    patch_mask = patch_mask[rand_ind,:,:,:]  
    
    return patch_mot, patch_res, patch_mask