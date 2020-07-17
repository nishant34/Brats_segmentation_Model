import os
import nibabel as nib
import numpy as np
from medpy.io import load
import h5py
import torch

data_path_1 = '/BraTS/Dataset/MICCAI_BraTS_2018_Data_Training/LGG/'
data_path_2 = 'BraTS/Dataset/MICCAI_BraTS_2018_Data_Training/HGG/'
target_path = '/training_brats'  

#def convert_label(label_img):
    #label_processed=np.zeros(label_img.shape[0:]).astype(np.uint8)
    #for i in range(label_img.shape[2]):
        #label_slice=label_img[:, :, i]
        #label_slice[label_slice == 4] = 3
        
        #label_processed[:, :, i]=label_slice
    #return label_processed

def cut_edge(data, keep_margin):
    '''
    function that cuts zero edge
    '''
    D, H, W = data.shape
    D_s, D_e = 0, D - 1
    H_s, H_e = 0, H - 1
    W_s, W_e = 0, W - 1

    while D_s < D:
        if data[D_s].sum() != 0:
            break
        D_s += 1
    while D_e > D_s:
        if data[D_e].sum() != 0:
            break
        D_e -= 1
    while H_s < H:
        if data[:, H_s].sum() != 0:
            break
        H_s += 1
    while H_e > H_s:
        if data[:, H_e].sum() != 0:
            break
        H_e -= 1
    while W_s < W:
        if data[:, :, W_s].sum() != 0:
            break
        W_s += 1
    while W_e > W_s:
        if data[:, :, W_e].sum() != 0:
            break
        W_e -= 1

    if keep_margin != 0:
        D_s = max(0, D_s - keep_margin)
        D_e = min(D - 1, D_e + keep_margin)
        H_s = max(0, H_s - keep_margin)
        H_e = min(H - 1, H_e + keep_margin)
        W_s = max(0, W_s - keep_margin)
        W_e = min(W - 1, W_e + keep_margin)

    return int(D_s), int(D_e), int(H_s), int(H_e), int(W_s), int(W_e)

def build_h5_dataset(data_path_1,data_path_2,target_path):
 list_input_1 = np.array(os.listdir(data_path_1))
 list_input_2 = np.array(os.listdir(data_path_2))
 list_input = list_input_1 + list_input_2   
 i= 0
 for filename in list_input_1:
     data_path_11 = data_path_1 + filename + '/'
     #data_path_21 = data_path_2 + filename + '/'
     #list_input_11 = np.array(os.listdir(data_path_11))
     #list_input_21 = np.array(os.listdir(data_path_21))
     input_path_t1 = filename + '_t1.nii.gz'
     input_path_t1ce = filename + '_t1ce.nii.gz'
     input_path_t2 = filename + '_t2.nii.gz'
     input_path_flair = filename + '_flair.nii.gz'
     input_path_seg = filename + '_seg.nii.gz'

     f_t1 = os.path.join(data_path_11,input_path_t1)
     img_t1,header_t1 = load(f_t1)

     f_t1ce = os.path.join(data_path_11,input_path_t1ce)
     img_t1ce,header_t1ce = load(f_t1ce)

     f_t2 = os.path.join(data_path_11,input_path_t2)
     img_t2,header_t2 = load(f_t2)

     f_flair = os.path.join(data_path_11,input_path_flair)
     img_flair,header_flair = load(f_flair)

     f_seg = os.path.join(data_path_11,input_path_seg)
     img_seg,header_seg = load(f_seg)

     inputs_t1 = img_t1.astype(np.float32)
     inputs_t1ce = img_t1ce.astype(np.float32)
     inputs_t2 = img_t2.astype(np.float32)
     inputs_flair = img_flair.astype(np.float32)
     inputs_seg = img_seg.astype(np.uint8)
     inputs_seg = convert_label(inputs_seg)

     mask = inputs_t1>0

     inputs_tmp_t1 = (inputs_t1 - inputs_t1[mask].mean()) / inputs_t1[mask].std()
     inputs_tmp_t1ce = (inputs_t1ce - inputs_t1ce[mask].mean()) / inputs_t1ce[mask].std()
     inputs_tmp_t2 = (inputs_t2 - inputs_t2[mask].mean()) / inputs_t2[mask].std()
     inputs_tmp_flair = (inputs_flair - inputs_flair[mask].mean()) / inputs_flair[mask].std()

     margin = 64/2   # training_patch_size / 2
     mask = mask.astype(np.uint8)
     min_D_s, max_D_e, min_H_s, max_H_e, min_W_s, max_W_e = cut_edge(mask, margin)
     inputs_tmp_t1 = inputs_t1_norm[min_D_s:max_D_e + 1, min_H_s: max_H_e + 1, min_W_s:max_W_e + 1]
     inputs_tmp_t1ce = inputs_t1ce_norm[min_D_s:max_D_e + 1, min_H_s: max_H_e + 1, min_W_s:max_W_e + 1]
     inputs_tmp_t2 = inputs_t2_norm[min_D_s:max_D_e + 1, min_H_s: max_H_e + 1, min_W_s:max_W_e + 1]
     inputs_tmp_flair = inputs_flair_norm[min_D_s:max_D_e + 1, min_H_s: max_H_e + 1, min_W_s:max_W_e + 1]

     inputs_seg_tmp = inputs_seg[min_D_s:max_D_e + 1, min_H_s: max_H_e + 1, min_W_s:max_W_e + 1]

     
     inputs_tmp_t1 = inputs_tmp_t1[:, :, :, None]
     inputs_tmp_t1ce = inputs_tmp_t1ce[:, :, :, None]
     inputs_tmp_t2 = inputs_tmp_t2[:, :, :, None]
     inputs_tmp_flair = inputs_tmp_flair[:, :, :, None]

     inputs = np.concatenate((inputs_tmp_t1,inputs_tmp_t1ce,inputs_tmp_t2,inputs_tmp_flair),axis=3)
     inputs_caffe = inputs_tmp_t2#[None, :, :, :, :]
     inputs_seg_caffe = inputs_seg_tmp[None, :, :, :, :]
     
     inputs_caffe = inputs_caffe.transpose(0, 4, 3, 1, 2)
     inputs_seg_caffe = inputs_seg_caffe.transpose(0, 4, 3, 1, 2)

     print (inputs_caffe.shape, inputs_seg_caffe.shape)
     #print (inputs_caffe.shape)
     with h5py.File(os.path.join(target_path, 'train_brats_nocut_%s.h5' % (i+1)), 'w') as f:
         f['data'] = inputs_caffe  # for caffe num channel x d x h x w
         f['label'] = torch.Tensor([0])
     i = i+1   

 

if __name__ == '__main__':
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    build_h5_dataset(data_path, target_path)     




      


       








   




       


