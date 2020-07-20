# Brats_segmentation_Model
Implementation of a Model architecture similar to focusnet for MICCAI Brats 2018 Challenge . 
The model is similar to a model named focusnet - a model for image segmentation . 
The dataset is available on the MICCAI website as a part of Brats-2018 challenge . 
The code is structured as follows:
* prepare_data : to convert data from .nii.gz format to .h5 format with all 4 types of scans(t1,t2,t1ce,flair) into a single .h5 file .
* dataloader.py : to load the converted data .
* model.py : contains code of  the focusnet model used for segmentation .
* train.py : contains the code for training the model on  data using 64x64x64 portions of the given MRI scan.
* val.py : contains the code to validate the model for whole of a particular example (by selecting 64x64x64 regions with a stride and selecting most frequent class for overlapping regions).
* metrics.py : contains the code for metrics for analysing the performance . 
* common.py : contains the values of hyperparams for triaining along with functions useful while training .



