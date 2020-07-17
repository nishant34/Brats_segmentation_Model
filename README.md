# Brats_segmenation_Model
Implementation of a Model architecture similar to focusnet for MICCAI Brats 2018 Challenge . 
The model is similar to a focusnet - a model for image segmentation .
The dataset is available on the MICCAI website as a part of Brats-2018 challenge .
The Code is organized as follows:
brats_data_prep: For preparing hd5 dataset from the given data .
dataloader: for loading the data for training .
model : architecture of the model used for segmentation .
train : Contains the code to train model on the given dataset .
val : for validation of metrics on particular subjects .
