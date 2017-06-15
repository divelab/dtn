# Dense Transformer Networks

This is the tensorflow implementation of our recent work, "Dense Transformer Networks". Please check the [paper](https://arxiv.org/abs/1705.08881) for details.

## Introduction

In this work, we propose Dense Transformer Networks to apply spatial transformation to semantic prediction tasks. 
Dense Transformer Networks can extract features based on irregular areas, whose shapes and sizes are based on data.
In the meantime, Dense Transformer Networks provide a method that efficiently restores spatial relations.

## Citation
If using this code, please cite our paper.
```
@article{li2017dtn,
  title={Dense Transformer Networks},
  author={Jun Li and Yongjun Chen and Lei Cai and Ian Davidson and Shuiwang
Ji},
  journal={arXiv preprint arXiv:1705.08881},
  year={2017}
}
```


## Experimental results:
We perform our experiment on two datasets to compare the baseline U-Net model and the proposed DTN model.

1. PASCAL dataset

![image](https://github.com/divelab/dtn/blob/master/results/PASCALresult.png)
Sample segmentation results on the PASCAL 2012 segmentation data set. The first and
second rows are the original images and the corresponding ground truth, respectively. The third and
fourth rows are the segmentation results of U-Net and DTN, respectively.

2. SNEMI3D dataset

![image](https://github.com/divelab/dtn/blob/master/results/SNEMI3Dresult.PNG)


## Dense Transformer Networks

![image](https://github.com/divelab/dtn/blob/master/results/architecture.PNG)

## Configure the network

All network hyperparameters are configured in main.py.

#### Training

max_epoch: how many iterations or steps to train

test_step: how many steps to perform a mini test or validation

save_step: how many steps to save the model

summary_step: how many steps to save the summary

keep_prob: dropout probability

#### Validing

valid_start_epoch: start step to test a model

valid_end_epoch: end step to test a model

valid_stride_of_epoch: stride to test a model

#### Data

data_dir: data directory

train_data: h5 file for training

valid_data: h5 file for validation

test_data: h5 file for testing

batch: batch size

channel: input image channel number

height, width: height and width of input image

#### Debug

logdir: where to store log

modeldir: where to store saved models

sampledir: where to store predicted samples, please add a / at the end for convinience

model_name: the name prefix of saved models

reload_epoch: where to return training

test_epoch: which step to test or predict

random_seed: random seed for tensorflow

#### Network architecture

network_depth: how deep of the U-Net including the bottom layer

class_num: how many classes. Usually number of classes plus one for background

start_channel_num: the number of channel for the first conv layer


conv_name: use which convolutional layer in decoder. We have conv2d for standard convolutional layer, and ipixel_cl for input pixel convolutional layer proposed in our paper.

deconv_name: use which upsampling layer in decoder. We have deconv for standard deconvolutional layer, ipixel_dcl for input pixel deconvolutional layer, and pixel_dcl for pixel deconvolutional layer proposed in our paper.

#### Dense Transformer Networks

add_dtn: add Dense Transformer Netwroks or not.

dtn_location: The Dense Transformer Networks location.

control_points_ratio: the ratio of control_points comparing with the Dense transformer networks input size.

## Training and Testing

#### Start training

After configure the network, we can start to train. Run
```
python main.py
```
The training of a U-Net for semantic segmentation will start.

#### Training process visualization

We employ tensorboard to visualize the training process.

```
tensorboard --logdir=logdir/
```

The segmentation results including training and validation accuracies, and the prediction outputs are all available in tensorboard.

#### Testing and prediction

Select a good point to test your model based on validation result.

Fill the valid_start_epoch, valid_end_epoch and valid_stride_of_epoch in configure. Then run

```
python main.py --action=test

```
It will show the accuracy, loss and mean_iou at each epoch.

If you want to make some predictions, run

```
python main.py --action=predict

```
The predicted segmentation results will be in sampledir set in main.py, colored.

## Use Dense Transformer Networks 

If you want to use Dense Transformer Networks, just Fill the add_dtn, dtn_location and control_points_ratio in configure function.





