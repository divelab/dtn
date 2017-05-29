# Dense Transformer Networks

This is the tensorflow implementation of our recent work, "Dense Transformer Networks". Please check the [paper](https://arxiv.org/abs/1705.08881) for details.

## Introduction

In this work, we propose Dense Transformer Networks to apply spatial transformation to semantic prediction tasks. 
Dense Transformer Networks can extract features based on irregular areas, whose shapes and sizes are based on data.
In the meantime, Dense Transformer Networks provide a method that efficiently restore spatial information.

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


## How to use

![image](https://github.com/divelab/dtn/blob/master/results/architecture.PNG)

### transformer(U,U_local,Column_controlP_number,Row_controlP_number,out_size)

*U: the input of spatial transformer.  
*U_local: the input of localization networks.  
*Column_controlP_number: the number of columns of the fiducial points.  
*Row_controlP_number: the number of rows of the fiducial points.  
*out_size: the size of output feature maps after spatial transformer. 

### inverse_transformer(U,Column_controlP_number,Row_controlP_number,out_size)

* U: the input of spatial transformer.  
* Column_controlP_number: the number of columns of the fiducial points.  
* Row_controlP_number: the number of rows of the fiducial points.  
* out_size: the size of output feature maps after spatial decoder transformer.







