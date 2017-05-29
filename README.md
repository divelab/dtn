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


## How to use

![image](https://github.com/divelab/dtn/blob/master/results/architecture.PNG)

### TPS_transformer

```
Parameters  

* U: the input of spatial transformer.  
* U_local: the input of localization networks.  
```

### TPS_decoder

```
Parameters  

* U: the input of spatial deocder transformer.  
* U_org: the original feature maps to fill the missing pixels.  
* T: the transformation shared with TPS_transformer. 
```
### A simple example

	U=tf.linspace(1.0,10.0,100)
	U =tf.reshape(U,[2,5,5,2])

	#encoder layer initial
	X_controlP_number = 4
	Y_controlP_number = 4
	tps_out_size = (40,40)
	#decoder layer initial
	X_controlP_number_D = 4
	Y_controlP_number_D = 4
	out_size_D = (40, 40)
	# encoder layer 
	transform = transformer(U,U,X_controlP_number,Y_controlP_number,tps_out_size)
	conv1,T,cp= transform.TPS_transformer(U,U)
	#decoder layer 
	inverse_trans = inverse_transformer(conv1,X_controlP_number_D,Y_controlP_number_D,out_size_D)
	conv2 = inverse_trans.TPS_decoder(conv1,conv1,T)






