# Anticipating Accidents in Dashcam Videos (Transformers approach)


### Introduction

Anticipating Accidents in Dashcam Videos is initially described in a [ACCV 2016 paper](https://drive.google.com/file/d/0ByuDEGFYmWsbNkVxcUxhdDRVRkU/view?usp=sharing&resourcekey=0-RsYavk2HgV_D-RXpUF7NEg).
This fork of the original project (from Fu-Hsiang Chan, Yu-Ting Chen, Yu Xiang, Min Sun) tries to implement a transformer in the model.

### Requirements

##### Tensoflow 2.x
##### Opencv
##### Matplotlib
##### Numpy

### Model Flowchart


### Dataset & Features

* Dataset : [link](http://aliensunmin.github.io/project/dashcam/) (Download the file and put it in "datatset/videos" folder.)

* CNN features : [link](https://drive.google.com/file/d/0B8xi2Pbo0n2gRGpzWUEzRTU2WUk/view?usp=sharing&resourcekey=0-e9lvHE70UAbFuVd79KWxZw) (Download the file and put it in "dataset/features" folder.)

* Annotation : [link](https://drive.google.com/file/d/0B8xi2Pbo0n2gdTlwT2NXdS1NTFE/view?usp=sharing&resourcekey=0-G5Vtj94Pdeiy1WJU84bcFA)

If you need the ground truth of object bounding box and accident location, you can download it.

The format of annotation:

<image name, track_ID, class , x1, y1, x2, y2, 0/1 (no accident/ has accident)>

### Usage

#### Run Demo

#### Training

#### Testing
