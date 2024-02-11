# ⚠️ WIP ⚠️
# Anticipating Accidents in Dashcam Videos (Transformers approach)


### Introduction

Anticipating Accidents in Dashcam Videos is initially described in a [ACCV 2016 paper](https://drive.google.com/file/d/0ByuDEGFYmWsbNkVxcUxhdDRVRkU/view?usp=sharing&resourcekey=0-RsYavk2HgV_D-RXpUF7NEg).
This fork of the original project (from Fu-Hsiang Chan, Yu-Ting Chen, Yu Xiang, Min Sun) tries to implement a transformer-based model, instead of the LSTM-based model il the original project.

For the transformer implementation it was taken as a reference [this example](https://keras.io/examples/vision/video_transformers/#building-the-transformerbased-model) that uses a simplified transformer (encoder only) for video classification.

![simple](https://github.com/luibo/Anticipating-Accidents-with-Transformers/assets/26224625/b6b80378-db7f-435c-86ca-01f33ec3ee19)


### Requirements
All the required libraries are given with the env.yml file that can be used to set up a proper conda environment.

### Dataset & Features

* Dataset : [link](http://aliensunmin.github.io/project/dashcam/) (Download the file and put it in "datatset/videos" folder.)

* CNN features : [link](https://drive.google.com/file/d/0B8xi2Pbo0n2gRGpzWUEzRTU2WUk/view?usp=sharing&resourcekey=0-e9lvHE70UAbFuVd79KWxZw) (Download the file and put it in "dataset/features" folder.)

* Annotation : [link](https://drive.google.com/file/d/0B8xi2Pbo0n2gdTlwT2NXdS1NTFE/view?usp=sharing&resourcekey=0-G5Vtj94Pdeiy1WJU84bcFA)

If you need the ground truth of object bounding box and accident location, you can download it.

The format of annotation:

<image name, track_ID, class , x1, y1, x2, y2, 0/1 (no accident/ has accident)>

### Usage
It's adviced to create a conda environment using env.yml to have all dependencies ready.
```
conda env create -f env.yml
conda activate aat
```

For training:
```
python accident.py --mode train
```

For testing:
```
python accident.py --mode test
```

### Performances
Model performances are far from optimal, a deeper study of the problem should be done to improve its gerenalization capabilities.
The metrics that were measured:
- Accuracy: 71.36%
- Precision: 54.74%
- Recall: 67.54%

