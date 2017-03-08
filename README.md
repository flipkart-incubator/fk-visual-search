# fk-visual-search
This code allows you to train the Visnet model. Visnet, trained on Flipkart's proprietary internal dataset, powers Visual Recommendations at Flipkart. On the publically available dataset, [Street2Shop](http://tamaraberg.com/street2shop/), Visnet achieves state-of-the-art results. [Here](https://arxiv.org/abs/1703.02344) is the link to the arXiv tech report.

In this Repo, we have open-sourced the following:
* Training prototxts of Visnet
* Triplet sampling code, to generate the training files
* A CUDA based fast K-Nearest Neighbor Search library
* Other auxillary scripts, such as code to process [Street2Shop](http://tamaraberg.com/street2shop/) dataset, sampling triplets, etc.


We soon plan to add other useful scripts, such as:
* Our useful modifications over Caffe - the image augmentation layer, and triplet accuracy layer to aid the training of Visnet

## Visnet Architecture
VisNet is a Convolutional Neural Network (CNN) trained using triplet based deep ranking paradigm. It contains a deep CNN modelled after the VGG-16 network, coupled with parallel shallow convolution layers in order to capture both high-level and low-level image details simultaneously.
![img](https://drive.google.com/uc?export=view&id=0B4toQpysgMLVd09nNEJEVWc4VmM)

## Training
In order to train you need a set of triplets <q,p,n>. For compatibility with Caffe's ImageData layer, you need 3 sets of triplet files (one each for q, p and n). The lines in those files should correspond to triplets, i.e. line#i in each file should correspond to the i'th triplet. 

If you wish to train Visnet on [Street2Shop](http://tamaraberg.com/street2shop/) dataset, you need to:

1) Download the [Street2Shop](http://tamaraberg.com/street2shop/) dataset (This contains only the image URLs)

2) Download Street2Shop images (Have a look at scripts/image_downloader.py)

3) You can then format the data using scripts/create_structured_images.py and scripts/create_wtbi_crops.py

4) Use scripts/sampler.py to sample the triplet files

5) Change visnet/train.prototxt to include the location to your triplet files

6) Run training using Caffe


## Feature extraction and NN Search
We provide PyCaffe code to do Feature Extraction (scripts/feature_extractor.py), and a CUDA-based fast NN computer (scripts/cuda_knn.py).
